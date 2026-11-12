"""
Backtest report: one results dict (written as results.json) and one static
HTML page that renders it with Plotly.

Every number on the page is read from the embedded results; nothing is typed
by hand. Views: fill-timing impact, walk-forward equity, Sharpe uncertainty,
cost sensitivity and the engine benchmark.
"""

from __future__ import annotations

import html
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from quantsim.backtesting.engine import BacktestEngine
from quantsim.backtesting.execution import FillModel, NoSlippage, ZeroCommission
from quantsim.backtesting.vectorized import (
    VectorizedBacktester,
    momentum_signal,
    sma_crossover_signal,
)
from quantsim.backtesting.walk_forward import WalkForwardOptimizer
from quantsim.core.event_queue import EventQueue
from quantsim.data.synthetic import write_prices_to_db
from quantsim.reporting.advanced import AdvancedAnalytics
from quantsim.reporting.analytics import TRADING_DAYS_PER_YEAR
from quantsim.strategies.trend import SMAcrossover, TimeSeriesMomentum

PLOTLY_URL = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
INITIAL_CAPITAL = 100_000.0
POSITION_SIZE = 1.0
BOOTSTRAP_SIMULATIONS = 5000

STRATEGIES: Dict[str, Dict[str, Any]] = {
    "sma": {
        "label": "SMA crossover",
        "signal": sma_crossover_signal,
        "grid": {"fast": [20, 50], "slow": [100, 200]},
        "event": lambda queue, symbol, p: SMAcrossover(
            [symbol], queue, fast=p["fast"], slow=p["slow"]
        ),
    },
    "momentum": {
        "label": "Time-series momentum",
        "signal": momentum_signal,
        "grid": {"lookback": [126, 252], "skip": [21]},
        "event": lambda queue, symbol, p: TimeSeriesMomentum(
            [symbol],
            queue,
            lookback_months=round(p["lookback"] / 21),
            skip_months=round(p["skip"] / 21),
        ),
    },
}


def _dates(index) -> List[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in index]


def _round(values, digits=2) -> List[float]:
    return [round(float(v), digits) for v in values]


def _annualized_sharpe(returns: pd.Series) -> float:
    std = returns.std()
    return float(returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR)) if std > 0 else 0.0


def fill_timing_view(prices: pd.Series, strategy: str, params: Dict) -> Dict:
    """Run the event engine with next-bar-open fills and with the old same-bar fills."""
    frame = prices.to_frame()
    curves = {}
    trade_counts = {}
    with tempfile.TemporaryDirectory() as tmp:
        for mode in (FillModel.NEXT_BAR_OPEN, FillModel.SAME_BAR_OPEN_LOOKAHEAD):
            db_path = str(Path(tmp) / f"{mode}.db")
            write_prices_to_db(db_path, frame)
            queue = EventQueue()
            engine = BacktestEngine(
                strategies=[STRATEGIES[strategy]["event"](queue, prices.name, params)],
                start=prices.index[0].to_pydatetime(),
                end=prices.index[-1].to_pydatetime(),
                initial_capital=INITIAL_CAPITAL,
                slippage_model=NoSlippage(),
                commission_model=ZeroCommission(),
                fill_mode=mode,
                warmup_bars=0,
                db_path=db_path,
                verbose=False,
            )
            engine.run()
            history = pd.Series(
                [equity for _, equity in engine.portfolio._equity_history],
                index=pd.DatetimeIndex(
                    [ts for ts, _ in engine.portfolio._equity_history]
                ).normalize(),
            )
            curves[mode] = history[~history.index.duplicated(keep="last")]
            trade_counts[mode] = engine._fills_processed

    honest, lookahead = curves[FillModel.NEXT_BAR_OPEN], curves[FillModel.SAME_BAR_OPEN_LOOKAHEAD]
    common = honest.index.intersection(lookahead.index)
    honest, lookahead = honest.loc[common], lookahead.loc[common]
    honest_return = float(honest.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    lookahead_return = float(lookahead.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    gap = round(lookahead_return - honest_return, 1)
    return {
        "strategy_used": f"{STRATEGIES[strategy]['label']} {params}",
        "dates": _dates(common),
        "next_bar_open": _round(honest),
        "same_bar_open": _round(lookahead),
        "trade_count": int(trade_counts[FillModel.NEXT_BAR_OPEN]),
        "next_bar_open_return_pct": round(honest_return, 1),
        "same_bar_open_return_pct": round(lookahead_return, 1),
        "gap_pp": gap,
        "takeaway": (
            f"Same-bar fills {'overstated' if gap >= 0 else 'understated'} total return by "
            f"{abs(gap)} pp ({round(lookahead_return, 1)}% vs {round(honest_return, 1)}% "
            "with next-bar-open fills)."
        ),
    }


def walk_forward_view(prices: pd.Series, strategy: str, cost: float, seed: int) -> Dict:
    spec = STRATEGIES[strategy]
    frame = prices.to_frame()
    wfo = WalkForwardOptimizer(frame, train_years=3, test_months=12, step_months=12)
    result = wfo.optimize_and_evaluate(
        spec["signal"], spec["grid"], commission_rate=cost, slippage_pct=0.0
    )
    windows = []
    stitched: List[pd.Series] = []
    level = INITIAL_CAPITAL
    rng = np.random.default_rng(seed)
    advanced = AdvancedAnalytics()
    for row in result["window_results"]:
        equity = row["oos_equity"] / row["oos_equity"].iloc[0] * level
        level = float(equity.iloc[-1])
        stitched.append(equity if not stitched else equity.iloc[1:])
        returns = row["oos_equity"].pct_change().dropna()
        interval = advanced.monte_carlo_sharpe(
            returns, n_simulations=2000, risk_free_rate=0.0, seed=rng
        )
        windows.append(
            {
                "label": f"{row['test_start']:%Y-%m-%d} to {row['test_end']:%Y-%m-%d}",
                "train_start": f"{row['train_start']:%Y-%m-%d}",
                "train_end": f"{row['train_end']:%Y-%m-%d}",
                "test_start": f"{row['test_start']:%Y-%m-%d}",
                "test_end": f"{row['test_end']:%Y-%m-%d}",
                "params": row["best_params"],
                "oos_sharpe": round(_annualized_sharpe(returns), 2),
                "ci_low": round(float(interval["sharpe_ci_low"]), 2),
                "ci_high": round(float(interval["sharpe_ci_high"]), 2),
            }
        )
    oos = pd.concat(stitched)
    hold = prices.loc[oos.index] / prices.loc[oos.index[0]] * INITIAL_CAPITAL
    oos_return = round(float(oos.iloc[-1] / INITIAL_CAPITAL - 1) * 100, 1)
    hold_return = round(float(hold.iloc[-1] / INITIAL_CAPITAL - 1) * 100, 1)
    return {
        "dates": _dates(oos.index),
        "oos_equity": _round(oos),
        "buy_and_hold": _round(hold),
        "windows": windows,
        "n_configs": result["n_configs_tested"],
        "total_trials": result["total_configs_tested"],
        "dsr": round(float(result["deflated_sharpe_corrected"]), 3),
        "oos_equity_series": oos,
        "oos_return_pct": oos_return,
        "buy_and_hold_return_pct": hold_return,
        "takeaway": f"Out of sample the strategy returned {oos_return}% against "
        f"{hold_return}% for buy-and-hold over {len(windows)} one-year test windows.",
    }


def sharpe_uncertainty_view(oos_equity: pd.Series, dsr: float, trials: int, seed: int) -> Dict:
    returns = oos_equity.pct_change().dropna()
    interval = AdvancedAnalytics.monte_carlo_sharpe(
        returns,
        n_simulations=BOOTSTRAP_SIMULATIONS,
        risk_free_rate=0.0,
        seed=seed,
        return_samples=True,
    )
    counts, edges = np.histogram(interval["samples"], bins=40)
    point = round(float(interval["sharpe_point_estimate"]), 2)
    low, high = (
        round(float(interval["sharpe_ci_low"]), 2),
        round(float(interval["sharpe_ci_high"]), 2),
    )
    verdict = "significant" if dsr > 0.95 else "not significant"
    return {
        "bin_edges": _round(edges, 3),
        "counts": [int(c) for c in counts],
        "point": point,
        "ci_low": low,
        "ci_high": high,
        "dsr": dsr,
        "n_simulations": BOOTSTRAP_SIMULATIONS,
        "takeaway": f"Out-of-sample Sharpe {point}, 95% interval {low} to {high}; "
        f"DSR {dsr} after {trials} trials ({verdict} at 0.95).",
    }


def cost_sensitivity_view(
    prices: pd.Series, strategy: str, params: Dict, costs_bps: List[float]
) -> Dict:
    sharpes = []
    for bps in costs_bps:
        backtester = VectorizedBacktester(
            prices.to_frame(),
            commission_rate=bps / 10_000,
            slippage_pct=0.0,
            position_size_pct=POSITION_SIZE,
        )
        result = backtester.run(STRATEGIES[strategy]["signal"], params)
        returns = pd.Series(result["_equity_curve"]).pct_change().dropna()
        sharpes.append(round(_annualized_sharpe(returns), 2))
    positive = [c for c, s in zip(costs_bps, sharpes) if s > 0]
    if len(positive) == len(costs_bps):
        takeaway = f"Sharpe stays positive up to {costs_bps[-1]:g} bps per trade ({sharpes[-1]})."
    elif not positive:
        takeaway = f"Sharpe is not positive even at {costs_bps[0]:g} bps ({sharpes[0]})."
    else:
        takeaway = f"Sharpe turns negative above {positive[-1]:g} bps per trade."
    return {
        "params": params,
        "cost_bps": [float(c) for c in costs_bps],
        "sharpe": sharpes,
        "takeaway": takeaway,
    }


def benchmark_view(path: Optional[Path]) -> Optional[Dict]:
    if path is None or not path.exists():
        return None
    bench = json.loads(path.read_text())
    largest = bench["rows"][-1]
    bench["takeaway"] = (
        f"On 10 years of daily bars for {largest['assets']} assets the vectorized engine ran in "
        f"{largest['vectorized_median_s']:.2f} s and the event engine in "
        f"{largest['event_median_s']:.1f} s ({largest['speedup']:.0f}x)."
    )
    return bench


def build_results(
    prices: pd.Series,
    strategies: List[str],
    costs_bps: List[float],
    data_info: Dict,
    meta: Dict,
    benchmark_path: Optional[Path] = None,
    seed: int = 0,
) -> Dict:
    """Run every view for each strategy on one price series and collect the results."""
    results: Dict[str, Any] = {"meta": {**meta, "data": data_info}, "strategies": {}}
    base_cost = costs_bps[0] / 10_000
    for name in strategies:
        wf = walk_forward_view(prices, name, base_cost, seed)
        oos_series = wf.pop("oos_equity_series")
        params = most_common_params(wf["windows"])
        results["strategies"][name] = {
            "label": STRATEGIES[name]["label"],
            "params": params,
            "fill_timing": fill_timing_view(prices, name, params),
            "walk_forward": wf,
            "sharpe_uncertainty": sharpe_uncertainty_view(
                oos_series, wf["dsr"], wf["total_trials"], seed
            ),
            "cost_sensitivity": cost_sensitivity_view(prices, name, params, costs_bps),
        }
        results["strategies"][name]["trade_count"] = results["strategies"][name]["fill_timing"][
            "trade_count"
        ]
    results["benchmark"] = benchmark_view(benchmark_path)
    return results


def most_common_params(windows: List[Dict]) -> Dict:
    keys = [json.dumps(w["params"], sort_keys=True) for w in windows]
    return json.loads(max(set(keys), key=keys.count))


def render_html(results: Dict) -> str:
    data = results["meta"]["data"]
    if data.get("synthetic"):
        banner = (
            f"Synthetic data, seed {data['seed']}, generated by {data['generator']}. "
            "These are not market prices."
        )
    else:
        banner = f"{data['source']}, {data['symbol']}, {data['start']} to {data['end']}."
    payload = json.dumps(results, separators=(",", ":")).replace("</", "<\\/")
    return (
        PAGE_TEMPLATE.replace("__BANNER__", html.escape(banner))
        .replace("__PLOTLY_URL__", PLOTLY_URL)
        .replace("__RESULTS__", payload)
    )


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>quantsim backtest report</title>
<style>
:root { --bg:#ffffff; --fg:#1d1d1f; --muted:#5f6368; --line:#d8dadf; --panel:#f6f7f9; --warn:#7a4b00; --warnbg:#fff4dc; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#16181c; --fg:#e8e8ea; --muted:#a0a4ab; --line:#33363d; --panel:#1f2228; --warn:#ffd48a; --warnbg:#3a2e16; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:960px; margin:0 auto; padding:16px; }
h1 { font-size:1.5rem; margin:0.5rem 0; }
h2 { font-size:1.15rem; margin:0 0 0.25rem; }
.banner { background:var(--warnbg); color:var(--warn); border-radius:6px; padding:8px 12px; }
.meta { color:var(--muted); font-size:0.875rem; overflow-wrap:anywhere; }
section { border-top:1px solid var(--line); padding:16px 0; }
.takeaway { margin:0 0 8px; }
.chart { width:100%; height:360px; }
.error { color:#b3261e; background:var(--panel); padding:8px 12px; border-radius:6px; }
.empty { color:var(--muted); background:var(--panel); padding:8px 12px; border-radius:6px; }
label { font-weight:600; margin-right:8px; }
select { font:inherit; padding:4px 8px; background:var(--panel); color:var(--fg); border:1px solid var(--line); border-radius:4px; }
table { border-collapse:collapse; width:100%; font-size:0.875rem; }
td, th { border-bottom:1px solid var(--line); padding:4px 6px; text-align:left; }
.table-wrap { overflow-x:auto; }
</style>
</head>
<body>
<main>
<h1>quantsim backtest report</h1>
<p class="banner" id="banner">__BANNER__</p>
<p class="meta" id="meta"></p>
<p id="selector-row" hidden><label for="strategy">Strategy</label><select id="strategy"></select></p>
<section id="view-fill"><h2>Fill-timing impact</h2><p class="takeaway"></p><div class="chart"></div></section>
<section id="view-wfo"><h2>Walk-forward out-of-sample equity</h2><p class="takeaway"></p><div class="chart"></div><div class="table-wrap"></div></section>
<section id="view-sharpe"><h2>Sharpe uncertainty</h2><p class="takeaway"></p><div class="chart"></div></section>
<section id="view-cost"><h2>Cost sensitivity</h2><p class="takeaway"></p><div class="chart"></div></section>
<section id="view-bench"><h2>Engine benchmark</h2><p class="takeaway"></p><div class="chart"></div></section>
</main>
<script type="application/json" id="results">__RESULTS__</script>
<script src="__PLOTLY_URL__"></script>
<script>
(function () {
  const results = JSON.parse(document.getElementById("results").textContent);
  const okabe = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"];
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const fg = dark ? "#e8e8ea" : "#1d1d1f";
  const grid = dark ? "#33363d" : "#d8dadf";
  const config = { responsive: true, displaylogo: false };

  function layout(extra) {
    return Object.assign({
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: fg, size: 12 }, margin: { l: 64, r: 16, t: 8, b: 48 },
      legend: { orientation: "h", y: -0.2 }, hovermode: "x unified",
    }, extra);
  }
  function axis(title, extra) { return Object.assign({ title: { text: title }, gridcolor: grid, zerolinecolor: grid }, extra || {}); }
  function need(obj, key, view) {
    if (obj === null || obj === undefined || !(key in obj)) throw new Error("results.json has no '" + key + "' for " + view);
    return obj[key];
  }
  function render(id, fn) {
    const section = document.getElementById(id);
    const chart = section.querySelector(".chart");
    try { fn(section, chart); }
    catch (err) {
      chart.innerHTML = "";
      chart.style.height = "auto";
      const box = document.createElement("p");
      box.className = "error";
      box.textContent = "This view failed: " + err.message;
      chart.appendChild(box);
    }
  }
  function empty(chart, text) {
    chart.style.height = "auto";
    chart.innerHTML = "<p class='empty'></p>";
    chart.firstChild.textContent = text;
  }

  const meta = results.meta || {};
  document.getElementById("meta").textContent =
    "Generated " + meta.generated_at + " from commit " + meta.sha + ". Command: " + meta.command;

  const names = Object.keys(results.strategies || {});
  const select = document.getElementById("strategy");
  names.forEach(function (n) {
    const opt = document.createElement("option");
    opt.value = n; opt.textContent = results.strategies[n].label; select.appendChild(opt);
  });
  if (names.length > 1) document.getElementById("selector-row").hidden = false;

  function draw(name) {
    const s = results.strategies[name];
    render("view-fill", function (section, chart) {
      const v = need(s, "fill_timing", "fill timing");
      section.querySelector(".takeaway").textContent = need(v, "takeaway", "fill timing");
      if (need(s, "trade_count", "fill timing") === 0) return empty(chart, "No trades in this run.");
      Plotly.react(chart, [
        { x: v.dates, y: v.next_bar_open, name: "Next-bar-open fills (current)", mode: "lines", line: { color: okabe[0], width: 2 }, hovertemplate: "%{y:$,.0f}" },
        { x: v.dates, y: v.same_bar_open, name: "Same-bar fills (pre-3.0 look-ahead)", mode: "lines", line: { color: okabe[1], width: 2, dash: "dash" }, hovertemplate: "%{y:$,.0f}" },
      ], layout({ xaxis: axis("Date"), yaxis: axis("Equity (USD, log scale)", { type: "log", tickprefix: "$" }) }), config);
    });
    render("view-wfo", function (section, chart) {
      const v = need(s, "walk_forward", "walk-forward");
      section.querySelector(".takeaway").textContent = need(v, "takeaway", "walk-forward");
      const shapes = need(v, "windows", "walk-forward").map(function (w) {
        return { type: "rect", xref: "x", yref: "paper", x0: w.train_start, x1: w.train_end, y0: 0, y1: 1, fillcolor: grid, opacity: 0.12, line: { width: 0 }, layer: "below" };
      });
      Plotly.react(chart, [
        { x: v.dates, y: v.oos_equity, name: "Out-of-sample equity", mode: "lines", line: { color: okabe[2], width: 2 }, hovertemplate: "%{y:$,.0f}" },
        { x: v.dates, y: v.buy_and_hold, name: "Buy and hold", mode: "lines", line: { color: okabe[3], width: 2, dash: "dot" }, hovertemplate: "%{y:$,.0f}" },
      ], layout({ shapes: shapes, xaxis: axis("Date"), yaxis: axis("Equity (USD)", { tickprefix: "$" }) }), config);
      const rows = v.windows.map(function (w) {
        return "<tr><td>" + w.label + "</td><td>" + JSON.stringify(w.params) + "</td><td>" + w.oos_sharpe + "</td><td>" + w.ci_low + " to " + w.ci_high + "</td></tr>";
      }).join("");
      section.querySelector(".table-wrap").innerHTML =
        "<table><thead><tr><th>Test window</th><th>Chosen parameters</th><th>OOS Sharpe</th><th>95% interval</th></tr></thead><tbody>" + rows + "</tbody></table>";
    });
    render("view-sharpe", function (section, chart) {
      const v = need(s, "sharpe_uncertainty", "Sharpe uncertainty");
      section.querySelector(".takeaway").textContent = need(v, "takeaway", "Sharpe uncertainty");
      const edges = need(v, "bin_edges", "Sharpe uncertainty");
      const mids = edges.slice(0, -1).map(function (e, i) { return (e + edges[i + 1]) / 2; });
      const top = Math.max.apply(null, v.counts);
      const rule = function (x, label, dash, width) {
        return { x: [x, x], y: [0, top], mode: "lines", name: label, line: { color: fg, dash: dash, width: width }, hoverinfo: "skip" };
      };
      Plotly.react(chart, [
        { x: mids, y: v.counts, type: "bar", name: "Bootstrap samples", marker: { color: okabe[4] }, hovertemplate: "Sharpe %{x:.2f}: %{y} samples<extra></extra>" },
        rule(v.point, "Point estimate " + v.point, "solid", 2),
        rule(v.ci_low, "95% interval " + v.ci_low + " to " + v.ci_high, "dash", 1.5),
        Object.assign(rule(v.ci_high, "", "dash", 1.5), { showlegend: false }),
      ], layout({ hovermode: "closest", bargap: 0.05, xaxis: axis("Annualized Sharpe ratio"), yaxis: axis("Bootstrap samples (count)") }), config);
    });
    render("view-cost", function (section, chart) {
      const v = need(s, "cost_sensitivity", "cost sensitivity");
      section.querySelector(".takeaway").textContent = need(v, "takeaway", "cost sensitivity");
      Plotly.react(chart, [
        { x: v.cost_bps, y: v.sharpe, name: "Annualized Sharpe", mode: "lines+markers", line: { color: okabe[5], width: 2 }, marker: { symbol: "diamond", size: 9 }, hovertemplate: "%{x} bps: Sharpe %{y}<extra></extra>" },
      ], layout({ hovermode: "closest", showlegend: false, shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: fg, dash: "dash", width: 1 } }], xaxis: axis("Cost per trade (bps)"), yaxis: axis("Annualized Sharpe ratio") }), config);
    });
  }

  render("view-bench", function (section, chart) {
    const v = results.benchmark;
    if (!v) return empty(chart, "No benchmark results in this run. Run benchmarks/bench_engines.py first.");
    section.querySelector(".takeaway").textContent = need(v, "takeaway", "benchmark");
    const assets = v.rows.map(function (r) { return r.assets === 1 ? "1 asset" : String(r.assets) + " assets"; });
    const narrow = window.innerWidth < 600;
    Plotly.react(chart, [
      { x: assets, y: v.rows.map(function (r) { return r.vectorized_median_s; }), type: "bar", name: "Vectorized", marker: { color: okabe[0] }, text: v.rows.map(function (r) { return "Vectorized " + r.vectorized_median_s.toFixed(3) + " s"; }), textposition: narrow ? "none" : "outside" },
      { x: assets, y: v.rows.map(function (r) { return r.event_median_s; }), type: "bar", name: "Event-driven", marker: { color: okabe[1], pattern: { shape: "/" } }, text: v.rows.map(function (r) { return "Event " + r.event_median_s.toFixed(1) + " s"; }), textposition: narrow ? "none" : "outside" },
    ], layout({ barmode: "group", hovermode: "closest", xaxis: axis("Asset count"), yaxis: axis("Median seconds, log scale", { type: "log" }) }), config);
  });

  if (names.length === 0) {
    ["view-fill", "view-wfo", "view-sharpe", "view-cost"].forEach(function (id) {
      render(id, function () { throw new Error("results.json has no strategies"); });
    });
  } else {
    select.addEventListener("change", function () { draw(select.value); });
    draw(names[0]);
  }
})();
</script>
</body>
</html>
"""
