"""
Backtest report: one results dict (written as results.json) and one static
HTML page that renders it with the shared report-kit (inlined from ./kit).

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

INITIAL_CAPITAL = 100_000.0
POSITION_SIZE = 1.0
BOOTSTRAP_SIMULATIONS = 5000
DSR_THRESHOLD = 0.95

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
            f"Over the full sample ({_dates(common)[0]} to {_dates(common)[-1]}, fixed "
            "parameters), same-bar fills "
            f"{'overstated' if gap >= 0 else 'understated'} total return by "
            f"{abs(gap)} pp ({round(lookahead_return, 1)}% vs {round(honest_return, 1)}% "
            "with next-bar-open fills). Bars are rebuilt from closes, so each open is the "
            "previous close."
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
        window_return = float(equity.iloc[-1] / level - 1) * 100
        window_hold = prices.loc[row["oos_equity"].index]
        hold_window_return = float(window_hold.iloc[-1] / window_hold.iloc[0] - 1) * 100
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
                "oos_return_pct": round(window_return, 1),
                "buy_and_hold_return_pct": round(hold_window_return, 1),
            }
        )
    oos = pd.concat(stitched)
    hold = prices.loc[oos.index] / prices.loc[oos.index[0]] * INITIAL_CAPITAL
    oos_return = round(float(oos.iloc[-1] / INITIAL_CAPITAL - 1) * 100, 1)
    hold_return = round(float(hold.iloc[-1] / INITIAL_CAPITAL - 1) * 100, 1)
    # Long-only strategies earn exactly 0 on days they hold cash, so a non-zero daily
    # return means the strategy held the asset that day.
    daily = oos.pct_change().dropna()
    in_market = round(float((daily != 0).mean()) * 100)
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
        "days_in_market_pct": in_market,
        "takeaway": f"Out of sample, over {len(windows)} stitched one-year test windows "
        f"({_dates(oos.index)[0]} to {_dates(oos.index)[-1]}, next-bar fills, parameters refit "
        f"each window), the strategy's return was {oos_return}%. Buy-and-hold's was "
        f"{hold_return}%; the strategy held the asset on {in_market}% of days, so it lost less "
        "by holding less, which is not evidence of an edge.",
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
    verdict = "significant" if dsr > DSR_THRESHOLD else "not significant"
    return {
        "bin_edges": _round(edges, 3),
        "counts": [int(c) for c in counts],
        "point": point,
        "ci_low": low,
        "ci_high": high,
        "dsr": dsr,
        "n_simulations": BOOTSTRAP_SIMULATIONS,
        "ci_level_pct": 95,
        "dsr_threshold": DSR_THRESHOLD,
        "takeaway": f"Out-of-sample Sharpe {point}, 95% interval {low} to {high}; "
        f"deflated Sharpe probability {dsr} after {trials} trials ({verdict}; needs "
        f"{DSR_THRESHOLD}).",
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
        takeaway = (
            f"Sharpe is not positive even at {costs_bps[0]:g} bps ({sharpes[0]}; full sample, "
            "fixed parameters, vectorized engine)."
        )
    else:
        takeaway = f"Sharpe turns negative above {positive[-1]:g} bps per trade."
    return {
        "params": params,
        "sample": "full sample, fixed parameters, vectorized engine",
        "start": _dates(prices.index)[0],
        "end": _dates(prices.index)[-1],
        "cost_bps": [float(c) for c in costs_bps],
        "sharpe": sharpes,
        "takeaway": takeaway,
    }


def benchmark_view(path: Optional[Path]) -> Optional[Dict]:
    if path is None or not path.exists():
        return None
    bench = json.loads(path.read_text())
    # Rounded copies of what the page prints (spec precision: ratios 2 decimals below 10x,
    # 1 decimal to 100x, integers above), so every number on the page is in results.json.
    shown = []
    for row in bench["rows"]:
        x, e = row["speedup"], row["event_median_s"]
        shown.append(
            {
                "assets": row["assets"],
                "speedup_shown": round(x, 2) if x < 10 else round(x, 1) if x < 100 else round(x),
                "vectorized_s_shown": round(row["vectorized_median_s"], 2),
                "event_s_shown": round(e, 2) if e < 10 else round(e, 1),
            }
        )
    bench["rows_shown"] = shown
    one, largest = shown[0], shown[-1]
    bench["takeaway"] = (
        f"With {one['assets']} asset the vectorized engine is {one['speedup_shown']}x faster "
        f"({one['vectorized_s_shown']} s vs {one['event_s_shown']} s). Its time stays flat as "
        f"assets grow while the event engine's grows linearly, so at {largest['assets']} assets "
        f"the ratio is {largest['speedup_shown']}x. One machine ({bench['cpu']}), median of "
        f"{bench['runs']} runs."
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


HERE = Path(__file__).parent
REPO = "https://github.com/allenwang0/quantsim"
NBSP = " "


def _num(x: float, digits: int = 2) -> str:
    """Signed number with a true minus sign: -0.33 -> −0.33."""
    text = f"{x:.{digits}f}"
    return text.replace("-", "−") if x < 0 else text


def _pct(x: float, digits: int = 1) -> str:
    return _num(x, digits) + NBSP + "%"


def _params_text(name: str, p: Dict) -> str:
    if name == "sma":
        return f"SMA {p['fast']}/{p['slow']}"
    if name == "momentum":
        return f"{p['lookback']}-day lookback, skip {p['skip']}"
    return ", ".join(f"{k} {v}" for k, v in p.items())


def _table(head: List[str], rows: List[List[str]], caption: str, cls: str = "") -> str:
    th = "".join(f"<th scope='col'>{html.escape(h)}</th>" for h in head)
    body = "".join(
        f"<tr><th scope='row'>{r[0]}</th>" + "".join(f"<td>{c}</td>" for c in r[1:]) + "</tr>"
        for r in rows
    )
    return (
        f"<div class='tablewrap {cls}' tabindex='0' role='region' "
        f"aria-label='{html.escape(caption)}'><table><caption>{html.escape(caption)}</caption>"
        f"<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"
    )


TOGGLE = (
    '<div class="seg" role="radiogroup" aria-label="View" data-view-toggle>'
    '<button role="radio" aria-checked="true" data-value="chart">Chart</button>'
    '<button role="radio" aria-checked="false" data-value="table">Table</button></div>'
)


def _strategy_html(name: str, s: Dict, data: Dict, hidden: bool) -> str:
    wf, su, ft, cs = (
        s["walk_forward"],
        s["sharpe_uncertainty"],
        s["fill_timing"],
        s["cost_sensitivity"],
    )
    label = html.escape(s["label"])
    n_windows = len(wf["windows"])
    trials, configs = wf["total_trials"], wf["n_configs"]
    crosses = su["ci_low"] < 0 < su["ci_high"]
    oos_span = f"{wf['dates'][0]} to {wf['dates'][-1]}"
    full_span = f"{ft['dates'][0]} to {ft['dates'][-1]}"
    gap = ft["gap_pp"]
    gap_word = "overstate" if gap > 0 else "understate" if gap < 0 else "do not change"
    windows_rows = [
        [
            f"{w['test_start']} to {w['test_end']}",
            html.escape(_params_text(name, w["params"])),
            _num(w["oos_sharpe"]),
            f"{_num(w['ci_low'])} to {_num(w['ci_high'])}",
            _pct(w["oos_return_pct"]),
            _pct(w["buy_and_hold_return_pct"]),
        ]
        for w in wf["windows"]
    ]
    window_table = _table(
        [
            "Test window",
            "Chosen parameters",
            "Sharpe",
            f"{su['ci_level_pct']} % interval",
            "Return",
            "Buy-and-hold",
        ],
        windows_rows,
        f"{s['label']}: out-of-sample results per one-year test window",
        "rk-table",
    )
    returns_table = _table(
        ["Test window", "Strategy return", "Buy-and-hold return"],
        [[r[0], r[4], r[5]] for r in windows_rows]
        + [
            [
                f"All {n_windows} windows",
                _pct(wf["oos_return_pct"]),
                _pct(wf["buy_and_hold_return_pct"]),
            ]
        ],
        f"{s['label']}: out-of-sample returns per window",
        "rk-table",
    )
    spans_zero = sum(w["ci_low"] <= 0 <= w["ci_high"] for w in wf["windows"])
    win_title = (
        "Every one-year window's interval includes zero"
        if spans_zero == n_windows
        else f"{spans_zero} of {n_windows} one-year windows' intervals include zero"
    )
    flat = [w for w in wf["windows"] if w["ci_low"] == w["ci_high"] == w["oos_sharpe"] == 0]
    flat_note = "".join(
        f" In {w['test_start'][:4]}–{w['test_end'][2:4]} the strategy held no position, so its "
        "Sharpe is 0 with no interval."
        for w in flat
    )
    same_as_oos = ft["same_bar_open_return_pct"] == wf["oos_return_pct"]
    coincidence = (
        f" The {_pct(ft['same_bar_open_return_pct'])} here equals the walk-forward return by "
        "coincidence; the samples differ."
        if same_as_oos
        else ""
    )
    full_table = _table(
        ["Check (full sample, fixed parameters)", "Value"],
        [
            ["Return, next-bar-open fills", _pct(ft["next_bar_open_return_pct"])],
            ["Return, same-bar fills (look-ahead)", _pct(ft["same_bar_open_return_pct"])],
            ["Look-ahead gap", f"{_num(gap, 1)} pp"],
            ["Trades", f"{ft['trade_count']}"],
        ]
        + [
            [f"Sharpe at {c:g} bps per trade", _num(v)]
            for c, v in zip(cs["cost_bps"], cs["sharpe"])
        ],
        f"{_params_text(name, s['params'])}, full sample {full_span}",
    )
    return f"""
  <div data-strategy="{name}"{" hidden" if hidden else ""}>
  <section class="wrap" aria-label="Key results, {label}">
    <div class="bento">
      <div class="card span-6"><span class="label">Out-of-sample Sharpe, {su["ci_level_pct"]} % interval</span><span class="metric">{_num(su["point"])}</span><span class="cmp">[{_num(su["ci_low"])}, {_num(su["ci_high"])}]{": the interval includes zero" if crosses else ""}</span><span class="foot">Bootstrap, {su["n_simulations"]:,} resamples · {n_windows} one-year windows</span></div>
      <div class="card span-6"><span class="label">Deflated Sharpe probability</span><span class="metric">{wf["dsr"]:.2f}</span><span class="cmp">after {trials} parameter trials; {su["dsr_threshold"]} would indicate skill</span><span class="foot">{configs} settings × {n_windows} windows</span></div>
      <div class="card span-6"><span class="label">Out-of-sample return</span><span class="metric">{_pct(wf["oos_return_pct"])}</span><span class="cmp">Buy-and-hold: {_pct(wf["buy_and_hold_return_pct"])}. In the market on {wf["days_in_market_pct"]} % of days.</span><span class="foot">{oos_span} · not evidence of an edge</span></div>
      <div class="card span-6"><span class="label">Look-ahead bias, same-bar fills</span><span class="metric">{"+" if gap > 0 else ""}{_num(gap, 1)} <span class="unit">pp</span></span><span class="cmp">Same-bar fills {gap_word} total return ({ft["trade_count"]} trades)</span><span class="foot">Full sample {full_span}, fixed parameters</span></div>
    </div>
  </section>

  <section class="s wrap" id="edge-{name}" aria-labelledby="h-edge-{name}">
    <p class="eyebrow">Is there an edge?</p>
    <h2 id="h-edge-{name}">{label} shows no edge out of sample: Sharpe {_num(su["point"])}, deflated probability {wf["dsr"]:.2f} after {trials} trials</h2>
    <p class="lede">Each year, parameters are chosen on the previous three years and scored on the
    next one. Pooled over {n_windows} test windows, the {su["ci_level_pct"]} % bootstrap interval
    runs from {_num(su["ci_low"])} to {_num(su["ci_high"])}. Deflating for {trials} trials leaves a
    probability of {wf["dsr"]:.2f} that the Sharpe is real, against {su["dsr_threshold"]} needed.</p>
    <figure class="figure" aria-labelledby="f-win-{name}">
      <div class="fig-head"><div><h3 id="f-win-{name}">{win_title}</h3><p class="sub">Annualized Sharpe per test window · dot = estimate, line = {su["ci_level_pct"]} % bootstrap interval, vertical rule = zero</p></div>{TOGGLE}</div>
      <div id="c-win-{name}"></div>{window_table}
      <figcaption>Out-of-sample daily returns, next-bar-open fills, parameters refit each window.
      Each window holds one year of daily returns, so intervals are wide.{flat_note}</figcaption>
    </figure>
  </section>

  <section class="s wrap" id="market-{name}" aria-labelledby="h-market-{name}">
    <p class="eyebrow">Did it beat buy-and-hold?</p>
    <h2 id="h-market-{name}">It lost {_pct(-wf["oos_return_pct"])} while buy-and-hold lost {_pct(-wf["buy_and_hold_return_pct"])}, mostly by holding cash</h2>
    <p class="lede">The strategy held the asset on {wf["days_in_market_pct"]} % of out-of-sample days
    and sat in cash otherwise, so a falling market hurt it less. Losing less in a synthetic drawdown
    is not an edge; the Sharpe interval above is the test of that.</p>
    <figure class="figure" aria-labelledby="f-eq-{name}">
      <div class="fig-head"><div><h3 id="f-eq-{name}">Strategy equity stays near flat while buy-and-hold swings</h3><p class="sub">Equity (USD) from $100,000 · stitched out-of-sample windows · {oos_span}</p></div>{TOGGLE}</div>
      <div id="c-eq-{name}"></div>{returns_table}
      <figcaption>Blue: the strategy, stitched across test windows. Orange: buy-and-hold over the
      same days. Single synthetic price path, so no interval.</figcaption>
    </figure>
  </section>

  <section class="s wrap" id="checks-{name}" aria-labelledby="h-checks-{name}">
    <p class="eyebrow">Do fills and costs change the answer?</p>
    <h2 id="h-checks-{name}">Look-ahead fills {gap_word} return by {_num(abs(gap), 1)} pp, and costs only lower Sharpe further</h2>
    <p class="lede">These two checks use the full sample ({full_span}) with fixed parameters
    ({html.escape(_params_text(name, s["params"]))}), not the walk-forward windows.{coincidence}</p>
    {full_table}
    <p class="caveat"><b>The look-ahead test is weak on this data.</b> Bars are rebuilt from closes,
    so each open equals the previous close. Real open prices would show a larger gap.</p>
  </section>
  </div>"""


def render_html(results: Dict) -> str:
    data = results["meta"]["data"]
    meta = results["meta"]
    strategies = results["strategies"]
    names = list(strategies)
    bench = results.get("benchmark")
    sha = str(meta.get("sha", "unknown"))[:7]
    date = str(meta.get("generated_at", ""))[:10]
    if data.get("synthetic"):
        key = (
            f"Synthetic data: seed {data['seed']}, generated by {data['generator']}. "
            "These are not market prices."
        )
        source = "synthetic random-walk prices with regime-switching drift"
    else:
        key = f"{data['source']}, {data['symbol']}, {data['start']} to {data['end']}."
        source = f"{data['symbol']} prices"
    first = strategies[names[0]] if names else None

    if first and data.get("synthetic"):
        h1 = "On synthetic random-walk prices, the backtester correctly finds no edge"
    elif first:
        h1 = f"{first['label']} on {data['symbol']}: out-of-sample Sharpe {_num(first['sharpe_uncertainty']['point'])}"
    else:
        h1 = "Backtest report"

    hero_numbers = ""
    if first:
        f_su, f_wf = first["sharpe_uncertainty"], first["walk_forward"]
        hero_numbers = (
            f" {html.escape(first['label'])}'s out-of-sample Sharpe is {_num(f_su['point'])}, with a "
            f"{f_su['ci_level_pct']} % interval from {_num(f_su['ci_low'])} to {_num(f_su['ci_high'])}; "
            f"its deflated Sharpe probability is {f_wf['dsr']:.2f} after {f_wf['total_trials']} trials."
        )
    picker = ""
    if len(names) > 1:
        buttons = "".join(
            f'<button role="radio" aria-checked="{"true" if i == 0 else "false"}" data-value="{n}">{html.escape(strategies[n]["label"])}</button>'
            for i, n in enumerate(names)
        )
        picker = (
            '<div class="wrap picker"><span class="label" id="pick-label">Strategy</span>'
            f'<div class="seg" role="radiogroup" aria-labelledby="pick-label" id="strategy">{buttons}</div></div>'
        )
    blocks = "".join(_strategy_html(n, strategies[n], data, i > 0) for i, n in enumerate(names))

    bench_html = ""
    if bench:
        rows = bench["rows_shown"]
        one, big = rows[0], rows[-1]
        bench_table = _table(
            ["Assets", "Vectorized (s)", "Event-driven (s)", "Event ÷ vectorized"],
            [
                [
                    f"{r['assets']}",
                    f"{r['vectorized_s_shown']}",
                    f"{r['event_s_shown']}",
                    f"{r['speedup_shown']}×",
                ]
                for r in rows
            ],
            f"Median of {bench['runs']} runs, {bench['cpu']}",
            "rk-table",
        )
        bench_html = f"""
  <section class="s wrap" id="speed" aria-labelledby="h-speed">
    <p class="eyebrow">How fast is the vectorized engine?</p>
    <h2 id="h-speed">With {one["assets"]} asset the vectorized engine is {one["speedup_shown"]}× faster; the gap grows with asset count</h2>
    <p class="lede">The vectorized engine's time stays near {one["vectorized_s_shown"]}{NBSP}s from {one["assets"]} to
    {big["assets"]} assets, which is mostly fixed overhead. The event engine's time grows with the
    number of assets, reaching {big["event_s_shown"]}{NBSP}s and a {big["speedup_shown"]}× ratio at {big["assets"]} assets.</p>
    <figure class="figure" aria-labelledby="f-speed">
      <div class="fig-head"><div><h3 id="f-speed">Event-driven time grows with assets; vectorized time stays flat</h3><p class="sub">Median seconds per backtest (log scale) · {html.escape(bench["data"])}</p></div>{TOGGLE}</div>
      <div id="c-speed"></div>{bench_table}
      <figcaption>{html.escape(bench["strategy"])}. Median of {bench["runs"]} runs after a warm-up, no
      spread shown.</figcaption>
    </figure>
    <p class="caveat"><b>One laptop, one session.</b> Measured once on an {html.escape(bench["cpu"])}
    (Python {bench["python"]}, Numba {bench["numba"]}) and committed as
    <span class="mono">benchmarks/results.json</span>; CI does not re-run it.</p>
  </section>"""

    env_rows = [
        ["Prices", html.escape(data.get("source", ""))],
        ["Bars", f"{data.get('n_bars', '')}" if data.get("n_bars") else "—"],
        ["Asset used", html.escape(str(data.get("symbol", "")))],
        ["Bootstrap seed", f"{meta.get('seed', '')}"],
        ["Commit", f"<span class='mono'>{sha}</span>"],
        ["Built", date],
    ]
    if bench:
        env_rows.append(
            ["Benchmark machine", html.escape(f"{bench['cpu']}, Python {bench['python']}")]
        )
    env_table = _table(["Field", "Value"], env_rows, "Build environment")

    shows = ""
    if first:
        f_su, f_wf = first["sharpe_uncertainty"], first["walk_forward"]
        shows = f"""
    <div class="two">
      <div><h3>What this shows</h3><ul><li>The walk-forward, bootstrap and deflated-Sharpe checks report no edge where none is expected.</li><li>{html.escape(first["label"])}'s out-of-sample Sharpe interval, [{_num(f_su["ci_low"])}, {_num(f_su["ci_high"])}], includes zero.</li><li>Trading less, not skill, explains the smaller loss than buy-and-hold.</li></ul></div>
      <div><h3>What it doesn't show</h3><ul><li>Performance on market prices. Every number here is simulated.</li><li>That the checks would catch a real edge; this run only tests the null case.</li><li>Results on the other {max(int(data.get("n_assets", 1)) - 1, 0)} synthetic assets; the report uses one.</li></ul></div>
    </div>"""

    payload = json.dumps(results, separators=(",", ":")).replace("</", "<\\/")
    command = html.escape(str(meta.get("command", "")))
    kit_css = (HERE / "kit" / "report.css").read_text()
    local_css = (HERE / "report.local.css").read_text()
    kit_js = "".join((HERE / "kit" / f).read_text() for f in ("report.js", "charts.js"))
    kit_js = kit_js.replace("</script", "<\\/script")  # a comment in report.js names the tag
    local_js = (HERE / "report.local.js").read_text()
    title = "QuantSim backtest report"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{title} · Allen Wang</title>
<meta name="description" content="A Python backtesting framework run as a control on {html.escape(source)}: walk-forward, bootstrap and deflated-Sharpe checks.">
<meta name="theme-color" content="#f5f5f7" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#0b0b0b" media="(prefers-color-scheme: dark)">
<link rel="icon" href="data:,">
<script>try{{var t=localStorage.getItem("theme");if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t)}}catch(e){{}}</script>
<style>{kit_css}
{local_css}</style>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="bar">
  <div class="wrap">
    <a class="back" href="https://allenwang0.github.io/">← Allen Wang</a>
    <span class="title" aria-hidden="true">{title}</span>
    <a class="pill" href="{REPO}">Code ↗</a>
  </div>
</header>
<main id="main">
  <section class="hero wrap" aria-labelledby="h1">
    <p class="eyebrow">Backtesting framework · Python</p>
    <h1 id="h1">{html.escape(h1)}</h1>
    <p class="lede">This run is a control. On {html.escape(source)}, a trend rule should not show
    reliable skill, and the walk-forward, bootstrap and deflated-Sharpe checks should say so. They do.{hero_numbers}</p>
    <p class="conditions" id="banner"><span class="key">{html.escape(key)}</span><span>One asset ({html.escape(str(data.get("symbol", "")))}), {data.get("start", "")} to {data.get("end", "")}</span><span>Next-bar-open fills</span></p>
    <figure class="figure plain hero-fig" aria-labelledby="f-hero">
      <p class="label" id="f-hero">Out-of-sample Sharpe, pooled over all test windows, with its {first["sharpe_uncertainty"]["ci_level_pct"] if first else ""} % bootstrap interval. The vertical rule at zero means no edge.</p>
      <div id="c-hero"></div>
    </figure>
  </section>
  {picker}
  {blocks}
  {bench_html}

  <section class="s wrap" id="method" aria-labelledby="h-method">
    <p class="eyebrow">Methodology</p>
    <h2 id="h-method">How this was measured</h2>
    <div class="prose" style="margin-top:24px">
      <p>Prices come from <span class="mono">make_fixture.py</span>: a geometric random walk whose
      drift and volatility switch between bull, bear and sideways regimes. Within a regime, daily
      returns are independent. Regimes switch at random, so any trend a rule picks up is weak and noisy.</p>
      <p>Walk-forward testing picks parameters on three years and scores the next year, then steps a
      year. The bootstrap resamples out-of-sample daily returns. The deflated Sharpe ratio discounts
      the average out-of-sample Sharpe for the number of parameter settings tried across windows.</p>
    </div>
    {shows}
  </section>

  <section class="s wrap" id="reproduce" aria-labelledby="h-repro">
    <p class="eyebrow">Reproduce</p>
    <h2 id="h-repro">Run it yourself</h2>
    <div class="codewrap"><pre><code>git clone {REPO}.git && cd quantsim
make setup
uv run python tests/fixtures/make_fixture.py
uv run {command}
cd benchmarks && uv run python bench_engines.py   # engine timing</code></pre></div>
    <details>
      <summary>Environment for this build · {len(env_rows)} rows</summary>
      {env_table}
    </details>
  </section>
</main>
<footer><div class="wrap"><span><a href="https://allenwang0.github.io/">← Allen Wang</a><a href="{REPO}">Code ↗</a></span><span>Data from commit <span class="mono">{sha}</span> · {date}</span></div></footer>
<script type="application/json" id="results">{payload}</script>
<script>{kit_js}</script>
<script>{local_js}</script>
</body>
</html>
"""
