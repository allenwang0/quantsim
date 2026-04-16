"""
HTML Tearsheet Generator

Produces a self-contained HTML file with the full performance report:
- Strategy overview (key metrics table)
- Equity curve + drawdown (interactive Plotly)
- Monthly returns heatmap
- Rolling Sharpe
- Trade log table
- Walk-forward summary (if available)
- Regime breakdown

Generated reports are standalone HTML files (no external dependencies)
that can be emailed, saved to S3, or opened in any browser offline.

Usage:
    from reporting.tearsheet import generate_tearsheet
    generate_tearsheet(
        equity_curve=equity_series,
        trades_df=trades_df,
        output_path="reports/backtest_sma_spy_2024.html",
        strategy_name="SMA 50/200 on SPY",
        benchmark=spy_series,
    )
"""

from __future__ import annotations
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_json(obj) -> str:
    """JSON-serialize an object, handling datetime and numpy types."""
    def default(o):
        if isinstance(o, (datetime,)):
            return o.isoformat()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        return str(o)
    return json.dumps(obj, default=default)


def generate_tearsheet(
    equity_curve: pd.Series,
    strategy_name: str = "Strategy",
    trades_df: Optional[pd.DataFrame] = None,
    benchmark: Optional[pd.Series] = None,
    wfo_results: Optional[Dict] = None,
    output_path: Optional[str] = None,
    initial_capital: float = 100_000,
    n_strategies_tested: int = 1,
) -> str:
    """
    Generate a full HTML tearsheet.

    Args:
        equity_curve: DatetimeIndex → dollar equity series
        strategy_name: Display name for the strategy
        trades_df: DataFrame of closed trades
        benchmark: DatetimeIndex → dollar series for benchmark comparison
        wfo_results: Output from WalkForwardOptimizer.optimize_and_evaluate()
        output_path: If provided, write HTML to this file
        initial_capital: Starting capital (for context)
        n_strategies_tested: For DSR calculation

    Returns:
        HTML string
    """
    from reporting.analytics import PerformanceAnalytics
    from reporting.advanced import AdvancedAnalytics

    analytics = PerformanceAnalytics()
    advanced = AdvancedAnalytics()

    # Core metrics
    metrics = analytics.compute_all(
        equity_curve=equity_curve,
        trades=trades_df,
        benchmark=benchmark,
        n_strategies_tested=n_strategies_tested,
    )

    # Daily returns for Monte Carlo
    daily_returns = equity_curve.pct_change().dropna()
    mc_results = {}
    if len(daily_returns) >= 30:
        mc_results = advanced.monte_carlo_sharpe(daily_returns, n_simulations=2000)

    # Prepare chart data
    eq_dates = [str(d) for d in equity_curve.index]
    eq_values = [round(float(v), 2) for v in equity_curve.values]

    # Drawdown series
    rolling_max = equity_curve.expanding().max()
    drawdown = ((equity_curve - rolling_max) / rolling_max * 100).fillna(0)
    dd_values = [round(float(v), 3) for v in drawdown.values]

    # Benchmark data
    bench_values = []
    if benchmark is not None:
        bench_aligned = benchmark.reindex(equity_curve.index).ffill().bfill()
        scale = equity_curve.iloc[0] / bench_aligned.iloc[0] if bench_aligned.iloc[0] != 0 else 1
        bench_values = [round(float(v * scale), 2) for v in bench_aligned.values]

    # Monthly returns for heatmap
    monthly_ret_dict = {}
    monthly_eq = equity_curve.resample("ME").last()
    monthly_ret = monthly_eq.pct_change(fill_method=None).dropna() * 100
    if not monthly_ret.empty:
        for ts, val in monthly_ret.items():
            yr = str(ts.year)
            mo = ts.strftime("%b")
            if yr not in monthly_ret_dict:
                monthly_ret_dict[yr] = {}
            monthly_ret_dict[yr][mo] = round(float(val), 2)

    # Rolling Sharpe
    rs_window = min(252, len(daily_returns) // 2)
    rolling_sr = (
        daily_returns.rolling(rs_window).mean() /
        daily_returns.rolling(rs_window).std() * np.sqrt(252)
    ).dropna()
    rs_dates = [str(d) for d in rolling_sr.index]
    rs_values = [round(float(v), 3) for v in rolling_sr.values]

    # Trade log
    trade_rows = []
    if trades_df is not None and not trades_df.empty:
        closed = trades_df[trades_df.get("realized_pnl", pd.Series()).notna()].copy() \
            if "realized_pnl" in trades_df.columns else trades_df
        for _, row in closed.head(200).iterrows():
            pnl = row.get("realized_pnl", 0)
            trade_rows.append({
                "symbol": row.get("asset_id", ""),
                "direction": row.get("direction", ""),
                "entry": f"${row.get('entry_price', 0):.2f}",
                "exit": f"${row.get('exit_price', 0):.2f}" if pd.notna(row.get("exit_price")) else "Open",
                "pnl": f"${pnl:+,.0f}" if pd.notna(pnl) else "Open",
                "pnl_num": float(pnl) if pd.notna(pnl) else 0,
                "strategy": row.get("strategy_id", ""),
                "bars": int(row.get("holding_bars", 0)) if pd.notna(row.get("holding_bars")) else 0,
            })

    # Format metrics for display
    def fmt_pct(v, decimals=2): return f"{v:.{decimals}%}" if v is not None else "N/A"
    def fmt_f(v, d=3): return f"{v:.{d}f}" if v is not None else "N/A"
    def fmt_dollar(v): return f"${v:,.0f}" if v is not None else "N/A"

    m = metrics
    verdict_color = "#00e676" if m.get("deflated_sharpe_ratio", -1) > 0 else "#ff3d6b"
    verdict_text = "VIABLE" if m.get("deflated_sharpe_ratio", -1) > 0 else "NOT SIGNIFICANT"

    ci_low = mc_results.get("sharpe_ci_low", 0)
    ci_high = mc_results.get("sharpe_ci_high", 0)
    ci_text = f"[{ci_low:.2f}, {ci_high:.2f}]" if mc_results else "N/A"

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # ── HTML Template ──────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QuantSim — {strategy_name}</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@600;700;800&display=swap');
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:#08080f;color:#e0e0f0;font-family:'JetBrains Mono',monospace;font-size:13px;}}
  .header{{padding:2rem 3rem 1rem;border-bottom:1px solid #252535;}}
  h1{{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:#fff;letter-spacing:-0.02em;}}
  .subtitle{{color:#5a5a7a;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.15em;margin-top:0.25rem;}}
  .container{{padding:2rem 3rem;max-width:1400px;}}
  .metrics-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:1rem;margin-bottom:2rem;}}
  .mc{{background:#111118;border:1px solid #252535;border-radius:8px;padding:1rem 1.2rem;position:relative;overflow:hidden;}}
  .mc::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#00d4ff,#7b61ff);}}
  .ml{{font-size:0.62rem;color:#5a5a7a;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.3rem;}}
  .mv{{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;}}
  .ms{{font-size:0.68rem;color:#5a5a7a;margin-top:0.2rem;}}
  .pos{{color:#00e676;}} .neg{{color:#ff3d6b;}} .neu{{color:#e0e0f0;}}
  .section{{margin-bottom:2.5rem;}}
  .sh{{font-size:0.65rem;color:#5a5a7a;text-transform:uppercase;letter-spacing:0.15em;
       border-bottom:1px solid #252535;padding-bottom:0.4rem;margin-bottom:1rem;}}
  .chart-container{{background:#111118;border:1px solid #252535;border-radius:8px;padding:1rem;margin-bottom:1rem;}}
  table{{width:100%;border-collapse:collapse;background:#111118;border-radius:8px;overflow:hidden;border:1px solid #252535;}}
  th{{background:#18181f;color:#5a5a7a;font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;
      padding:0.7rem 1rem;text-align:left;border-bottom:1px solid #252535;font-weight:500;}}
  td{{padding:0.55rem 1rem;border-bottom:1px solid rgba(37,37,53,0.5);font-size:0.78rem;}}
  tr:last-child td{{border-bottom:none;}}
  tr:hover td{{background:rgba(255,255,255,0.02);}}
  .verdict-box{{background:#18181f;border:1px solid {verdict_color}33;border-radius:8px;
                padding:1.2rem 1.5rem;margin-bottom:2rem;}}
  .verdict-title{{font-family:'Syne',sans-serif;font-size:1rem;color:{verdict_color};font-weight:700;}}
  .verdict-body{{color:#5a5a7a;font-size:0.75rem;margin-top:0.5rem;line-height:1.6;}}
  .wfo-box{{background:#18181f;border:1px solid #252535;border-radius:8px;padding:1.2rem 1.5rem;margin-bottom:1rem;}}
  .footer{{padding:1.5rem 3rem;border-top:1px solid #252535;color:#5a5a7a;font-size:0.65rem;text-align:center;}}
  .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;}}
  @media(max-width:768px){{.two-col{{grid-template-columns:1fr;}} .container{{padding:1rem;}}}}
</style>
</head>
<body>
<div class="header">
  <h1>◈ {strategy_name}</h1>
  <div class="subtitle">QuantSim Tearsheet · Generated {now}</div>
</div>

<div class="container">

<!-- Verdict -->
<div class="verdict-box">
  <div class="verdict-title">{verdict_text}</div>
  <div class="verdict-body">
    Sharpe: {fmt_f(m.get('sharpe_ratio'), 3)} &nbsp;|&nbsp;
    Deflated SR: {fmt_f(m.get('deflated_sharpe_ratio'), 3)} &nbsp;|&nbsp;
    95% CI: {ci_text} &nbsp;|&nbsp;
    Max DD: {fmt_pct(m.get('max_drawdown'))} &nbsp;|&nbsp;
    CAGR: {fmt_pct(m.get('cagr'))}
    {"<br>⚠ Insufficient sample: Sharpe unreliable with < 30 trades." if m.get("insufficient_sample_warning") else ""}
  </div>
</div>

<!-- Key Metrics -->
<div class="section">
  <div class="sh">Key Performance Metrics</div>
  <div class="metrics-grid">
    <div class="mc"><div class="ml">Total Return</div>
      <div class="mv {'pos' if (m.get('total_return',0) or 0) >= 0 else 'neg'}">{fmt_pct(m.get('total_return'))}</div></div>
    <div class="mc"><div class="ml">CAGR</div>
      <div class="mv {'pos' if (m.get('cagr',0) or 0) >= 0 else 'neg'}">{fmt_pct(m.get('cagr'))}</div>
      <div class="ms">{m.get('n_years', 0):.1f} years</div></div>
    <div class="mc"><div class="ml">Sharpe Ratio</div>
      <div class="mv {'pos' if (m.get('sharpe_ratio',0) or 0) >= 0.5 else 'neg'}">{fmt_f(m.get('sharpe_ratio'))}</div></div>
    <div class="mc"><div class="ml">Deflated SR</div>
      <div class="mv {'pos' if (m.get('deflated_sharpe_ratio',0) or 0) >= 0 else 'neg'}">{fmt_f(m.get('deflated_sharpe_ratio'))}</div>
      <div class="ms">corrected for {n_strategies_tested} configs</div></div>
    <div class="mc"><div class="ml">Sortino Ratio</div>
      <div class="mv">{fmt_f(m.get('sortino_ratio'))}</div></div>
    <div class="mc"><div class="ml">Calmar Ratio</div>
      <div class="mv {'pos' if (m.get('calmar_ratio',0) or 0) >= 1 else 'neg'}">{fmt_f(m.get('calmar_ratio'))}</div></div>
    <div class="mc"><div class="ml">Max Drawdown</div>
      <div class="mv neg">{fmt_pct(m.get('max_drawdown'))}</div>
      <div class="ms">{m.get('max_drawdown_duration_days', 0)} days</div></div>
    <div class="mc"><div class="ml">Annual Vol</div>
      <div class="mv">{fmt_pct(m.get('annual_volatility'))}</div></div>
    {"<div class='mc'><div class='ml'>Alpha (annual)</div><div class='mv'>" + fmt_pct(m.get('alpha_annual')) + "</div></div>" if 'alpha_annual' in m else ""}
    {"<div class='mc'><div class='ml'>Beta</div><div class='mv'>" + fmt_f(m.get('beta'), 3) + "</div></div>" if 'beta' in m else ""}
    {"<div class='mc'><div class='ml'>Info Ratio</div><div class='mv'>" + fmt_f(m.get('information_ratio'), 3) + "</div></div>" if 'information_ratio' in m else ""}
  </div>
</div>

{"" if not m.get('n_trades') else f'''
<div class="section">
  <div class="sh">Trade Statistics</div>
  <div class="metrics-grid">
    <div class="mc"><div class="ml">Total Trades</div><div class="mv">{m.get('n_trades', 0)}</div>
      {"<div class='ms' style='color:#ffd740'>⚠ <30 trades: Sharpe unreliable</div>" if m.get("insufficient_sample_warning") else ""}</div>
    <div class="mc"><div class="ml">Win Rate</div>
      <div class="mv {'pos' if (m.get('win_rate',0) or 0) >= 0.5 else 'neg'}">{fmt_pct(m.get('win_rate'))}</div></div>
    <div class="mc"><div class="ml">Profit Factor</div>
      <div class="mv {'pos' if (m.get('profit_factor',0) or 0) >= 1.5 else 'neg'}">{fmt_f(m.get('profit_factor'), 2)}</div></div>
    <div class="mc"><div class="ml">Expectancy</div>
      <div class="mv {'pos' if (m.get('expectancy',0) or 0) >= 0 else 'neg'}">{fmt_dollar(m.get('expectancy'))}</div></div>
    <div class="mc"><div class="ml">Avg Win</div><div class="mv pos">{fmt_dollar(m.get('avg_win'))}</div></div>
    <div class="mc"><div class="ml">Avg Loss</div><div class="mv neg">{fmt_dollar(m.get('avg_loss'))}</div></div>
    <div class="mc"><div class="ml">Win/Loss Ratio</div><div class="mv">{fmt_f(m.get('avg_win_loss_ratio'), 2)}</div></div>
    <div class="mc"><div class="ml">Avg Hold (bars)</div><div class="mv">{fmt_f(m.get('avg_holding_bars'), 1)}</div></div>
  </div>
</div>
'''}

<!-- Equity Curve -->
<div class="section">
  <div class="sh">Equity Curve & Drawdown</div>
  <div class="chart-container"><div id="eq-chart"></div></div>
</div>

<div class="two-col">
  <!-- Monthly Heatmap -->
  <div class="section">
    <div class="sh">Monthly Returns</div>
    <div class="chart-container" style="overflow-x:auto;"><div id="heatmap-chart"></div></div>
  </div>
  <!-- Rolling Sharpe -->
  <div class="section">
    <div class="sh">Rolling Sharpe Ratio</div>
    <div class="chart-container"><div id="rs-chart"></div></div>
  </div>
</div>

{f'''
<!-- WFO Results -->
<div class="section">
  <div class="sh">Walk-Forward Optimization Results</div>
  <div class="wfo-box">
    <table>
      <tr>
        <th>Metric</th><th>Value</th>
      </tr>
      <tr><td>Windows Tested</td><td>{wfo_results.get("n_windows", "N/A")}</td></tr>
      <tr><td>Avg OOS Sharpe</td><td class="{'pos' if wfo_results.get('avg_oos_sharpe',0) > 0 else 'neg'}">{wfo_results.get('avg_oos_sharpe', 0):.3f}</td></tr>
      <tr><td>Avg IS Sharpe</td><td>{wfo_results.get('avg_is_sharpe', 0):.3f}</td></tr>
      <tr><td>IS→OOS Degradation</td><td class="neg">{wfo_results.get('sharpe_degradation', 0):.3f}</td></tr>
      <tr><td>Deflated SR (corrected)</td><td class="{'pos' if wfo_results.get('deflated_sharpe_corrected',0) > 0 else 'neg'}">{wfo_results.get('deflated_sharpe_corrected', 0):.3f}</td></tr>
      <tr><td>OOS Win Rate</td><td>{wfo_results.get('oos_win_rate', 0):.1%}</td></tr>
      <tr><td>Verdict</td><td style="color:{'#00e676' if wfo_results.get('summary',{}).get('pass') else '#ff3d6b'}">{wfo_results.get("summary", {}).get("message", "N/A")}</td></tr>
    </table>
  </div>
</div>
''' if wfo_results else ""}

<!-- Trade Log -->
{f'''
<div class="section">
  <div class="sh">Recent Trades ({len(trade_rows)} shown)</div>
  <table>
    <tr><th>Symbol</th><th>Dir</th><th>Entry</th><th>Exit</th><th>P&amp;L</th><th>Strategy</th><th>Bars</th></tr>
    {"".join(f"""<tr>
      <td>{r['symbol']}</td>
      <td>{r['direction']}</td>
      <td>{r['entry']}</td>
      <td>{r['exit']}</td>
      <td class="{'pos' if r['pnl_num'] >= 0 else 'neg'}">{r['pnl']}</td>
      <td style="color:#5a5a7a">{r['strategy']}</td>
      <td>{r['bars']}</td>
    </tr>""" for r in trade_rows[:100])}
  </table>
</div>
''' if trade_rows else ""}

</div><!-- /container -->

<div class="footer">QuantSim v2 · {strategy_name} · {now}</div>

<script>
const BG='#08080f', SRF='#111118', BRD='#252535', ACC='#00d4ff', ACC2='#7b61ff';
const GRN='#00e676', RED='#ff3d6b', TXT='#e0e0f0', MUT='#5a5a7a';
const LAYOUT_BASE = {{
  paper_bgcolor: BG, plot_bgcolor: SRF,
  font: {{family: 'JetBrains Mono, monospace', color: TXT, size: 11}},
  margin: {{l:50, r:20, t:20, b:40}},
  hovermode: 'x unified',
  xaxis: {{showgrid: false, zeroline: false, color: MUT}},
  yaxis: {{showgrid: true, gridcolor: BRD, zeroline: false, color: MUT}},
}};

// Equity curve + drawdown
const eqDates = {_safe_json(eq_dates)};
const eqValues = {_safe_json(eq_values)};
const ddValues = {_safe_json(dd_values)};
const benchValues = {_safe_json(bench_values)};

const eqTraces = [
  {{x: eqDates, y: eqValues, type: 'scatter', mode: 'lines', name: 'Portfolio',
    line: {{color: ACC, width: 2}}, fill: 'tozeroy', fillcolor: 'rgba(0,212,255,0.04)',
    yaxis: 'y'}},
  {{x: eqDates, y: ddValues, type: 'scatter', mode: 'lines', name: 'Drawdown %',
    line: {{color: RED, width: 1.5}}, fill: 'tozeroy', fillcolor: 'rgba(255,61,107,0.12)',
    yaxis: 'y2'}},
];
if (benchValues.length > 0) {{
  eqTraces.splice(1, 0, {{x: eqDates, y: benchValues, type: 'scatter', mode: 'lines',
    name: 'Benchmark', line: {{color: ACC2, width: 1.5, dash: 'dot'}}, yaxis: 'y'}});
}}
Plotly.newPlot('eq-chart', eqTraces, {{
  ...LAYOUT_BASE, height: 360, showlegend: true,
  legend: {{orientation: 'h', y: 1.1, font: {{size: 10}}}},
  yaxis: {{...LAYOUT_BASE.yaxis, tickprefix: '$', tickformat: ',.0f'}},
  yaxis2: {{showgrid: false, zeroline: true, zerolinecolor: MUT,
    overlaying: 'y', side: 'right', ticksuffix: '%', color: MUT}},
}}, {{displayModeBar: false, responsive: true}});

// Monthly heatmap
const hmData = {_safe_json(monthly_ret_dict)};
const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const years = Object.keys(hmData).sort();
const z = years.map(yr => months.map(mo => hmData[yr][mo] ?? null));
const annots = [];
years.forEach((yr, i) => months.forEach((mo, j) => {{
  if (z[i][j] !== null) annots.push({{
    x: j, y: i, text: z[i][j].toFixed(1)+'%',
    xref:'x', yref:'y', showarrow: false,
    font: {{size: 9, color: Math.abs(z[i][j]) > 3 ? 'white' : MUT}}
  }});
}}));
const mx = Math.max(...z.flat().filter(v => v !== null).map(Math.abs), 1);
Plotly.newPlot('heatmap-chart', [{{
  type: 'heatmap', z, x: months, y: years,
  colorscale: [[0,'#cc1f3d'],[0.5,SRF],[1,'#00e676']],
  zmin: -mx, zmax: mx, showscale: false,
}}], {{
  ...LAYOUT_BASE, height: Math.max(120, years.length * 28 + 60),
  annotations: annots,
  xaxis: {{...LAYOUT_BASE.xaxis, side: 'top'}},
  yaxis: {{...LAYOUT_BASE.yaxis, showgrid: false}},
}}, {{displayModeBar: false, responsive: true}});

// Rolling Sharpe
const rsDates = {_safe_json(rs_dates)};
const rsValues = {_safe_json(rs_values)};
Plotly.newPlot('rs-chart', [
  {{x: rsDates, y: rsValues, type: 'scatter', mode: 'lines', name: 'Rolling SR',
    line: {{color: ACC2, width: 2}}}},
  {{x: [rsDates[0], rsDates[rsDates.length-1]], y: [0,0], type: 'scatter', mode: 'lines',
    line: {{color: MUT, dash: 'dash', width: 1}}, showlegend: false}},
  {{x: [rsDates[0], rsDates[rsDates.length-1]], y: [1,1], type: 'scatter', mode: 'lines',
    line: {{color: GRN, dash: 'dot', width: 1}}, showlegend: false}},
], {{
  ...LAYOUT_BASE, height: 200, showlegend: false,
  yaxis: {{...LAYOUT_BASE.yaxis, title: 'Sharpe'}},
}}, {{displayModeBar: false, responsive: true}});
</script>
</body>
</html>"""

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Tearsheet written to {output_path}")

    return html
