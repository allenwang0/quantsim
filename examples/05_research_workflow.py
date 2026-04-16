"""
Example 5: Complete Research Workflow
======================================

Demonstrates the full quant research workflow on a single strategy:

1. Hypothesis: SMA crossover on a diversified basket should work
2. Sanity check: vectorized backtest with DSR filtering
3. Parameter optimization: sweep 12 configurations
4. Walk-forward validation: out-of-sample Sharpe is the only truth
5. Regime analysis: does the strategy work across market regimes?
6. Full backtest: event-driven with realistic slippage
7. Tearsheet generation: offline HTML report

Key lesson: the walk-forward OOS Sharpe is the number that matters.
Everything else is in-sample fitting.

Run:
    python examples/05_research_workflow.py
"""

import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
import time

from core.logging_config import setup_logging
from core.database_v2 import init_full_db
from core.database import db_conn
from backtesting.vectorized import VectorizedBacktester, sma_crossover_signal, momentum_signal
from backtesting.walk_forward import WalkForwardOptimizer, RegimeAnalyzer
from strategies.trend import SMAcrossover
from backtesting.engine import BacktestEngine
from backtesting.execution import VolumeProportionalSlippage, ZeroCommission
from core.event_queue import EventQueue
from reporting.tearsheet import generate_tearsheet
from reporting.analytics import load_equity_curve_from_db

setup_logging(level="WARNING")

# ── Synthetic multi-asset data ─────────────────────────────────────────────────
print("=" * 60)
print("QUANTSIM v2 — Complete Research Workflow")
print("=" * 60)

DB_PATH = "/tmp/quantsim_research_workflow.db"
if os.path.exists(DB_PATH):
    os.unlink(DB_PATH)
os.environ["QUANTSIM_DB"] = DB_PATH
init_full_db(DB_PATH)

np.random.seed(42)
SYMBOLS = ["SPY", "QQQ", "GLD", "TLT", "IWM"]
N_BARS = 756  # 3 years
DRIFTS = [0.0004, 0.0005, 0.0001, -0.0001, 0.0003]
VOLS   = [0.012, 0.015, 0.008, 0.006, 0.014]
STARTS = [300, 350, 170, 120, 180]

dates = pd.date_range("2018-01-02", periods=N_BARS, freq="B")
price_dict = {}

for sym, drift, vol, start in zip(SYMBOLS, DRIFTS, VOLS, STARTS):
    prices = start * np.cumprod(1 + np.random.normal(drift, vol, N_BARS))
    price_dict[sym] = prices
    with db_conn(DB_PATH) as conn:
        for d, p in zip(dates, prices):
            ts = int(d.timestamp())
            conn.execute(
                "INSERT OR REPLACE INTO raw_bars "
                "(asset_id,timestamp,open,high,low,close,volume,source) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (sym, ts, p*0.998, p*1.005, p*0.995, p, 5_000_000, "synthetic"),
            )
            conn.execute(
                "INSERT OR REPLACE INTO adjustment_factors VALUES (?,?,1.0,0.0)",
                (sym, ts),
            )

prices_df = pd.DataFrame(price_dict, index=dates)
print(f"✓ Data: {len(SYMBOLS)} assets × {N_BARS} bars")

# ── Step 1: Quick sanity check (vectorized, 1 config) ─────────────────────────
print("\n── Step 1: Sanity check (vectorized SMA 20/100) ──────────────────")
t0 = time.perf_counter()
bt = VectorizedBacktester(prices_df, initial_capital=100_000)
sanity = bt.run(sma_crossover_signal, {"fast": 20, "slow": 100})
elapsed = (time.perf_counter() - t0) * 1000
print(f"  Elapsed:      {elapsed:.1f}ms")
print(f"  Total return: {sanity.get('total_return', 0):+.2%}")
print(f"  Sharpe:       {sanity.get('sharpe_ratio', 0):.3f}")
print(f"  Max DD:       {sanity.get('max_drawdown', 0):.2%}")
if (sanity.get("sharpe_ratio") or 0) < -0.5:
    print("  ⚠ Negative Sharpe on sanity check. Investigate before proceeding.")

# ── Step 2: Parameter sweep (12 configs, vectorized) ─────────────────────────
print("\n── Step 2: Parameter sweep (SMA, 12 configurations) ─────────────")
param_grid = {"fast": [10, 20, 50], "slow": [80, 120, 200], "long_only": [True]}
t0 = time.perf_counter()
sweep_df = bt.parameter_sweep(sma_crossover_signal, param_grid)
elapsed = (time.perf_counter() - t0) * 1000
print(f"  {len(sweep_df)} configs in {elapsed:.0f}ms")
n_significant = (sweep_df["deflated_sharpe_corrected"] > 0).sum()
print(f"  Statistically significant (DSR > 0): {n_significant}/{len(sweep_df)}")
print(f"\n  Top 3 configurations:")
for _, row in sweep_df.head(3).iterrows():
    print(f"    fast={int(row.fast)}, slow={int(row.slow)} | "
          f"Sharpe={row.sharpe_ratio:.3f}, DSR={row.deflated_sharpe_corrected:.3f}, "
          f"CAGR={row.cagr:+.2%}, MaxDD={row.max_drawdown:.2%}")

if n_significant == 0:
    print("\n  ⚠ No significant configurations. Strategy may not be viable.")
    best_params = {"fast": 20, "slow": 100}
else:
    best = sweep_df.iloc[0]
    best_params = {"fast": int(best.fast), "slow": int(best.slow)}
    print(f"\n  Best params: {best_params}")

# ── Step 3: Walk-forward optimization ─────────────────────────────────────────
print("\n── Step 3: Walk-forward optimization (OOS only) ──────────────────")
wfo = WalkForwardOptimizer(prices_df, train_years=1.5, test_months=6, step_months=6)
wfo_results = wfo.optimize_and_evaluate(
    signal_fn=sma_crossover_signal,
    param_grid={"fast": [10, 20, 50], "slow": [80, 120]},
)

print(f"  Windows: {wfo_results['n_windows']}")
print(f"  Avg IS Sharpe:  {wfo_results['avg_is_sharpe']:.3f}")
print(f"  Avg OOS Sharpe: {wfo_results['avg_oos_sharpe']:.3f}")
print(f"  IS→OOS Degradation: {wfo_results['sharpe_degradation']:.3f}")
print(f"  DSR corrected:  {wfo_results['deflated_sharpe_corrected']:.3f}")
print(f"  OOS win rate:   {wfo_results['oos_win_rate']:.1%}")
print(f"\n  VERDICT: {wfo_results['summary']['message']}")

# ── Step 4: Regime analysis ────────────────────────────────────────────────────
print("\n── Step 4: Regime analysis ────────────────────────────────────────")
spy_prices = prices_df["SPY"]
regimes = RegimeAnalyzer.classify_regimes(spy_prices)
regime_counts = regimes.value_counts()
print("  Regime distribution:")
for regime, count in regime_counts.items():
    pct = count / len(regimes)
    print(f"    {regime:<20} {count:>3} bars ({pct:.1%})")

# ── Step 5: Full event-driven backtest ────────────────────────────────────────
print("\n── Step 5: Full backtest (event-driven, SPY only) ────────────────")
eq = EventQueue()
strategy = SMAcrossover(
    asset_ids=["SPY"], event_queue=eq,
    fast=best_params["fast"], slow=best_params["slow"],
    long_only=True,
)
engine = BacktestEngine(
    strategies=[strategy],
    start=datetime(2018, 1, 2), end=datetime(2020, 6, 30),
    initial_capital=100_000,
    slippage_model=VolumeProportionalSlippage(k=0.05),
    commission_model=ZeroCommission(),
    db_path=DB_PATH, verbose=False,
    warmup_bars=max(best_params["slow"] + 10, 110),
)
results = engine.run()

print(f"  Total return:  {results.get('total_return', 0):+.2%}")
print(f"  CAGR:          {results.get('cagr', 0):+.2%}")
print(f"  Sharpe:        {results.get('sharpe_ratio', 0):.4f}")
print(f"  Sortino:       {results.get('sortino_ratio', 0):.4f}")
print(f"  Max drawdown:  {results.get('max_drawdown', 0):.2%}")
print(f"  N trades:      {results.get('n_trades', 0)}")
print(f"  Fills:         {engine._fills_processed}")
print(f"  DSR:           {results.get('deflated_sharpe_ratio', 0):.4f}")

# Warn if results are in-sample overfit
if (results.get("sharpe_ratio") or 0) > (wfo_results["avg_oos_sharpe"] or 0) + 0.5:
    print("\n  ⚠ IS Sharpe significantly higher than OOS Sharpe — possible overfitting.")

# ── Step 6: Tearsheet ─────────────────────────────────────────────────────────
print("\n── Step 6: Tearsheet generation ───────────────────────────────────")
ec = load_equity_curve_from_db(DB_PATH)

if ec.empty or ec.std() == 0:
    # Use portfolio history fallback
    hist = engine.portfolio._equity_history
    if hist:
        ts_list = [t.replace(tzinfo=None) if hasattr(t,'tzinfo') and t.tzinfo else t for t,_ in hist]
        eq_list = [float(e) for _,e in hist]
        ec = pd.Series(eq_list, index=pd.DatetimeIndex(ts_list))
        ec = ec[~ec.index.duplicated(keep='last')].sort_index()

if not ec.empty and ec.std() > 0:
    tearsheet_path = "/tmp/quantsim_research_workflow.html"
    generate_tearsheet(
        equity_curve=ec,
        strategy_name=f"SMA {best_params['fast']}/{best_params['slow']} on SPY",
        wfo_results=wfo_results,
        output_path=tearsheet_path,
        n_strategies_tested=len(sweep_df),
    )
    print(f"  Tearsheet: {tearsheet_path}")
    print(f"  Size: {os.path.getsize(tearsheet_path):,} bytes")
else:
    print("  ⚠ Tearsheet skipped: insufficient equity data")

# ── Final summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESEARCH WORKFLOW COMPLETE")
print("=" * 60)
is_viable = (
    wfo_results.get("deflated_sharpe_corrected", -1) > 0
    and (results.get("total_return") or 0) > 0
)
status = "✓ VIABLE — proceed to paper trading" if is_viable else "✗ NOT VIABLE — revisit hypothesis"
print(f"  {status}")
print(f"  IS Sharpe: {results.get('sharpe_ratio',0):.3f}  "
      f"OOS Sharpe: {wfo_results['avg_oos_sharpe']:.3f}  "
      f"DSR: {wfo_results['deflated_sharpe_corrected']:.3f}")
print()
print("  Next step: paper trade for 30+ days before risking real capital.")
print("  Command: python scripts/run_paper_trading.py --strategy sma --symbol SPY")
