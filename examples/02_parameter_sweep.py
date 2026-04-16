"""
Example 2: Vectorized Parameter Sweep
======================================

Shows how to use the vectorized backtester to sweep 30 parameter
configurations in under a second, then apply Deflated Sharpe Ratio
to find which configurations are statistically significant.

Key insight: running 30 configurations makes any individual Sharpe ratio
less meaningful. DSR corrects for this.

Run:
    cd examples && python 02_parameter_sweep.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
import time

from core.logging_config import setup_logging
from backtesting.vectorized import VectorizedBacktester, sma_crossover_signal

setup_logging(level="WARNING")  # quiet for this example

# ── Generate synthetic multi-asset prices ─────────────────────────────────────
np.random.seed(42)
n_bars = 756  # ~3 years
symbols = ["SPY", "QQQ", "GLD", "TLT"]
dates = pd.date_range("2018-01-01", periods=n_bars, freq="B")
drifts = [0.0004, 0.0005, 0.0001, -0.0001]
vols = [0.012, 0.015, 0.008, 0.007]

prices = pd.DataFrame({
    sym: 100 * np.cumprod(1 + np.random.normal(d, v, n_bars))
    for sym, d, v in zip(symbols, drifts, vols)
}, index=dates)

print(f"Loaded {n_bars} bars for {len(symbols)} assets")

# ── Run the parameter sweep ────────────────────────────────────────────────────
bt = VectorizedBacktester(
    prices=prices,
    initial_capital=100_000,
    commission_rate=0.001,
    slippage_pct=0.001,
)

param_grid = {
    "fast": [10, 20, 30, 50],
    "slow": [80, 100, 150, 200, 250],
    "long_only": [True],
}
n_configs = 4 * 5 * 1  # = 20

print(f"\nRunning parameter sweep: {n_configs} configurations...")
t0 = time.perf_counter()

results_df = bt.parameter_sweep(sma_crossover_signal, param_grid)

elapsed = time.perf_counter() - t0
print(f"Completed in {elapsed:.2f}s ({elapsed/n_configs*1000:.1f}ms per config)")

# ── Display results ────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("TOP 5 CONFIGURATIONS (by Deflated Sharpe Ratio)")
print(f"{'='*65}")
display_cols = ["fast", "slow", "sharpe_ratio", "cagr",
                "max_drawdown", "deflated_sharpe_corrected"]
display_cols = [c for c in display_cols if c in results_df.columns]
print(results_df[display_cols].head(5).to_string(index=False))

print(f"\n{'='*65}")
print("BOTTOM 3 CONFIGURATIONS (avoid these)")
print(f"{'='*65}")
print(results_df[display_cols].tail(3).to_string(index=False))

# ── Key insight: DSR corrects for multiple comparisons ─────────────────────────
top = results_df.iloc[0]
n_significant = (results_df["deflated_sharpe_corrected"] > 0).sum()

print(f"\n{'='*65}")
print("INTERPRETATION")
print(f"{'='*65}")
print(f"Configurations tested:    {len(results_df)}")
print(f"Statistically significant (DSR > 0): {n_significant}/{len(results_df)}")
print(f"Best raw Sharpe:          {results_df['sharpe_ratio'].max():.3f}")
print(f"Best Deflated Sharpe:     {results_df['deflated_sharpe_corrected'].max():.3f}")

if n_significant == 0:
    print("\n⚠ No configuration survived the DSR correction.")
    print("  This is common on short data samples. Need more history.")
elif n_significant < 3:
    print(f"\n⚠ Only {n_significant} configs are significant. Be conservative.")
else:
    print(f"\n✓ {n_significant} configurations are statistically significant.")
    print(f"  Best: fast={int(top.get('fast',0))}, slow={int(top.get('slow',0))}")
    print(f"  DSR={top.get('deflated_sharpe_corrected',0):.3f}")
