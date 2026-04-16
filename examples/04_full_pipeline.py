"""
Example 4: Full Pipeline with Stop-Loss and Tearsheet
======================================================

Demonstrates the complete production pipeline:
1. Generate synthetic data with realistic volatility clustering
2. Run SMA crossover with OrderManager stop-losses
3. Generate an HTML tearsheet
4. Run health monitor post-backtest

This is the pattern you'd use for a real strategy evaluation.

Run:
    cd examples && python 04_full_pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime

from core.logging_config import setup_logging
from core.database_v2 import init_full_db
from core.database import db_conn
from strategies.trend import SMAcrossover
from core.event_queue import EventQueue
from backtesting.engine import BacktestEngine
from backtesting.order_manager import OrderManager
from backtesting.execution import VolumeProportionalSlippage, ZeroCommission
from reporting.analytics import PerformanceAnalytics, load_equity_curve_from_db, load_trades_from_db
from reporting.tearsheet import generate_tearsheet
from reporting.monitor import StrategyHealthMonitor

setup_logging(level="INFO")

# ── Setup ──────────────────────────────────────────────────────────────────────
DB_PATH = "/tmp/quantsim_example_04.db"
os.environ["QUANTSIM_DB"] = DB_PATH
init_full_db(DB_PATH)

# Synthetic SPY with GARCH-like volatility clustering
np.random.seed(7)
n = 756
dates = pd.date_range("2018-01-01", periods=n, freq="B")
returns = []
sigma = 0.012
for _ in range(n):
    r = np.random.normal(0.0004, sigma)
    returns.append(r)
    sigma = np.sqrt(0.000001 + 0.09 * r**2 + 0.88 * sigma**2)

prices = 300 * np.cumprod(1 + np.array(returns))

with db_conn(DB_PATH) as conn:
    for date, price in zip(dates, prices):
        ts = int(date.timestamp())
        conn.execute(
            "INSERT OR REPLACE INTO raw_bars "
            "(asset_id, timestamp, open, high, low, close, volume, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("SPY", ts, price*0.999, price*1.005, price*0.995, price, 10_000_000, "garch_synth"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO adjustment_factors "
            "(asset_id, effective_date, cumulative_split_factor, cumulative_div_adjustment) "
            "VALUES (?, ?, 1.0, 0.0)", ("SPY", ts),
        )

print(f"Synthetic data: {n} bars, vol-clustered GARCH process")

# ── Run backtest with stop-loss protection ─────────────────────────────────────
eq = EventQueue()
strategy = SMAcrossover(
    asset_ids=["SPY"],
    event_queue=eq,
    fast=20, slow=100,
    long_only=True,
)

engine = BacktestEngine(
    strategies=[strategy],
    start=datetime(2018, 1, 1),
    end=datetime(2020, 12, 31),
    initial_capital=100_000,
    slippage_model=VolumeProportionalSlippage(k=0.05),
    commission_model=ZeroCommission(),
    db_path=DB_PATH,
    verbose=True,
)

# Add a trailing stop to protect against large drawdowns
engine.order_manager.add_trailing_stop(
    "SPY", entry_price=300.0, strategy_id="SMA_20_100",
    trail_pct=0.05, direction="LONG",
)

results = engine.run()

# ── Post-backtest health check ─────────────────────────────────────────────────
monitor = StrategyHealthMonitor(db_path=DB_PATH)
health_alerts = monitor.check_all(engine.portfolio, datetime(2020, 12, 31))
print(f"\nHealth check: {len(health_alerts)} alerts")
for a in health_alerts:
    print(f"  [{a['severity']}] {a['message']}")

# ── Generate tearsheet ─────────────────────────────────────────────────────────
equity = load_equity_curve_from_db(DB_PATH)
trades = load_trades_from_db(DB_PATH)

tearsheet_path = "/tmp/quantsim_example_04_tearsheet.html"
if not equity.empty:
    generate_tearsheet(
        equity_curve=equity,
        strategy_name="SMA 20/100 on SPY (with trailing stop)",
        trades_df=trades if not trades.empty else None,
        output_path=tearsheet_path,
        initial_capital=100_000,
    )
    print(f"\nTearsheet saved: {tearsheet_path}")
    print("Open in any browser for the full interactive report.")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Total Return:  {results.get('total_return', 0):+.2%}")
print(f"Sharpe Ratio:  {results.get('sharpe_ratio', 0):.3f}")
print(f"Deflated SR:   {results.get('deflated_sharpe_ratio', 0):.3f}")
print(f"Max Drawdown:  {results.get('max_drawdown', 0):.2%}")
print(f"GARCH-tracked assets: {len(engine.garch_adapter.get_all_forecasts())}")
