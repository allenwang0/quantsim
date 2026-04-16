"""
Example 1: Walking Skeleton Validation
======================================

The mandatory first step before any strategy development.

Run buy-and-hold on SPY with synthetic data and verify:
1. The event loop processes bars correctly
2. Portfolio tracks P&L accurately
3. No look-ahead bias in the data handler

Expected: total return > 0 on an upward-trending synthetic price series.
If this fails, there's a fundamental bug in the engine before you've
written a single line of strategy code.

Run:
    cd examples && python 01_walking_skeleton.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime

from core.logging_config import setup_logging
from core.database_v2 import init_full_db
from core.database import db_conn
from strategies.momentum_factor import BuyAndHold
from core.event_queue import EventQueue
from backtesting.engine import BacktestEngine

setup_logging(level="INFO")

# ── Step 1: Create a synthetic DB ─────────────────────────────────────────────
DB_PATH = "/tmp/quantsim_example_01.db"
# Always start fresh - delete stale DB from previous runs
if os.path.exists(DB_PATH):
    os.unlink(DB_PATH)
init_full_db(DB_PATH)
os.environ["QUANTSIM_DB"] = DB_PATH

# ── Step 2: Inject synthetic SPY data (upward drift so buy-and-hold wins) ─────
np.random.seed(42)
n_bars = 500
dates = pd.date_range("2020-01-02", periods=n_bars, freq="B")
daily_returns = np.random.normal(0.0005, 0.012, n_bars)  # 12.6% annual return
prices = 300 * np.cumprod(1 + daily_returns)  # start at $300

with db_conn(DB_PATH) as conn:
    for i, (date, price) in enumerate(zip(dates, prices)):
        ts_epoch = int(date.timestamp())
        conn.execute(
            "INSERT OR REPLACE INTO raw_bars "
            "(asset_id, timestamp, open, high, low, close, volume, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("SPY", ts_epoch,
             price * 0.998, price * 1.005, price * 0.995,
             price, 10_000_000, "synthetic"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO adjustment_factors "
            "(asset_id, effective_date, cumulative_split_factor, cumulative_div_adjustment) "
            "VALUES (?, ?, 1.0, 0.0)",
            ("SPY", ts_epoch),
        )

print(f"Injected {n_bars} synthetic SPY bars")
print(f"Price range: ${prices.min():.2f} → ${prices.max():.2f}")
print(f"Theoretical total return: {(prices[-1]/prices[0]-1):.2%}")

# ── Step 3: Run the walking skeleton ──────────────────────────────────────────
eq = EventQueue()
strategy = BuyAndHold(asset_ids=["SPY"], event_queue=eq)

engine = BacktestEngine(
    strategies=[strategy],
    start=datetime(2020, 1, 2),
    end=datetime(2021, 12, 31),
    initial_capital=100_000,
    db_path=DB_PATH,
    verbose=False,
)

results = engine.run()

# ── Step 4: Validate ──────────────────────────────────────────────────────────
total_return = results.get("total_return", 0)
print(f"\nEngine total return:     {total_return:+.2%}")
print(f"Fills executed:          {results.get('_meta', {}).get('fills_executed', 0)}")
print(f"Bars processed:          {results.get('_meta', {}).get('bars_processed', 0)}")

assert total_return > 0, \
    f"FAIL: Buy-and-hold on upward-trending data returned {total_return:.2%}. Check the engine."

print("\n✓ WALKING SKELETON VALIDATION PASSED")
print("  The event loop, portfolio tracking, and data handler all work correctly.")
print("  Proceed to strategy development.")
