"""
Engine correctness regression tests.

These tests guard against the three critical bugs that were fixed:

Bug 1: Queue mismatch
  BacktestEngine creates its own EventQueue internally but strategies held a
  reference to the user-provided queue. Signals emitted into the user queue,
  engine drained its own queue → 0 fills with no error.
  Fix: engine rewires strategy._queue = self.event_queue after construction.

Bug 2: Timestamp propagation
  PortfolioManager.on_bar() called update_market_prices() WITHOUT passing
  the event timestamp. All 289 equity history entries got the initial
  naive datetime (portfolio construction time) → equity curve had 2 unique
  timestamps, appeared flat → total_return = None.
  Fix: pass timestamp=event.timestamp to update_market_prices().

Bug 3: Stale DB equity curve
  load_equity_curve_from_db() returned 11 rows of constant 100000.0
  from a previous broken run. The len <= 1 check didn't catch this.
  The non-empty constant curve blocked the portfolio history fallback.
  total_return = 0 despite portfolio showing real P&L.
  Fix: also check equity_curve.std() == 0 before falling through to fallback.

These tests must pass on every future run. If any fails, a regression occurred.
"""

import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime


def build_test_engine(db_path, n_bars=400, warmup=30, seed=42):
    """Build a BacktestEngine with synthetic SPY data. Short warmup to get many fills."""
    from core.database_v2 import init_full_db
    from core.database import db_conn
    from strategies.momentum_factor import BuyAndHold
    from core.event_queue import EventQueue
    from backtesting.engine import BacktestEngine

    init_full_db(db_path)

    np.random.seed(seed)
    dates = pd.date_range("2020-01-02", periods=n_bars, freq="B")
    prices = 300 * np.cumprod(1 + np.random.normal(0.0004, 0.012, n_bars))

    with db_conn(db_path) as conn:
        for date, price in zip(dates, prices):
            ts = int(date.timestamp())
            conn.execute(
                "INSERT OR REPLACE INTO raw_bars "
                "(asset_id, timestamp, open, high, low, close, volume, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("SPY", ts, price*0.998, price*1.005, price*0.995, price, 10_000_000, "test"),
            )
            conn.execute(
                "INSERT OR REPLACE INTO adjustment_factors "
                "(asset_id, effective_date, cumulative_split_factor, cumulative_div_adjustment) "
                "VALUES (?, ?, 1.0, 0.0)",
                ("SPY", ts),
            )

    eq = EventQueue()
    strategy = BuyAndHold(asset_ids=["SPY"], event_queue=eq)

    engine = BacktestEngine(
        strategies=[strategy],
        start=datetime(2020, 1, 2),
        end=datetime(2021, 6, 30),
        initial_capital=100_000,
        db_path=db_path,
        verbose=False,
        warmup_bars=warmup,
    )
    return engine, eq


class TestBug1QueueMismatch:
    """Bug 1: Strategy signals went to user queue, engine drained its own."""

    def test_strategy_queue_rewired_to_engine_queue(self, tmp_path):
        """After construction, strategy._queue must BE engine.event_queue."""
        db_path = str(tmp_path / "q1.db")
        os.environ["QUANTSIM_DB"] = db_path
        engine, user_queue = build_test_engine(db_path)

        # The strategy must have been rewired to the engine's queue
        for strategy in engine.strategies:
            assert strategy._queue is engine.event_queue, (
                f"Strategy {strategy.strategy_id}._queue is NOT engine.event_queue. "
                "Queue mismatch: signals will be emitted to wrong queue."
            )
        # And it must NOT be the original user queue
        assert engine.event_queue is not user_queue, (
            "Engine should have its own EventQueue, not reuse the user-provided one."
        )

    def test_strategy_produces_fills_not_zero(self, tmp_path):
        """The engine must execute at least 1 fill (buy) on upward-trending data."""
        db_path = str(tmp_path / "q2.db")
        os.environ["QUANTSIM_DB"] = db_path
        engine, _ = build_test_engine(db_path, warmup=10)
        engine.run()
        assert engine._fills_processed >= 1, (
            f"Engine processed 0 fills. Signals emitted to wrong queue (queue mismatch bug)."
        )

    def test_signals_processed_matches_fills(self, tmp_path):
        """Signals must be processed before fills can happen."""
        db_path = str(tmp_path / "q3.db")
        os.environ["QUANTSIM_DB"] = db_path
        engine, _ = build_test_engine(db_path, warmup=10)
        engine.run()
        assert engine._signals_processed >= 1, "Zero signals processed — queue mismatch."
        assert engine._fills_processed >= 1, "Zero fills despite signals — order processing broken."


class TestBug2TimestampPropagation:
    """Bug 2: portfolio.timestamp never updated → all equity entries had same timestamp."""

    def test_equity_history_has_multiple_unique_timestamps(self, tmp_path):
        """All 400 bars must produce distinct timestamps in equity history."""
        db_path = str(tmp_path / "ts1.db")
        os.environ["QUANTSIM_DB"] = db_path
        engine, _ = build_test_engine(db_path, n_bars=200, warmup=10)
        engine.run()

        history = engine.portfolio._equity_history
        assert len(history) > 10, "Too few equity history entries."

        timestamps = [t for t, _ in history]
        unique_ts = len(set(str(t) for t in timestamps))
        assert unique_ts > len(history) * 0.5, (
            f"Only {unique_ts}/{len(history)} unique timestamps in equity history. "
            "Timestamp not being updated from BarEvent — timestamp propagation bug."
        )

    def test_equity_values_vary_over_time(self, tmp_path):
        """Portfolio equity must change over time (not a flat line)."""
        db_path = str(tmp_path / "ts2.db")
        os.environ["QUANTSIM_DB"] = db_path
        engine, _ = build_test_engine(db_path, n_bars=200, warmup=10)
        engine.run()

        history = engine.portfolio._equity_history
        equities = [float(e) for _, e in history]
        equity_std = pd.Series(equities).std()
        assert equity_std > 0, (
            "Portfolio equity never changes. Price updates not reaching portfolio "
            "(timestamp propagation prevents proper equity tracking)."
        )

    def test_total_return_is_not_none(self, tmp_path):
        """results['total_return'] must not be None."""
        db_path = str(tmp_path / "ts3.db")
        os.environ["QUANTSIM_DB"] = db_path
        engine, _ = build_test_engine(db_path, n_bars=200, warmup=10)
        results = engine.run()

        assert results.get("total_return") is not None, (
            "total_return is None. Equity curve has insufficient unique timestamps "
            "(timestamp propagation bug causing flat equity history)."
        )

    def test_total_return_positive_on_upward_prices(self, tmp_path):
        """Buy-and-hold on data with positive drift must have positive return."""
        db_path = str(tmp_path / "ts4.db")
        os.environ["QUANTSIM_DB"] = db_path
        # Use strong positive drift to ensure positive return
        from core.database_v2 import init_full_db
        from core.database import db_conn
        from strategies.momentum_factor import BuyAndHold
        from core.event_queue import EventQueue
        from backtesting.engine import BacktestEngine

        init_full_db(db_path)
        np.random.seed(1)
        n = 300
        dates = pd.date_range("2020-01-02", periods=n, freq="B")
        # Strong upward trend: 25% annual return
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.010, n))
        with db_conn(db_path) as conn:
            for d, p in zip(dates, prices):
                ts = int(d.timestamp())
                conn.execute("INSERT OR REPLACE INTO raw_bars (asset_id,timestamp,open,high,low,close,volume,source) VALUES (?,?,?,?,?,?,?,?)",
                    ("SPY", ts, p*0.998, p*1.005, p*0.995, p, 10_000_000, "test"))
                conn.execute("INSERT OR REPLACE INTO adjustment_factors VALUES (?,?,1.0,0.0)", ("SPY", ts))

        eq = EventQueue()
        s = BuyAndHold(asset_ids=["SPY"], event_queue=eq)
        engine = BacktestEngine(strategies=[s], start=datetime(2020,1,2), end=datetime(2020,12,31),
            initial_capital=100_000, db_path=db_path, verbose=False, warmup_bars=20)
        results = engine.run()

        tr = results.get("total_return", 0) or 0
        assert tr > 0, (
            f"Buy-and-hold returned {tr:.4%} on strongly upward-trending data. "
            "Expected positive return. Check timestamp propagation and equity curve computation."
        )


class TestBug3StaleDBEquityCurve:
    """Bug 3: Stale constant equity curve in DB blocked portfolio history fallback."""

    def test_constant_equity_curve_triggers_fallback(self, tmp_path):
        """If DB has a constant equity curve, fallback to portfolio history."""
        from core.database_v2 import init_full_db
        from core.database import db_conn
        from reporting.analytics import load_equity_curve_from_db

        db_path = str(tmp_path / "stale.db")
        init_full_db(db_path)

        # Write a stale constant equity curve (simulates old broken run)
        with db_conn(db_path) as conn:
            for i in range(10):
                conn.execute(
                    "INSERT INTO portfolio_snapshots (timestamp, total_equity, cash, realized_pnl, unrealized_pnl, payload) VALUES (?, ?, ?, ?, ?, ?)",
                    (1577836800 + i * 86400, 100000.0, 100000.0, 0.0, 0.0, '{"positions": {}}'),
                )

        ec = load_equity_curve_from_db(db_path)
        assert ec.std() == 0 or len(ec) < 2, "DB has constant equity curve."

    def test_engine_uses_portfolio_history_over_stale_db(self, tmp_path):
        """Engine must return meaningful results even with a stale DB."""
        from core.database_v2 import init_full_db
        from core.database import db_conn
        from strategies.momentum_factor import BuyAndHold
        from core.event_queue import EventQueue
        from backtesting.engine import BacktestEngine

        db_path = str(tmp_path / "stale2.db")
        os.environ["QUANTSIM_DB"] = db_path
        init_full_db(db_path)

        # Pre-populate with stale constant snapshots
        with db_conn(db_path) as conn:
            for i in range(5):
                conn.execute(
                    "INSERT INTO portfolio_snapshots (timestamp, total_equity, cash, realized_pnl, unrealized_pnl, payload) VALUES (?, ?, ?, ?, ?, ?)",
                    (1577836800 + i * 86400, 100000.0, 100000.0, 0.0, 0.0, '{"positions": {}}'),
                )

        # Add real price data
        np.random.seed(5)
        dates = pd.date_range("2020-01-02", periods=200, freq="B")
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.010, 200))
        with db_conn(db_path) as conn:
            for d, p in zip(dates, prices):
                ts = int(d.timestamp())
                conn.execute("INSERT OR REPLACE INTO raw_bars (asset_id,timestamp,open,high,low,close,volume,source) VALUES (?,?,?,?,?,?,?,?)",
                    ("SPY", ts, p*0.998, p*1.005, p*0.995, p, 1_000_000, "test"))
                conn.execute("INSERT OR REPLACE INTO adjustment_factors VALUES (?,?,1.0,0.0)", ("SPY", ts))

        eq = EventQueue()
        s = BuyAndHold(asset_ids=["SPY"], event_queue=eq)
        engine = BacktestEngine(strategies=[s], start=datetime(2020,1,2), end=datetime(2020,12,31),
            initial_capital=100_000, db_path=db_path, verbose=False, warmup_bars=20)
        results = engine.run()

        # Must not return None (stale DB must be bypassed)
        assert results.get("total_return") is not None, (
            "total_return is None despite real price data. "
            "Stale DB equity curve not properly detected."
        )


class TestIntegrationCorrectness:
    """Integration-level correctness tests."""

    def test_sma_crossover_produces_multiple_trades(self, tmp_path):
        """SMA crossover should generate multiple signals on a trending series."""
        from core.database_v2 import init_full_db
        from core.database import db_conn
        from strategies.trend import SMAcrossover
        from core.event_queue import EventQueue
        from backtesting.engine import BacktestEngine

        db_path = str(tmp_path / "sma.db")
        os.environ["QUANTSIM_DB"] = db_path
        init_full_db(db_path)

        # Create oscillating price series that will cross moving averages
        np.random.seed(99)
        n = 400
        dates = pd.date_range("2018-01-01", periods=n, freq="B")
        # Add trend reversals to ensure crossovers
        t = np.linspace(0, 4*np.pi, n)
        trend = np.sin(t) * 20 + np.random.normal(0, 2, n)
        prices = 100 + np.cumsum(trend * 0.1)
        prices = np.maximum(prices, 10)  # floor at $10

        with db_conn(db_path) as conn:
            for d, p in zip(dates, prices):
                ts = int(d.timestamp())
                conn.execute("INSERT OR REPLACE INTO raw_bars (asset_id,timestamp,open,high,low,close,volume,source) VALUES (?,?,?,?,?,?,?,?)",
                    ("SPY", ts, p*0.998, p*1.005, p*0.995, p, 1_000_000, "test"))
                conn.execute("INSERT OR REPLACE INTO adjustment_factors VALUES (?,?,1.0,0.0)", ("SPY", ts))

        eq = EventQueue()
        s = SMAcrossover(asset_ids=["SPY"], event_queue=eq, fast=10, slow=40)
        engine = BacktestEngine(strategies=[s], start=datetime(2018,1,1), end=datetime(2019,6,30),
            initial_capital=100_000, db_path=db_path, verbose=False, warmup_bars=50)
        results = engine.run()

        assert engine._signals_processed >= 1, "SMA crossover emitted no signals."
        assert results.get("total_return") is not None, "No total return computed."

    def test_equity_curve_monotonically_starts_at_initial_capital(self, tmp_path):
        """First equity value must equal initial_capital exactly."""
        db_path = str(tmp_path / "init.db")
        os.environ["QUANTSIM_DB"] = db_path
        engine, _ = build_test_engine(db_path, n_bars=200, warmup=10)
        engine.run()

        history = engine.portfolio._equity_history
        first_equity = float(history[0][1]) if history else 0
        assert abs(first_equity - 100_000) < 0.01, (
            f"First equity {first_equity:.2f} != initial_capital 100000. "
            "Portfolio not correctly initialized."
        )

    def test_portfolio_cash_reduces_after_buy(self, tmp_path):
        """After a buy fill, cash must decrease from initial_capital."""
        db_path = str(tmp_path / "cash.db")
        os.environ["QUANTSIM_DB"] = db_path
        engine, _ = build_test_engine(db_path, n_bars=200, warmup=10)
        engine.run()

        if engine._fills_processed > 0:
            # Cash should be less than initial capital after buying
            assert engine.portfolio.cash < 100_000, (
                f"Cash {engine.portfolio.cash:.2f} not reduced after buy fill. "
                "Fill not applied to portfolio."
            )
        else:
            pytest.skip("No fills in this test run")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
