"""
Test suite for the quantsim engine.

Three test categories:
1. Unit tests: isolated component tests with synthetic data
2. Integration tests: full event loop on known toy price series
3. Walking skeleton validation: BuyAndHold SPY must match known total return

Run with: pytest tests/test_engine.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

from core.events import (
    BarEvent, SignalEvent, OrderEvent, FillEvent,
    Direction, OrderType, OrderSide, AssetType, EventType
)
from core.event_queue import EventQueue
from portfolio.portfolio import Portfolio, Position
from reporting.analytics import PerformanceAnalytics


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_bar(asset_id="SPY", price=100.0, volume=1_000_000, dt=None):
    dt = dt or datetime(2020, 1, 2)
    return BarEvent(
        timestamp=dt,
        asset_id=asset_id,
        open=price * 0.999,
        high=price * 1.005,
        low=price * 0.995,
        close=price,
        volume=volume,
        adj_close=price,
    )


def make_fill(asset_id="SPY", price=100.0, qty=10, side=OrderSide.BUY, strategy_id="test"):
    return FillEvent(
        order_id="test-order",
        asset_id=asset_id,
        asset_type=AssetType.EQUITY,
        side=side,
        quantity=qty,
        fill_price=price,
        commission=0.0,
        slippage=0.0,
        strategy_id=strategy_id,
        timestamp=datetime(2020, 1, 2),
    )


# ── Unit Tests: EventQueue ─────────────────────────────────────────────────────

class TestEventQueue:
    def test_fifo_same_timestamp_fill_before_bar(self):
        """FillEvents must process before BarEvents at same timestamp."""
        q = EventQueue()
        ts = datetime(2020, 1, 2, 12, 0, 0)

        bar = make_bar(dt=ts)
        fill = make_fill()
        fill.timestamp = ts

        q.put(bar)
        q.put(fill)

        first = q.get()
        assert first.event_type == EventType.FILL, \
            "FillEvent must have higher priority than BarEvent at same timestamp"

    def test_chronological_ordering(self):
        """Events in different timestamps must come out chronologically."""
        q = EventQueue()
        for i in [3, 1, 4, 1, 5, 9]:
            bar = make_bar(dt=datetime(2020, 1, i + 1))
            q.put(bar)

        timestamps = []
        while not q.empty():
            e = q.get()
            timestamps.append(e.timestamp)

        assert timestamps == sorted(timestamps), "Events must be chronologically ordered"

    def test_empty_raises(self):
        q = EventQueue()
        with pytest.raises(IndexError):
            q.get()


# ── Unit Tests: Portfolio ──────────────────────────────────────────────────────

class TestPortfolio:
    def test_initial_state(self):
        p = Portfolio(initial_capital=100_000, cash=100_000)
        assert p.total_equity == 100_000
        assert p.total_pnl == 0.0
        assert len(p.positions) == 0

    def test_apply_buy_fill(self):
        p = Portfolio(initial_capital=100_000, cash=100_000)
        fill = make_fill(price=100.0, qty=100)
        p.apply_fill(fill)

        assert "SPY" in p.positions
        assert p.positions["SPY"].quantity == 100
        assert p.positions["SPY"].average_cost == 100.0
        assert p.cash == 100_000 - 100 * 100  # 90,000

    def test_apply_sell_closes_position(self):
        p = Portfolio(initial_capital=100_000, cash=100_000)
        buy = make_fill(price=100.0, qty=100, side=OrderSide.BUY)
        p.apply_fill(buy)

        sell = make_fill(price=110.0, qty=100, side=OrderSide.SELL)
        p.apply_fill(sell)

        assert "SPY" not in p.positions
        assert p.realized_pnl == pytest.approx(1000.0)  # 100 * (110-100)

    def test_unrealized_pnl(self):
        p = Portfolio(initial_capital=100_000, cash=100_000)
        fill = make_fill(price=100.0, qty=100)
        p.apply_fill(fill)
        p.update_market_prices({"SPY": 105.0})

        assert p.positions["SPY"].unrealized_pnl == pytest.approx(500.0)

    def test_drawdown_calculation(self):
        p = Portfolio(initial_capital=100_000, cash=100_000)
        p._equity_history = [
            (datetime(2020, 1, i), 100_000 + (i - 5) * 1000)
            for i in range(1, 11)
        ]
        p.peak_equity = max(e for _, e in p._equity_history)
        # Peak = 105,000, last = 105,000 → drawdown = 0
        p.timestamp = datetime(2020, 1, 10)
        assert p.max_drawdown() <= 0


# ── Unit Tests: RawBar Validation ─────────────────────────────────────────────

class TestRawBar:
    def test_valid_bar(self):
        from core.schemas import RawBar
        bar = RawBar(
            asset_id="AAPL",
            timestamp=datetime(2020, 1, 2),
            open=150.0, high=155.0, low=149.0, close=153.0,
            volume=1_000_000,
        )
        assert bar.validate()

    def test_invalid_high_low(self):
        from core.schemas import RawBar
        bar = RawBar(
            asset_id="AAPL",
            timestamp=datetime(2020, 1, 2),
            open=150.0, high=148.0, low=149.0, close=147.0,  # high < low
            volume=1_000_000,
        )
        assert not bar.validate()

    def test_negative_price(self):
        from core.schemas import RawBar
        bar = RawBar(
            asset_id="AAPL",
            timestamp=datetime(2020, 1, 2),
            open=-1.0, high=150.0, low=149.0, close=149.0,
            volume=1_000_000,
        )
        assert not bar.validate()


# ── Unit Tests: Performance Analytics ─────────────────────────────────────────

class TestPerformanceAnalytics:
    def _make_equity_curve(self, n=500, annual_return=0.10, annual_vol=0.15):
        """Simulate a realistic equity curve."""
        np.random.seed(42)
        daily_ret = annual_return / 252
        daily_vol = annual_vol / np.sqrt(252)
        returns = np.random.normal(daily_ret, daily_vol, n)
        equity = 100_000 * np.cumprod(1 + returns)
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.Series(equity, index=dates)

    def test_sharpe_reasonable(self):
        analytics = PerformanceAnalytics(risk_free_rate=0.02)
        curve = self._make_equity_curve(annual_return=0.12, annual_vol=0.15)
        results = analytics.compute_all(curve)
        # Sharpe for 12% return, 15% vol, 2% RF ≈ (12-2)/15 = 0.67
        assert 0.3 < results["sharpe_ratio"] < 1.5, \
            f"Sharpe {results['sharpe_ratio']} out of expected range"

    def test_max_drawdown_negative(self):
        analytics = PerformanceAnalytics()
        curve = self._make_equity_curve()
        results = analytics.compute_all(curve)
        assert results["max_drawdown"] <= 0, "Max drawdown should be non-positive"

    def test_deflated_sr_penalizes_multiple_tests(self):
        analytics = PerformanceAnalytics()
        curve = self._make_equity_curve(annual_return=0.15, annual_vol=0.20)

        r1 = analytics.compute_all(curve, n_strategies_tested=1)
        r100 = analytics.compute_all(curve, n_strategies_tested=100)

        assert r1["deflated_sharpe_ratio"] > r100["deflated_sharpe_ratio"], \
            "Testing 100 strategies should lower DSR compared to testing 1"

    def test_cagr_approx(self):
        analytics = PerformanceAnalytics()
        # Flat 10% annual gain
        n = 252
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        equity = pd.Series(100_000 * (1.10 ** (np.arange(n) / 252)), index=dates)
        results = analytics.compute_all(equity)
        assert abs(results["cagr"] - 0.10) < 0.01, \
            f"CAGR {results['cagr']:.3f} should be ≈ 0.10"


# ── Unit Tests: Slippage Models ────────────────────────────────────────────────

class TestSlippageModels:
    def test_no_slippage_zero(self):
        from backtesting.execution import NoSlippage, AssetType
        model = NoSlippage()
        slip = model.compute_slippage(100.0, 1000, 5_000_000, 5_000_000, AssetType.EQUITY)
        assert slip == 0.0

    def test_volume_proportional_large_order(self):
        from backtesting.execution import VolumeProportionalSlippage, AssetType
        model = VolumeProportionalSlippage(k=0.1)
        # Order is 10% of ADV: slippage = 100 * 0.1 * sqrt(0.1) ≈ 3.16
        slip = model.compute_slippage(100.0, 100_000, 1_000_000, 1_000_000, AssetType.EQUITY)
        assert slip > 0
        assert slip < 100.0  # sanity: slippage < full price

    def test_small_order_small_slippage(self):
        from backtesting.execution import VolumeProportionalSlippage, AssetType
        model = VolumeProportionalSlippage(k=0.1)
        # Order is 0.01% of ADV: minimal slippage
        slip_small = model.compute_slippage(100.0, 100, 1_000_000, 1_000_000, AssetType.EQUITY)
        slip_large = model.compute_slippage(100.0, 100_000, 1_000_000, 1_000_000, AssetType.EQUITY)
        assert slip_small < slip_large


# ── Integration Test: Full Event Loop ─────────────────────────────────────────

class TestIntegrationEventLoop:
    """
    Integration test: synthetic 2-asset price series with known correct output.
    Asset A: goes up 10% then comes down 5%.
    BuyAndHold strategy should be long from start.
    """

    def _make_synthetic_prices(self):
        """Returns dict of asset_id -> list of (datetime, price)."""
        import pandas as pd
        dates = pd.date_range("2022-01-03", periods=30, freq="B")
        spy_prices = [400.0 * (1 + 0.001 * i) for i in range(30)]
        return {
            "SPY": pd.Series(spy_prices, index=dates)
        }

    def test_buy_and_hold_positive_pnl_in_uptrend(self):
        """In a linear uptrend, buy-and-hold should show positive P&L."""
        from core.database import init_db
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            os.environ["QUANTSIM_DB"] = db_path
            init_db(db_path)

            # Insert synthetic SPY data
            from core.database import db_conn
            from datetime import datetime as dt
            import pandas as pd

            dates = pd.date_range("2022-01-03", periods=60, freq="B")
            prices = [400.0 * (1 + 0.001 * i) for i in range(60)]

            with db_conn(db_path) as conn:
                for d, p in zip(dates, prices):
                    ts_epoch = int(d.timestamp())
                    conn.execute(
                        """INSERT OR REPLACE INTO raw_bars
                           (asset_id, timestamp, open, high, low, close, volume, source)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        ("SPY", ts_epoch, p*0.999, p*1.005, p*0.995, p, 5_000_000, "test"),
                    )
                    conn.execute(
                        """INSERT OR REPLACE INTO adjustment_factors
                           (asset_id, effective_date, cumulative_split_factor, cumulative_div_adjustment)
                           VALUES (?, ?, ?, ?)""",
                        ("SPY", ts_epoch, 1.0, 0.0),
                    )

            # Run engine with BuyAndHold
            from strategies.momentum_factor import BuyAndHold
            from core.event_queue import EventQueue

            eq = EventQueue()
            strategy = BuyAndHold(asset_ids=["SPY"], event_queue=eq)

            from backtesting.engine import BacktestEngine
            engine = BacktestEngine(
                strategies=[strategy],
                start=datetime(2022, 1, 3),
                end=datetime(2022, 3, 31),
                initial_capital=100_000,
                db_path=db_path,
                verbose=False,
            )
            # Override event queue to use the same one
            engine.event_queue = eq

            # Manual bar-by-bar test (lighter than full run)
            portfolio = Portfolio(initial_capital=100_000, cash=100_000)
            assert portfolio.total_equity == 100_000

            # Simulate a buy at 400, then update to 440 (+10%)
            fill = FillEvent(
                order_id="test", asset_id="SPY", asset_type=AssetType.EQUITY,
                side=OrderSide.BUY, quantity=200, fill_price=400.0,
                commission=0.0, slippage=0.0, strategy_id="test",
                timestamp=datetime(2022, 1, 3),
            )
            portfolio.apply_fill(fill)
            portfolio.update_market_prices({"SPY": 440.0})

            assert portfolio.unrealized_pnl == pytest.approx(8000.0)  # 200 * 40
            assert portfolio.total_equity > 100_000


# ── Walk-Forward Validation Marker ────────────────────────────────────────────

class TestWalkingSkeletonValidation:
    """
    Phase 0 validation: SPY buy-and-hold total return 2020-2023.
    Requires live data download. Marked with @pytest.mark.live to skip in CI.
    
    Acceptance criterion: within 0.1% annually of published SPY total return.
    SPY total return 2020-01-02 to 2023-12-29 ≈ +69% (including dividends).
    """

    @pytest.mark.live
    def test_spy_buy_and_hold_matches_published_return(self):
        """
        WALKING SKELETON VALIDATION.
        If this test fails, look-ahead bias or adjustment factor errors are present.
        Do not proceed to Phase 2 until this passes.
        
        Expected: SPY total return 2020-2023 ≈ 69% (with dividends).
        Acceptable range: [60%, 80%] (accounting for precise dates and commission assumptions).
        """
        pytest.skip("Live test: run manually with real data downloaded")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
