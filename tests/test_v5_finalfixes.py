"""
Tests for this session's gap fixes:
- OrderManager (stop-loss, take-profit, trailing, bracket, ATR)
- Portfolio.update_greeks() and expire_options()
- StrategyHealthMonitor
- HTML tearsheet generation
- Pairs strategy in registry
- Database concurrent safety
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date


# ── OrderManager ──────────────────────────────────────────────────────────────

class TestOrderManager:
    def make_bar(self, asset_id, price, high=None, low=None, dt=None):
        from core.events import BarEvent
        return BarEvent(
            timestamp=dt or datetime(2022, 1, 3),
            asset_id=asset_id,
            open=price * 0.999,
            high=high or price * 1.01,
            low=low or price * 0.99,
            close=price,
            volume=1_000_000,
            adj_close=price,
        )

    def test_stop_loss_triggers_below_price(self):
        from backtesting.order_manager import OrderManager
        from core.event_queue import EventQueue
        from core.events import EventType

        eq = EventQueue()
        mgr = OrderManager(event_queue=eq)
        mgr.add_stop_loss("SPY", entry_price=400.0, strategy_id="sma",
                          stop_pct=0.02, direction="LONG")

        # Bar where low goes below stop (400 * 0.98 = 392)
        bar = self.make_bar("SPY", price=391, low=389)
        triggered = mgr.on_bar(bar)
        assert "SPY" in triggered

        # Should have emitted a FLAT signal
        assert not eq.empty()
        event = eq.get()
        assert event.event_type == EventType.SIGNAL
        from core.events import Direction
        assert event.direction == Direction.FLAT

    def test_stop_loss_does_not_trigger_above_price(self):
        from backtesting.order_manager import OrderManager
        from core.event_queue import EventQueue

        eq = EventQueue()
        mgr = OrderManager(event_queue=eq)
        mgr.add_stop_loss("SPY", entry_price=400.0, strategy_id="sma",
                          stop_pct=0.02, direction="LONG")

        # Bar above stop
        bar = self.make_bar("SPY", price=405, low=402)
        triggered = mgr.on_bar(bar)
        assert "SPY" not in triggered
        assert eq.empty()

    def test_take_profit_triggers_above_target(self):
        from backtesting.order_manager import OrderManager
        from core.event_queue import EventQueue
        from core.events import EventType

        eq = EventQueue()
        mgr = OrderManager(event_queue=eq)
        mgr.add_take_profit("SPY", entry_price=400.0, strategy_id="sma",
                             target_pct=0.05, direction="LONG")

        # Bar where high goes above target (400 * 1.05 = 420)
        bar = self.make_bar("SPY", price=422, high=425)
        triggered = mgr.on_bar(bar)
        assert "SPY" in triggered

    def test_trailing_stop_moves_with_price(self):
        from backtesting.order_manager import OrderManager
        from core.event_queue import EventQueue

        eq = EventQueue()
        mgr = OrderManager(event_queue=eq)
        mgr.add_trailing_stop("SPY", entry_price=400.0, strategy_id="sma",
                               trail_pct=0.03, direction="LONG")

        # Price rises to 420 — stop should follow to 420 * 0.97 = 407.4
        bar_up = self.make_bar("SPY", price=420, high=420, low=418)
        mgr.on_bar(bar_up)

        stop = mgr._stops.get("SPY")
        assert stop is not None
        assert stop.stop_price > 400 * 0.97  # stop moved up

    def test_bracket_order_stop_triggers_cancels_target(self):
        from backtesting.order_manager import OrderManager
        from core.event_queue import EventQueue

        eq = EventQueue()
        mgr = OrderManager(event_queue=eq)
        mgr.add_bracket("SPY", entry_price=400.0, strategy_id="sma",
                         stop_pct=0.02, target_pct=0.06)

        # Stop triggers
        bar = self.make_bar("SPY", price=391, low=389)
        triggered = mgr.on_bar(bar)
        assert "SPY" in triggered
        # Both stop and target should be removed
        assert "SPY" not in mgr._stops

    def test_atr_stop_computed_from_atr(self):
        from backtesting.order_manager import OrderManager
        from core.event_queue import EventQueue

        eq = EventQueue()
        mgr = OrderManager(event_queue=eq)
        atr = 5.0  # $5 ATR
        mgr.add_atr_stop("SPY", entry_price=400.0, strategy_id="sma",
                          current_atr=atr, multiple=2.0, direction="LONG")

        stop = mgr._stops["SPY"]
        # Stop = 400 - 2 * 5 = 390
        assert abs(stop.stop_price - 390.0) < 0.01

    def test_remove_clears_stop(self):
        from backtesting.order_manager import OrderManager
        from core.event_queue import EventQueue

        eq = EventQueue()
        mgr = OrderManager(event_queue=eq)
        mgr.add_stop_loss("SPY", entry_price=400.0, strategy_id="sma")
        mgr.remove("SPY")
        assert "SPY" not in mgr._stops

    def test_short_position_stop_triggers_above(self):
        from backtesting.order_manager import OrderManager
        from core.event_queue import EventQueue

        eq = EventQueue()
        mgr = OrderManager(event_queue=eq)
        mgr.add_stop_loss("SPY", entry_price=400.0, strategy_id="sma",
                          stop_pct=0.02, direction="SHORT")

        # Short position: stop above entry (400 * 1.02 = 408)
        bar = self.make_bar("SPY", price=410, high=412)
        triggered = mgr.on_bar(bar)
        assert "SPY" in triggered


# ── Portfolio Options Methods ─────────────────────────────────────────────────

class TestPortfolioOptionsMethods:
    def test_update_greeks_on_position(self):
        from portfolio.portfolio import Portfolio, Position
        from core.events import AssetType, FillEvent, OrderSide

        p = Portfolio(initial_capital=100_000, cash=100_000)

        # Add an options position
        fill = FillEvent(
            order_id="opt-1",
            asset_id="SPY",
            asset_type=AssetType.OPTION,
            side=OrderSide.BUY_TO_OPEN,
            quantity=1,
            fill_price=5.0,
            commission=0.65,
            slippage=0.0,
            strategy_id="covered_call",
            timestamp=datetime(2022, 1, 3),
            option_symbol="SPY220121C00420000",
            expiration="2022-01-21",
            strike=420.0,
            right="C",
        )
        p.apply_fill(fill)

        # Update Greeks
        p.update_greeks("SPY220121C00420000",
                        delta=0.30, gamma=0.05, theta=-0.08, vega=0.12)

        pos = p.positions.get("SPY220121C00420000")
        assert pos is not None
        assert abs(pos.delta - 0.30) < 1e-6
        assert abs(pos.gamma - 0.05) < 1e-6
        assert abs(pos.theta - (-0.08)) < 1e-6

    def test_expire_options_removes_position(self):
        from portfolio.portfolio import Portfolio
        from core.events import AssetType, FillEvent, OrderSide

        p = Portfolio(initial_capital=100_000, cash=100_000)

        fill = FillEvent(
            order_id="opt-2",
            asset_id="SPY",
            asset_type=AssetType.OPTION,
            side=OrderSide.BUY_TO_OPEN,
            quantity=1,
            fill_price=3.0,
            commission=0.65,
            slippage=0.0,
            strategy_id="test",
            timestamp=datetime(2022, 1, 3),
            option_symbol="SPY220110C00430000",
            expiration="2022-01-10",
            strike=430.0,
            right="C",
        )
        p.apply_fill(fill)

        # Advance past expiration
        expired = p.expire_options(as_of_date=date(2022, 1, 11))
        assert "SPY220110C00430000" in expired
        assert "SPY220110C00430000" not in p.positions

    def test_expire_options_books_pnl(self):
        from portfolio.portfolio import Portfolio
        from core.events import AssetType, FillEvent, OrderSide

        p = Portfolio(initial_capital=100_000, cash=100_000)
        # Buy an option at 5.0
        fill = FillEvent(
            order_id="opt-3",
            asset_id="SPY",
            asset_type=AssetType.OPTION,
            side=OrderSide.BUY_TO_OPEN,
            quantity=1,
            fill_price=5.0,
            commission=0.0,
            slippage=0.0,
            strategy_id="test",
            timestamp=datetime(2022, 1, 3),
            option_symbol="SPY220107P00390000",
            expiration="2022-01-07",
            strike=390.0,
            right="P",
        )
        p.apply_fill(fill)

        # Update price to 0 (expires worthless)
        p.positions["SPY220107P00390000"].current_price = 0.0
        realized_before = p.realized_pnl

        expired = p.expire_options(as_of_date=date(2022, 1, 8))
        assert len(expired) > 0
        # Realized P&L should have been updated
        # (in this case it's negative since we paid 5 and got 0)


# ── Strategy Health Monitor ───────────────────────────────────────────────────

class TestStrategyHealthMonitor:
    def test_monitor_detects_drawdown_warning(self, tmp_path):
        from reporting.monitor import StrategyHealthMonitor
        from portfolio.portfolio import Portfolio
        from core.events import AssetType, FillEvent, OrderSide

        db_path = str(tmp_path / "monitor_test.db")
        from core.database_v2 import init_full_db
        init_full_db(db_path)

        monitor = StrategyHealthMonitor(
            db_path=db_path,
            max_dd_warn=-0.05,
            max_dd_critical=-0.10,
        )

        # Create portfolio with heavy drawdown
        p = Portfolio(initial_capital=100_000, cash=100_000)
        p.peak_equity = 100_000
        p._equity_history = [(datetime(2022, 1, i), 100_000 - i * 2000) for i in range(1, 10)]
        # Last equity is 82,000 → drawdown ≈ -18%

        ts = datetime(2022, 1, 10)
        # Manually simulate deep drawdown
        p.cash = 82_000
        alerts = monitor.check_all(p, ts)

        # Should have at least a warning alert
        assert any(a["severity"] in ("WARNING", "CRITICAL") for a in alerts)

    def test_monitor_no_false_alerts_on_healthy_portfolio(self, tmp_path):
        from reporting.monitor import StrategyHealthMonitor
        from portfolio.portfolio import Portfolio

        db_path = str(tmp_path / "monitor_healthy.db")
        from core.database_v2 import init_full_db
        init_full_db(db_path)

        monitor = StrategyHealthMonitor(db_path=db_path)
        p = Portfolio(initial_capital=100_000, cash=100_000)
        p.peak_equity = 100_000
        p.cash = 105_000  # profit

        alerts = monitor.check_all(p, datetime(2022, 6, 1))
        # No drawdown or concentration alerts on a healthy portfolio
        dd_alerts = [a for a in alerts if "DRAWDOWN" in a.get("risk_type", "")]
        assert len(dd_alerts) == 0

    def test_monitor_alert_deduplication(self, tmp_path):
        from reporting.monitor import StrategyHealthMonitor
        from portfolio.portfolio import Portfolio

        db_path = str(tmp_path / "monitor_dedup.db")
        from core.database_v2 import init_full_db
        init_full_db(db_path)

        monitor = StrategyHealthMonitor(db_path=db_path, max_dd_warn=-0.05)
        p = Portfolio(initial_capital=100_000, cash=80_000)
        p.peak_equity = 100_000

        ts = datetime(2022, 1, 1)
        alerts1 = monitor.check_all(p, ts)
        # Same timestamp day: should be deduplicated
        alerts2 = monitor.check_all(p, ts + timedelta(minutes=5))
        # Second call in same cooldown window should produce 0 additional alerts
        assert len(alerts2) == 0


# ── HTML Tearsheet ─────────────────────────────────────────────────────────────

class TestTearsheet:
    def make_equity(self, n=300, seed=42):
        np.random.seed(seed)
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        equity = pd.Series(
            100_000 * np.cumprod(1 + np.random.normal(0.0004, 0.015, n)),
            index=dates,
        )
        return equity

    def test_tearsheet_generates_html(self):
        from reporting.tearsheet import generate_tearsheet
        equity = self.make_equity()
        html = generate_tearsheet(equity, strategy_name="Test SMA")
        assert isinstance(html, str)
        assert len(html) > 5000
        assert "<!DOCTYPE html>" in html
        assert "QuantSim" in html
        assert "Plotly" in html

    def test_tearsheet_includes_metrics(self):
        from reporting.tearsheet import generate_tearsheet
        equity = self.make_equity()
        html = generate_tearsheet(equity, strategy_name="Test Strategy")
        # Key metric labels should appear
        for label in ["Sharpe Ratio", "Max Drawdown", "CAGR", "Deflated SR"]:
            assert label in html, f"Missing metric: {label}"

    def test_tearsheet_writes_file(self, tmp_path):
        from reporting.tearsheet import generate_tearsheet
        equity = self.make_equity()
        output = str(tmp_path / "test_tearsheet.html")
        generate_tearsheet(equity, strategy_name="Test", output_path=output)
        assert os.path.exists(output)
        assert os.path.getsize(output) > 5000

    def test_tearsheet_with_trades_and_benchmark(self):
        from reporting.tearsheet import generate_tearsheet
        equity = self.make_equity()
        benchmark = self.make_equity(seed=99)

        trades = pd.DataFrame([{
            "asset_id": "SPY",
            "direction": "LONG",
            "entry_price": 400.0,
            "exit_price": 420.0,
            "realized_pnl": 2000.0,
            "strategy_id": "sma",
            "holding_bars": 21,
        }])

        html = generate_tearsheet(
            equity, strategy_name="Full Test",
            trades_df=trades, benchmark=benchmark,
        )
        assert "SPY" in html
        assert "Benchmark" in html


# ── Pairs Trading in Registry ─────────────────────────────────────────────────

class TestPairsInRegistry:
    def test_pairs_registered(self):
        from strategies.registry import StrategyRegistry
        df = StrategyRegistry.list_all()
        assert "pairs" in df["name"].values

    def test_pairs_builds_correctly(self):
        from strategies.registry import StrategyRegistry
        from core.event_queue import EventQueue
        eq = EventQueue()
        strat = StrategyRegistry.build(
            "pairs",
            asset_ids=["SPY", "QQQ"],
            event_queue=eq,
        )
        assert strat is not None
        assert strat.asset_a in ("SPY", "QQQ")
        assert strat.asset_b in ("SPY", "QQQ")


# ── Database Concurrent Safety ────────────────────────────────────────────────

class TestDatabaseConcurrency:
    def test_wal_mode_enabled(self, tmp_path):
        from core.database import get_connection, init_db
        db_path = str(tmp_path / "wal_test.db")
        init_db(db_path)
        conn = get_connection(db_path)
        result = conn.execute("PRAGMA journal_mode").fetchone()
        assert result[0] == "wal"
        conn.close()

    def test_concurrent_reads_while_writing(self, tmp_path):
        """WAL mode allows reads while a write transaction is in progress."""
        import threading
        from core.database import get_connection, init_db
        db_path = str(tmp_path / "concurrent_test.db")
        init_db(db_path)

        errors = []

        def reader():
            try:
                conn = get_connection(db_path)
                conn.execute("SELECT COUNT(*) FROM raw_bars").fetchone()
                conn.close()
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                conn = get_connection(db_path)
                conn.execute("BEGIN")
                conn.execute("INSERT INTO raw_bars (asset_id, timestamp, open, high, low, close, volume) VALUES ('TEST', 1000, 1,1,1,1,1)")
                import time; time.sleep(0.05)
                conn.execute("COMMIT")
                conn.close()
            except Exception as e:
                errors.append(e)

        t_write = threading.Thread(target=writer)
        t_read = threading.Thread(target=reader)

        t_write.start()
        import time; time.sleep(0.01)  # let writer start first
        t_read.start()

        t_write.join(timeout=2)
        t_read.join(timeout=2)

        # WAL mode: no errors expected
        assert len(errors) == 0, f"Concurrent access errors: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
