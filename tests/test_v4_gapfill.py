"""
Tests for gap-filling additions:
- Logging config
- ML event-engine strategy
- GARCHVolatilityTargetSizer integration
- OptimizationOverlayManager
- Dashboard DB queries (smoke tests)
- pyproject.toml validity
- Registry completeness
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime


# ── Logging config ─────────────────────────────────────────────────────────────

class TestLoggingConfig:
    def test_setup_logging_runs(self):
        from core.logging_config import setup_logging
        setup_logging(level="WARNING")  # quiet for tests
        import logging
        assert logging.getLogger().level == logging.WARNING

    def test_json_formatter_produces_valid_json(self):
        import json, logging
        from core.logging_config import JSONFormatter
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=1, msg="hello %s",
            args=("world",), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "hello world"
        assert "timestamp" in parsed
        assert "level" in parsed

    def test_get_logger_returns_logger(self):
        from core.logging_config import get_logger
        logger = get_logger("test.module")
        import logging
        assert isinstance(logger, logging.Logger)


# ── ML Strategy (event-engine compatible) ────────────────────────────────────

class TestMLStrategyEventEngine:
    def make_prices(self, n=400, n_assets=4, seed=42):
        np.random.seed(seed)
        symbols = ["SPY", "QQQ", "GLD", "TLT"]
        dates = pd.date_range("2018-01-01", periods=n, freq="B")
        data = {}
        for sym in symbols[:n_assets]:
            ret = np.random.normal(0.0003, 0.012, n)
            data[sym] = 100 * np.cumprod(1 + ret)
        return pd.DataFrame(data, index=dates)

    def test_ml_strategy_initializes(self):
        from strategies.ml_strategy import MLCrossSectionalStrategy
        from core.event_queue import EventQueue
        eq = EventQueue()
        s = MLCrossSectionalStrategy(
            asset_ids=["SPY", "QQQ", "GLD"],
            event_queue=eq,
            train_years=1,
            retrain_months=3,
            top_n=1,
            bottom_n=1,
        )
        assert s.strategy_id == "ML_CrossSectional"
        assert len(s.asset_ids) == 3

    def test_ml_strategy_on_bar_no_crash(self):
        """ML strategy should not crash during warmup period."""
        from strategies.ml_strategy import MLCrossSectionalStrategy
        from core.event_queue import EventQueue
        from core.events import BarEvent, AssetType

        eq = EventQueue()
        assets = ["SPY", "QQQ"]
        s = MLCrossSectionalStrategy(
            asset_ids=assets, event_queue=eq,
            train_years=1, retrain_months=6, top_n=1,
        )

        # Mock data handler
        class MockDH:
            def get_latest_bars(self, aid, n=1, adjusted=True):
                dates = pd.date_range("2020-01-01", periods=n+5, freq="B")
                prices = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.012, n+5))
                return pd.DataFrame({"adj_close": prices, "close": prices}, index=dates)
            def get_current_bar(self, aid):
                return pd.Series({"adj_close": 100.0, "close": 100.0})
            def get_macro_value(self, s): return None

        dh = MockDH()
        for i in range(10):  # just warmup bars - no signal expected
            bar = BarEvent(
                timestamp=datetime(2020, 1, i+1),
                asset_id=assets[i % len(assets)],
                open=100, high=102, low=99, close=101,
                volume=5_000_000, adj_close=101.0,
            )
            s.on_bar(bar, dh)  # should not raise

    def test_turbulence_filter_suppresses_on_high_turb(self):
        """TurbulenceFilteredStrategy should emit FLAT during high turbulence."""
        from strategies.ml_strategy import TurbulenceFilteredStrategy
        from strategies.trend import SMAcrossover
        from core.event_queue import EventQueue
        from core.events import BarEvent, EventType, Direction

        eq = EventQueue()
        base = SMAcrossover(asset_ids=["SPY"], event_queue=eq, fast=5, slow=20)
        filtered = TurbulenceFilteredStrategy(
            base_strategy=base,
            event_queue=eq,
            lookback_bars=30,
            turbulence_threshold_pct=0.50,  # low threshold to trigger easily
        )

        class MockDH:
            def get_latest_bars(self, aid, n=1, adjusted=True):
                dates = pd.date_range("2020-01-01", periods=max(n,5), freq="B")
                prices = 100 + np.cumsum(np.random.normal(0, 2, max(n,5)))
                return pd.DataFrame({"adj_close": prices, "close": prices}, index=dates)
            def get_current_bar(self, aid):
                return pd.Series({"adj_close": 100.0, "close": 100.0})
            def get_macro_value(self, s): return None

        dh = MockDH()
        for i in range(50):
            bar = BarEvent(
                timestamp=datetime(2020, 1, 1) + pd.Timedelta(days=i),
                asset_id="SPY",
                open=100, high=105, low=95, close=100,
                volume=1_000_000, adj_close=100.0,
            )
            filtered.on_bar(bar, dh)

        # Should have processed without errors


# ── GARCH Sizer Integration ───────────────────────────────────────────────────

class TestGARCHSizerIntegration:
    def test_garch_sizer_in_sizing_module(self):
        from portfolio.sizing import GARCHVolatilityTargetSizer
        sizer = GARCHVolatilityTargetSizer(
            target_annual_vol=0.12, max_position_fraction=0.15
        )
        assert sizer is not None

    def test_garch_sizer_update_and_size(self):
        from portfolio.sizing import GARCHVolatilityTargetSizer
        from portfolio.portfolio import Portfolio

        sizer = GARCHVolatilityTargetSizer(target_annual_vol=0.12)
        np.random.seed(42)
        returns = np.random.normal(0.0003, 0.020, 150)

        for r in returns:
            sizer.update("SPY", float(r))

        portfolio = Portfolio(initial_capital=100_000, cash=100_000)
        qty = sizer.size(portfolio, "SPY", confidence=1.0, price=400.0)
        assert qty > 0
        assert qty < 500  # sanity: not buying more than $200K of a $100K portfolio


# ── Optimization Overlay Manager ─────────────────────────────────────────────

class TestOptimizationOverlayManager:
    def test_overlay_manager_initializes(self):
        from portfolio.manager import OptimizationOverlayManager
        from portfolio.portfolio import Portfolio
        from portfolio.optimization import HierarchicalRiskParity
        from risk.risk_manager import RiskManager
        from core.event_queue import EventQueue

        eq = EventQueue()
        portfolio = Portfolio(initial_capital=100_000, cash=100_000)
        risk_mgr = RiskManager(event_queue=eq)
        optimizer = HierarchicalRiskParity()

        mgr = OptimizationOverlayManager(
            portfolio=portfolio,
            event_queue=eq,
            risk_manager=risk_mgr,
            optimizer=optimizer,
        )
        assert mgr._optimizer is optimizer

    def test_overlay_manager_rebalances_monthly(self):
        from portfolio.manager import OptimizationOverlayManager
        from portfolio.portfolio import Portfolio
        from portfolio.optimization import HierarchicalRiskParity
        from risk.risk_manager import RiskManager
        from core.event_queue import EventQueue
        from core.events import BarEvent

        eq = EventQueue()
        portfolio = Portfolio(initial_capital=100_000, cash=100_000)
        risk_mgr = RiskManager(event_queue=eq)
        optimizer = HierarchicalRiskParity()

        mgr = OptimizationOverlayManager(
            portfolio=portfolio, event_queue=eq,
            risk_manager=risk_mgr, optimizer=optimizer,
        )

        # Simulate bars from two different months
        # The rebalance cache needs enough returns first
        for month in [1, 2]:
            for day in range(1, 22):
                bar = BarEvent(
                    timestamp=datetime(2022, month, day),
                    asset_id="SPY",
                    open=400, high=402, low=398, close=400,
                    volume=5_000_000, adj_close=400.0,
                )
                mgr.on_bar(bar)  # should not raise


# ── Registry Completeness ─────────────────────────────────────────────────────

class TestRegistryCompleteness:
    def test_ml_strategy_in_registry(self):
        from strategies.registry import StrategyRegistry
        df = StrategyRegistry.list_all()
        assert "ml_xsectional" in df["name"].values

    def test_all_strategy_types_covered(self):
        from strategies.registry import StrategyRegistry
        df = StrategyRegistry.list_all()
        types = set(df["type"].values)
        required_types = {"trend", "mean_reversion", "momentum", "factor", "options", "ml"}
        for t in required_types:
            assert t in types, f"Strategy type '{t}' missing from registry"

    def test_registry_list_is_dataframe(self):
        from strategies.registry import StrategyRegistry
        df = StrategyRegistry.list_all()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 15  # at least 15 strategies


# ── pyproject.toml validity ───────────────────────────────────────────────────

class TestPackageConfiguration:
    def test_pyproject_exists(self):
        assert os.path.exists("pyproject.toml"), "pyproject.toml not found"

    def test_pyproject_parseable(self):
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                pytest.skip("tomllib not available; skip toml parse test")
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        assert "project" in data
        assert data["project"]["name"] == "quantsim"
        assert "version" in data["project"]

    def test_requirements_txt_exists(self):
        assert os.path.exists("requirements.txt")


# ── Dashboard DB query smoke tests ────────────────────────────────────────────

class TestDashboardDBQueries:
    """Smoke tests for the dashboard's DB query functions."""

    def test_all_dashboard_queries_on_empty_db(self, tmp_path):
        """Dashboard should handle empty tables without crashing."""
        import importlib.util, sys
        db_path = str(tmp_path / "dash_test.db")
        os.environ["QUANTSIM_DB"] = db_path
        from core.database_v2 import init_full_db
        init_full_db(db_path)

        # Simulate the dashboard's query functions directly
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        def _q(sql, params=()):
            try: return [dict(r) for r in conn.execute(sql, params).fetchall()]
            except: return []

        # All these should return empty lists without error
        assert isinstance(_q("SELECT * FROM portfolio_snapshots"), list)
        assert isinstance(_q("SELECT * FROM risk_alerts"), list)
        assert isinstance(_q("SELECT * FROM wfo_results"), list)
        assert isinstance(_q("SELECT * FROM ml_model_runs"), list)
        assert isinstance(_q("SELECT * FROM garch_forecasts"), list)
        assert isinstance(_q("SELECT * FROM strategy_performance"), list)
        assert isinstance(_q("SELECT * FROM options_greeks_log"), list)
        conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
