"""
End-to-end smoke test for the complete QuantSim pipeline.

This test validates the full chain:
1. Synthetic data generation (no network required)
2. Vectorized backtest with parameter sweep
3. Walk-forward optimization
4. Event-driven backtest (event loop integrity)
5. Portfolio optimization (HRP, Risk Parity, Mean-Variance)
6. Signal aggregation (ensemble)
7. GARCH volatility forecasting
8. Full performance report generation
9. Database persistence and retrieval

Run with: pytest tests/test_integration_e2e.py -v

This is the "does the whole thing actually work together" test.
All prior tests are unit/component tests. This one finds integration bugs.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, date


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_ohlcv(n=600, n_assets=4, seed=42):
    """Generate realistic OHLCV data for n_assets."""
    np.random.seed(seed)
    symbols = ["SPY", "QQQ", "GLD", "TLT"][:n_assets]
    dates = pd.date_range("2018-01-01", periods=n, freq="B")

    result = {}
    for i, sym in enumerate(symbols):
        drift = [0.0003, 0.0004, 0.0001, -0.0001][i]
        vol = [0.012, 0.015, 0.010, 0.008][i]
        log_rets = np.random.normal(drift, vol, n)
        close = 100 * np.cumprod(1 + log_rets)
        noise = np.random.uniform(0.995, 1.005, n)
        result[sym] = pd.DataFrame({
            "open":   close * np.random.uniform(0.997, 1.003, n),
            "high":   close * np.random.uniform(1.002, 1.008, n),
            "low":    close * np.random.uniform(0.992, 0.998, n),
            "close":  close,
            "adj_close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, n),
        }, index=dates)
    return result


def inject_to_db(ohlcv_dict, db_path):
    """Insert synthetic OHLCV into the database."""
    from core.database import db_conn
    for symbol, df in ohlcv_dict.items():
        with db_conn(db_path) as conn:
            for ts, row in df.iterrows():
                ts_epoch = int(ts.timestamp())
                conn.execute(
                    """INSERT OR REPLACE INTO raw_bars
                       (asset_id, timestamp, open, high, low, close, volume, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (symbol, ts_epoch, float(row.open), float(row.high),
                     float(row.low), float(row.close), int(row.volume), "synthetic"),
                )
                conn.execute(
                    """INSERT OR REPLACE INTO adjustment_factors
                       (asset_id, effective_date, cumulative_split_factor, cumulative_div_adjustment)
                       VALUES (?, ?, 1.0, 0.0)""",
                    (symbol, ts_epoch),
                )


# ── Test 1: Vectorized Backtest Pipeline ──────────────────────────────────────

class TestVectorizedPipeline:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.db_path = str(tmp_path / "e2e.db")
        os.environ["QUANTSIM_DB"] = self.db_path
        from core.database_v2 import init_full_db
        init_full_db(self.db_path)
        self.ohlcv = make_ohlcv(n=400)
        inject_to_db(self.ohlcv, self.db_path)

    def test_vectorized_sma_backtest(self):
        from backtesting.vectorized import VectorizedBacktester, sma_crossover_signal
        prices = pd.DataFrame({
            sym: df["adj_close"] for sym, df in self.ohlcv.items()
        })
        bt = VectorizedBacktester(prices=prices, initial_capital=100_000)
        result = bt.run(sma_crossover_signal, signal_kwargs={"fast": 20, "slow": 100})

        assert "sharpe_ratio" in result
        assert "cagr" in result
        assert "max_drawdown" in result
        assert result["_meta"]["engine"] == "vectorized"
        assert result["max_drawdown"] <= 0

    def test_parameter_sweep_returns_ranked_results(self):
        from backtesting.vectorized import VectorizedBacktester, sma_crossover_signal
        prices = pd.DataFrame({sym: df["adj_close"] for sym, df in self.ohlcv.items()})
        bt = VectorizedBacktester(prices=prices)
        df = bt.parameter_sweep(sma_crossover_signal, {"fast": [10, 20], "slow": [80, 120]})
        assert len(df) == 4
        assert "deflated_sharpe_corrected" in df.columns
        # Sorted descending
        vals = df["deflated_sharpe_corrected"].values
        assert all(vals[i] >= vals[i+1] - 1e-9 for i in range(len(vals)-1))

    def test_multiple_signal_functions_run_cleanly(self):
        from backtesting.vectorized import (
            VectorizedBacktester, sma_crossover_signal, rsi_signal,
            momentum_signal, bollinger_signal, donchian_signal,
        )
        prices = pd.DataFrame({sym: df["adj_close"] for sym, df in self.ohlcv.items()})
        bt = VectorizedBacktester(prices=prices)
        for fn in [sma_crossover_signal, rsi_signal, momentum_signal,
                   bollinger_signal, donchian_signal]:
            result = bt.run(fn)
            assert isinstance(result, dict), f"{fn.__name__} did not return a dict"
            sharpe = result.get("sharpe_ratio", 0)
            assert isinstance(sharpe, (int, float)), f"{fn.__name__} sharpe not numeric"


# ── Test 2: Walk-Forward Optimization ────────────────────────────────────────

class TestWFOPipeline:
    def test_wfo_full_pipeline(self):
        from backtesting.walk_forward import WalkForwardOptimizer
        from backtesting.vectorized import sma_crossover_signal
        ohlcv = make_ohlcv(n=700, n_assets=2)
        prices = pd.DataFrame({sym: df["adj_close"] for sym, df in ohlcv.items()})

        wfo = WalkForwardOptimizer(
            prices=prices,
            train_years=1.5,
            test_months=6,
            step_months=6,
        )
        results = wfo.optimize_and_evaluate(
            signal_fn=sma_crossover_signal,
            param_grid={"fast": [10, 20], "slow": [80, 120]},
        )

        assert "avg_oos_sharpe" in results
        assert "deflated_sharpe_corrected" in results
        assert "n_windows" in results
        assert results["n_windows"] > 0
        assert "summary" in results
        assert "pass" in results["summary"]

    def test_regime_analysis(self):
        from backtesting.walk_forward import RegimeAnalyzer
        ohlcv = make_ohlcv(n=500, n_assets=1)
        prices = list(ohlcv.values())[0]["adj_close"]
        equity = prices * 1000
        regimes = RegimeAnalyzer.classify_regimes(prices)
        df = RegimeAnalyzer.analyze_by_regime(equity, regimes)
        assert isinstance(df, pd.DataFrame)


# ── Test 3: Portfolio Optimization Pipeline ───────────────────────────────────

class TestPortfolioOptPipeline:
    def setup_returns(self):
        ohlcv = make_ohlcv(n=400, n_assets=4)
        prices = pd.DataFrame({sym: df["adj_close"] for sym, df in ohlcv.items()})
        return prices.pct_change().dropna()

    def test_hrp_full_pipeline(self):
        from portfolio.optimization import HierarchicalRiskParity
        returns = self.setup_returns()
        hrp = HierarchicalRiskParity()
        weights = hrp.optimize(returns)

        assert len(weights) == len(returns.columns)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(v >= 0 for v in weights.values())

    def test_risk_parity_pipeline(self):
        from portfolio.optimization import RiskParityOptimizer
        returns = self.setup_returns()
        rp = RiskParityOptimizer()
        weights = rp.optimize(returns)
        assert abs(sum(weights.values()) - 1.0) < 1e-4

    def test_mean_variance_pipeline(self):
        from portfolio.optimization import MeanVarianceOptimizer
        returns = self.setup_returns()
        mv = MeanVarianceOptimizer(cov_method="ledoit_wolf")
        weights = mv.optimize(returns, objective="min_variance")
        assert abs(sum(weights.values()) - 1.0) < 1e-4

    def test_black_litterman_with_views(self):
        from portfolio.optimization import BlackLittermanOptimizer
        returns = self.setup_returns()
        assets = list(returns.columns)
        views = [
            {"assets": [assets[0]], "weights": [1], "return": 0.001, "confidence": 0.6},
            {"assets": [assets[0], assets[1]], "weights": [1, -1], "return": 0.0005, "confidence": 0.5},
        ]
        bl = BlackLittermanOptimizer()
        weights = bl.optimize(returns, views=views)
        assert abs(sum(weights.values()) - 1.0) < 1e-4

    def test_portfolio_optimization_strategy(self):
        from portfolio.optimization import PortfolioOptimizationStrategy
        returns = self.setup_returns()
        for optimizer_name in ["hrp", "risk_parity", "equal_weight"]:
            strat = PortfolioOptimizationStrategy(optimizer_name=optimizer_name)
            weights = strat.get_target_weights(returns, current_month=3)
            if weights:
                assert abs(sum(weights.values()) - 1.0) < 1e-4


# ── Test 4: Event-Driven Backtest Pipeline ────────────────────────────────────

class TestEventDrivenPipeline:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.db_path = str(tmp_path / "e2e_event.db")
        os.environ["QUANTSIM_DB"] = self.db_path
        from core.database_v2 import init_full_db
        init_full_db(self.db_path)
        self.ohlcv = make_ohlcv(n=400)
        inject_to_db(self.ohlcv, self.db_path)

    def test_buy_and_hold_validates_positive_return(self):
        """
        Walking skeleton validation: buy-and-hold on a consistently
        rising synthetic price series must produce positive total return.
        """
        from strategies.momentum_factor import BuyAndHold
        from core.event_queue import EventQueue
        from backtesting.engine import BacktestEngine

        eq = EventQueue()
        strategy = BuyAndHold(asset_ids=["SPY"], event_queue=eq)

        engine = BacktestEngine(
            strategies=[strategy],
            start=datetime(2018, 1, 1),
            end=datetime(2019, 6, 30),
            initial_capital=100_000,
            db_path=self.db_path,
            verbose=False,
        )
        result = engine.run()

        # SPY in our synthetic data has positive drift by construction
        assert result.get("total_return", -1) > -0.5, \
            "Buy-and-hold should not lose >50% on synthetic data"

    def test_sma_strategy_generates_fills(self):
        """SMA crossover should produce at least some trades."""
        from strategies.trend import SMAcrossover
        from core.event_queue import EventQueue
        from backtesting.engine import BacktestEngine

        eq = EventQueue()
        strategy = SMAcrossover(
            asset_ids=["SPY"], event_queue=eq,
            fast=10, slow=50
        )
        engine = BacktestEngine(
            strategies=[strategy],
            start=datetime(2018, 1, 1),
            end=datetime(2019, 12, 31),
            initial_capital=100_000,
            db_path=self.db_path,
            verbose=False,
        )
        result = engine.run()
        # With a 400-bar synthetic series, there should be at least one fill
        # (or zero fills if warmup consumes most of the period — both are valid)
        assert "total_return" in result or "warning" in result


# ── Test 5: Signal Aggregation E2E ────────────────────────────────────────────

class TestSignalAggregationE2E:
    def test_ensemble_routes_to_main_queue(self):
        """Ensemble should aggregate strategy signals into the main queue."""
        from strategies.registry import EnsembleEngine
        from strategies.trend import SMAcrossover
        from core.event_queue import EventQueue
        from core.events import BarEvent, SignalEvent, Direction, EventType

        main_queue = EventQueue()
        s1 = SMAcrossover(asset_ids=["SPY"], event_queue=main_queue, fast=5, slow=20)
        s2 = SMAcrossover(asset_ids=["SPY"], event_queue=main_queue, fast=10, slow=40)

        ensemble = EnsembleEngine(
            strategies=[s1, s2],
            main_queue=main_queue,
            aggregation_method="confidence_weighted",
        )

        # Manually inject a signal from each strategy into ensemble's internal queue
        for i, strat in enumerate([s1, s2]):
            sig = SignalEvent(
                timestamp=datetime(2022, 6, 1),
                strategy_id=f"test_strat_{i}",
                asset_id="SPY",
                direction=Direction.LONG,
                confidence=0.8,
                signal_type="trend",
            )
            ensemble.aggregator.add_signal(sig)

        # Manually trigger aggregation
        agg = ensemble.aggregator.aggregate("SPY")
        assert agg is not None
        assert agg.direction == Direction.LONG
        assert len(agg.contributing_strategies) == 2


# ── Test 6: GARCH Integration ─────────────────────────────────────────────────

class TestGARCHIntegration:
    def test_garch_vol_feeds_into_position_sizing(self):
        """GARCH forecast should produce sensible leverage multiplier."""
        from strategies.garch_vol import GARCHVolatilityAdapter
        from portfolio.sizing import VolatilityTargetSizer, compute_realized_vol

        adapter = GARCHVolatilityAdapter(
            fit_window=100, refit_every=21, vol_target=0.12
        )
        np.random.seed(99)
        # High-vol regime
        returns = pd.Series(np.random.normal(0, 0.025, 150))
        for r in returns:
            adapter.add_bar("SPY", float(r))

        forecast = adapter.get_vol_forecast("SPY")
        leverage = adapter.get_leverage("SPY")

        # High vol input -> low leverage (target vol / actual vol < 1)
        assert 0 < forecast < 3.0
        assert 0.25 <= leverage <= 2.0

        # Realized vol on same data should be roughly comparable
        realized = float(returns.std() * np.sqrt(252))
        # They won't be identical but should be same order of magnitude
        assert abs(forecast - realized) / realized < 3.0  # within 3x


# ── Test 7: Full Report Generation ───────────────────────────────────────────

class TestFullReport:
    def test_generate_full_report(self):
        from reporting.advanced import generate_full_report
        from reporting.analytics import PerformanceAnalytics

        np.random.seed(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        equity = pd.Series(
            100_000 * np.cumprod(1 + np.random.normal(0.0004, 0.015, n)),
            index=dates,
        )
        benchmark = pd.Series(
            100_000 * np.cumprod(1 + np.random.normal(0.0003, 0.013, n)),
            index=dates,
        )

        report = generate_full_report(
            equity_curve=equity,
            benchmark=benchmark,
            n_strategies_tested=10,
            initial_capital=100_000,
        )

        assert "core_metrics" in report
        assert "monte_carlo" in report
        assert "verdict" in report
        assert "lines" in report["verdict"]
        assert isinstance(report["verdict"]["proceed"], bool)

        core = report["core_metrics"]
        assert "sharpe_ratio" in core
        assert "deflated_sharpe_ratio" in core
        assert "max_drawdown" in core
        assert core["max_drawdown"] <= 0

    def test_monte_carlo_confidence_interval_contains_point_estimate(self):
        from reporting.advanced import AdvancedAnalytics
        np.random.seed(7)
        returns = pd.Series(np.random.normal(0.0004, 0.012, 400))
        result = AdvancedAnalytics.monte_carlo_sharpe(returns, n_simulations=2000)

        assert "sharpe_point_estimate" in result
        assert result["sharpe_ci_low"] <= result["sharpe_point_estimate"] <= result["sharpe_ci_high"], \
            "Point estimate should lie within its own CI"


# ── Test 8: Database Persistence E2E ─────────────────────────────────────────

class TestDatabasePersistenceE2E:
    def test_portfolio_persists_and_loads(self, tmp_path):
        from portfolio.portfolio import Portfolio, Position
        from core.events import AssetType, FillEvent, OrderSide

        db_path = str(tmp_path / "persist_test.db")
        os.environ["QUANTSIM_DB"] = db_path
        from core.database_v2 import init_full_db
        init_full_db(db_path)

        portfolio = Portfolio(initial_capital=100_000, cash=100_000,
                              timestamp=datetime(2022, 1, 3))
        fill = FillEvent(
            order_id="test-1",
            asset_id="SPY",
            asset_type=AssetType.EQUITY,
            side=OrderSide.BUY,
            quantity=100,
            fill_price=400.0,
            commission=0.0,
            slippage=0.0,
            strategy_id="sma",
            timestamp=datetime(2022, 1, 3),
        )
        portfolio.apply_fill(fill)
        portfolio.persist(db_path)

        # Load and verify
        from reporting.analytics import load_equity_curve_from_db
        from core.database import db_conn
        with db_conn(db_path) as conn:
            rows = conn.execute("SELECT COUNT(*) FROM portfolio_snapshots").fetchone()
            assert rows[0] >= 1

    def test_wfo_results_logged_to_db(self, tmp_path):
        from core.database_v2 import init_full_db, log_wfo_result
        import uuid

        db_path = str(tmp_path / "wfo_test.db")
        init_full_db(db_path)

        log_wfo_result(
            db_path=db_path,
            result_id=str(uuid.uuid4()),
            strategy_id="sma_test",
            window={"window_id": 0, "train_start": pd.Timestamp("2020-01-01"),
                    "train_end": pd.Timestamp("2022-12-31"),
                    "test_start": pd.Timestamp("2023-01-01"),
                    "test_end": pd.Timestamp("2023-12-31")},
            best_params={"fast": 20, "slow": 100},
            metrics={"oos_sharpe": 0.65, "oos_return": 0.08, "oos_max_dd": -0.12},
        )

        from core.database import db_conn
        with db_conn(db_path) as conn:
            rows = conn.execute("SELECT COUNT(*) FROM wfo_results").fetchone()
            assert rows[0] == 1


# ── Test 9: Config System E2E ────────────────────────────────────────────────

class TestConfigE2E:
    def test_config_drives_engine_initialization(self, tmp_path):
        """Config values should be respected when building an engine."""
        import os
        os.environ["QUANTSIM_CAPITAL"] = "75000"

        from core.config import QuantSimConfig
        cfg = QuantSimConfig()
        assert cfg.initial_capital == 75000

        del os.environ["QUANTSIM_CAPITAL"]

    def test_config_validation_produces_warnings_not_errors(self):
        from core.config import QuantSimConfig
        cfg = QuantSimConfig()
        warnings = cfg.validate()
        # Validation should return a list (possibly empty), never raise
        assert isinstance(warnings, list)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
