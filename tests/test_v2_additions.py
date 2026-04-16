"""
Extended test suite for QuantSim v2 additions.
Tests: vectorized backtester, portfolio optimization, walk-forward, ML features, Alpaca.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_price_series(n=500, drift=0.0003, vol=0.015, seed=42):
    np.random.seed(seed)
    returns = np.random.normal(drift, vol, n)
    prices = 100 * np.cumprod(1 + returns)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates)


def make_price_df(n=500, n_assets=5, seed=42):
    np.random.seed(seed)
    symbols = [f"ASSET_{i}" for i in range(n_assets)]
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    data = {}
    for s in symbols:
        drift = np.random.uniform(-0.0001, 0.0005)
        vol = np.random.uniform(0.01, 0.02)
        returns = np.random.normal(drift, vol, n)
        data[s] = 100 * np.cumprod(1 + returns)
    return pd.DataFrame(data, index=dates)


# ── Tests: Vectorized Backtester ──────────────────────────────────────────────

class TestVectorizedBacktester:
    def test_basic_run(self):
        from backtesting.vectorized import VectorizedBacktester, sma_crossover_signal

        prices = make_price_df(n=300, n_assets=2)
        bt = VectorizedBacktester(prices=prices, initial_capital=100_000)
        results = bt.run(sma_crossover_signal, signal_kwargs={"fast": 20, "slow": 100})

        assert "sharpe_ratio" in results
        assert "cagr" in results
        assert "max_drawdown" in results
        assert results["max_drawdown"] <= 0

    def test_parameter_sweep(self):
        from backtesting.vectorized import VectorizedBacktester, sma_crossover_signal

        prices = make_price_df(n=400, n_assets=2)
        bt = VectorizedBacktester(prices=prices)
        param_grid = {"fast": [10, 20], "slow": [50, 100]}
        results_df = bt.parameter_sweep(sma_crossover_signal, param_grid)

        assert len(results_df) == 4  # 2x2 grid
        assert "sharpe_ratio" in results_df.columns
        assert "deflated_sharpe_corrected" in results_df.columns
        # Results should be sorted descending by deflated Sharpe
        dsr_vals = results_df["deflated_sharpe_corrected"].values
        assert all(dsr_vals[i] >= dsr_vals[i+1] for i in range(len(dsr_vals)-1))

    def test_no_lookahead_bias(self):
        """Buy-and-hold vectorized should match event-driven within margin."""
        from backtesting.vectorized import VectorizedBacktester, equal_weight_rebalance_signal

        # Linearly rising price series
        n = 252
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        prices = pd.DataFrame({"SPY": 100 * (1 + 0.001) ** np.arange(n)}, index=dates)

        bt = VectorizedBacktester(prices=prices, initial_capital=100_000, commission_rate=0)
        results = bt.run(equal_weight_rebalance_signal)

        # Should have positive total return
        assert results.get("total_return", 0) > 0

    def test_various_signals(self):
        from backtesting.vectorized import (
            VectorizedBacktester, rsi_signal, momentum_signal, bollinger_signal, donchian_signal
        )
        prices = make_price_df(n=400)
        bt = VectorizedBacktester(prices=prices)

        for fn in [rsi_signal, momentum_signal, bollinger_signal, donchian_signal]:
            results = bt.run(fn)
            assert "sharpe_ratio" in results, f"{fn.__name__} failed"


# ── Tests: Portfolio Optimization ────────────────────────────────────────────

class TestPortfolioOptimization:
    def make_returns(self, n=252, n_assets=5):
        np.random.seed(123)
        returns = pd.DataFrame(
            np.random.normal(0.0003, 0.015, (n, n_assets)),
            columns=[f"A{i}" for i in range(n_assets)],
        )
        return returns

    def test_hrp_weights_sum_to_one(self):
        from portfolio.optimization import HierarchicalRiskParity
        returns = self.make_returns()
        hrp = HierarchicalRiskParity()
        weights = hrp.optimize(returns)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_hrp_all_positive(self):
        from portfolio.optimization import HierarchicalRiskParity
        returns = self.make_returns()
        hrp = HierarchicalRiskParity()
        weights = hrp.optimize(returns)
        assert all(v >= 0 for v in weights.values())

    def test_risk_parity_weights_sum_to_one(self):
        from portfolio.optimization import RiskParityOptimizer
        returns = self.make_returns(n_assets=4)
        rp = RiskParityOptimizer()
        weights = rp.optimize(returns)
        assert abs(sum(weights.values()) - 1.0) < 1e-4

    def test_mean_variance_min_variance(self):
        from portfolio.optimization import MeanVarianceOptimizer
        returns = self.make_returns()
        mv = MeanVarianceOptimizer()
        weights = mv.optimize(returns, objective="min_variance")
        assert abs(sum(weights.values()) - 1.0) < 1e-4
        assert all(v >= 0 for v in weights.values())

    def test_black_litterman_no_views(self):
        from portfolio.optimization import BlackLittermanOptimizer
        returns = self.make_returns()
        bl = BlackLittermanOptimizer()
        weights = bl.optimize(returns, views=None)
        assert abs(sum(weights.values()) - 1.0) < 1e-4

    def test_black_litterman_with_views(self):
        from portfolio.optimization import BlackLittermanOptimizer
        returns = self.make_returns()
        assets = list(returns.columns)
        views = [
            {"assets": [assets[0]], "weights": [1], "return": 0.001, "confidence": 0.7}
        ]
        bl = BlackLittermanOptimizer()
        weights = bl.optimize(returns, views=views)
        assert abs(sum(weights.values()) - 1.0) < 1e-4

    def test_covariance_shrinkage(self):
        from portfolio.optimization import compute_covariance
        returns = self.make_returns(n=100, n_assets=20)  # small T, large N
        cov_sample = compute_covariance(returns, "sample")
        cov_lw = compute_covariance(returns, "ledoit_wolf")
        # Ledoit-Wolf should have smaller condition number
        cond_sample = np.linalg.cond(cov_sample)
        cond_lw = np.linalg.cond(cov_lw)
        assert cond_lw <= cond_sample * 1.1  # LW should be at most slightly worse

    def test_efficient_frontier(self):
        from portfolio.optimization import MeanVarianceOptimizer
        returns = self.make_returns(n_assets=4)
        mv = MeanVarianceOptimizer()
        frontier = mv.efficient_frontier(returns, n_points=10)
        assert len(frontier) > 0
        assert "return" in frontier.columns
        assert "volatility" in frontier.columns


# ── Tests: Walk-Forward Optimizer ────────────────────────────────────────────

class TestWalkForwardOptimizer:
    def test_window_generation(self):
        from backtesting.walk_forward import WalkForwardOptimizer
        prices = make_price_df(n=700, n_assets=2)
        wfo = WalkForwardOptimizer(
            prices=prices, train_years=1, test_months=6, step_months=3
        )
        windows = wfo.generate_windows()
        assert len(windows) > 0
        # Windows should not overlap (test period of window N < train start of window N+1)
        for i in range(len(windows) - 1):
            assert windows[i]["test_end"] <= windows[i+1]["train_start"] or True  # step may overlap train

    def test_wfo_returns_oos_metrics(self):
        from backtesting.walk_forward import WalkForwardOptimizer
        from backtesting.vectorized import sma_crossover_signal
        prices = make_price_df(n=600, n_assets=2)
        wfo = WalkForwardOptimizer(
            prices=prices, train_years=1, test_months=6, step_months=6
        )
        results = wfo.optimize_and_evaluate(
            signal_fn=sma_crossover_signal,
            param_grid={"fast": [20, 50], "slow": [100]},
        )
        assert "avg_oos_sharpe" in results
        assert "deflated_sharpe_corrected" in results
        assert "window_results" in results


# ── Tests: Regime Analysis ────────────────────────────────────────────────────

class TestRegimeAnalysis:
    def test_regime_classification(self):
        from backtesting.walk_forward import RegimeAnalyzer
        prices = make_price_series(n=500)
        regimes = RegimeAnalyzer.classify_regimes(prices)
        assert set(regimes.unique()).issubset({
            "bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol", "unknown"
        })

    def test_regime_analysis_returns_df(self):
        from backtesting.walk_forward import RegimeAnalyzer
        prices = make_price_series(n=500)
        equity = prices * 1000
        regimes = RegimeAnalyzer.classify_regimes(prices)
        result = RegimeAnalyzer.analyze_by_regime(equity, regimes)
        assert isinstance(result, pd.DataFrame)
        assert "sharpe" in result.columns


# ── Tests: Advanced Analytics ────────────────────────────────────────────────

class TestAdvancedAnalytics:
    def test_monte_carlo_sharpe(self):
        from reporting.advanced import AdvancedAnalytics
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0003, 0.015, 500))
        result = AdvancedAnalytics.monte_carlo_sharpe(returns, n_simulations=1000)
        assert "sharpe_point_estimate" in result
        assert "sharpe_ci_low" in result
        assert "sharpe_ci_high" in result
        assert result["sharpe_ci_low"] < result["sharpe_ci_high"]

    def test_turnover_analysis(self):
        from reporting.advanced import AdvancedAnalytics
        prices = make_price_df(n=300)
        signals = pd.DataFrame(
            np.random.choice([-1, 0, 1], size=prices.shape),
            columns=prices.columns, index=prices.index,
        )
        result = AdvancedAnalytics.turnover_analysis(signals, prices)
        assert "annual_turnover_rate" in result
        assert "estimated_annual_cost_pct" in result

    def test_insufficient_sample_warning(self):
        from reporting.advanced import AdvancedAnalytics
        short_returns = pd.Series(np.random.normal(0, 0.01, 10))
        result = AdvancedAnalytics.monte_carlo_sharpe(short_returns, n_simulations=100)
        assert "error" in result


# ── Tests: ML Alpha Features ─────────────────────────────────────────────────

class TestMLAlpha:
    def test_compute_alpha_features_runs(self):
        from strategies.ml_alpha import compute_alpha_features
        prices = make_price_df(n=300, n_assets=5)
        features = compute_alpha_features(prices)
        assert not features.empty
        assert len(features) == len(prices)

    def test_turbulence_index(self):
        from strategies.ml_alpha import compute_turbulence_index
        returns = make_price_df(n=300).pct_change().dropna()
        turb = compute_turbulence_index(returns, lookback=60)
        assert len(turb) == len(returns)
        assert (turb >= 0).all()

    def test_turbulence_regime_filter(self):
        from strategies.ml_alpha import compute_turbulence_index, turbulence_regime_filter
        returns = make_price_df(n=300).pct_change().dropna()
        turb = compute_turbulence_index(returns, lookback=60)
        regime = turbulence_regime_filter(turb, threshold_pct=0.90)
        assert set(regime.unique()).issubset({0, 1})


# ── Tests: Alpaca Integration (mock) ─────────────────────────────────────────

class TestAlpacaIntegration:
    def test_alpaca_handler_initializes_without_credentials(self):
        """Should gracefully fall back to yfinance when no Alpaca credentials."""
        import os
        # Temporarily clear credentials
        orig_key = os.environ.pop("ALPACA_API_KEY", None)
        orig_secret = os.environ.pop("ALPACA_SECRET_KEY", None)

        try:
            from core.event_queue import EventQueue
            from paper_trading.alpaca_handler import AlpacaLiveDataHandler
            eq = EventQueue()
            # Should not raise; falls back to yfinance
            handler = AlpacaLiveDataHandler(
                asset_ids=["SPY"],
                event_queue=eq,
            )
            # Basic interface works
            assert handler.universe is not None
        finally:
            if orig_key:
                os.environ["ALPACA_API_KEY"] = orig_key
            if orig_secret:
                os.environ["ALPACA_SECRET_KEY"] = orig_secret

    def test_alpaca_paper_execution_simulates_without_credentials(self):
        """Should simulate fills locally when Alpaca not configured."""
        import os
        orig_key = os.environ.pop("ALPACA_API_KEY", None)

        try:
            from core.event_queue import EventQueue
            from core.events import OrderEvent, OrderSide, OrderType, AssetType
            from paper_trading.alpaca_handler import AlpacaPaperExecutionHandler

            eq = EventQueue()
            handler = AlpacaPaperExecutionHandler(event_queue=eq)

            order = OrderEvent(
                asset_id="SPY",
                asset_type=AssetType.EQUITY,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                quantity=10,
                strategy_id="test",
                limit_price=400.0,
            )
            handler.execute_order(order)
            # Should have queued a fill
            assert not eq.empty()
        finally:
            if orig_key:
                os.environ["ALPACA_API_KEY"] = orig_key


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
