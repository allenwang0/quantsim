"""Portfolio optimizers and the optimization overlay manager."""

from datetime import datetime

import numpy as np
import pandas as pd


class TestPortfolioOptimization:
    def make_returns(self, n=252, n_assets=5):
        np.random.seed(123)
        returns = pd.DataFrame(
            np.random.normal(0.0003, 0.015, (n, n_assets)),
            columns=[f"A{i}" for i in range(n_assets)],
        )
        return returns

    def test_hrp_weights_sum_to_one(self):
        from quantsim.portfolio.optimization import HierarchicalRiskParity

        returns = self.make_returns()
        hrp = HierarchicalRiskParity()
        weights = hrp.optimize(returns)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_hrp_all_positive(self):
        from quantsim.portfolio.optimization import HierarchicalRiskParity

        returns = self.make_returns()
        hrp = HierarchicalRiskParity()
        weights = hrp.optimize(returns)
        assert all(v >= 0 for v in weights.values())

    def test_risk_parity_weights_sum_to_one(self):
        from quantsim.portfolio.optimization import RiskParityOptimizer

        returns = self.make_returns(n_assets=4)
        rp = RiskParityOptimizer()
        weights = rp.optimize(returns)
        assert abs(sum(weights.values()) - 1.0) < 1e-4

    def test_mean_variance_min_variance(self):
        from quantsim.portfolio.optimization import MeanVarianceOptimizer

        returns = self.make_returns()
        mv = MeanVarianceOptimizer()
        weights = mv.optimize(returns, objective="min_variance")
        assert abs(sum(weights.values()) - 1.0) < 1e-4
        assert all(v >= 0 for v in weights.values())

    def test_black_litterman_no_views(self):
        from quantsim.portfolio.optimization import BlackLittermanOptimizer

        returns = self.make_returns()
        bl = BlackLittermanOptimizer()
        weights = bl.optimize(returns, views=None)
        assert abs(sum(weights.values()) - 1.0) < 1e-4

    def test_black_litterman_with_views(self):
        from quantsim.portfolio.optimization import BlackLittermanOptimizer

        returns = self.make_returns()
        assets = list(returns.columns)
        views = [{"assets": [assets[0]], "weights": [1], "return": 0.001, "confidence": 0.7}]
        bl = BlackLittermanOptimizer()
        weights = bl.optimize(returns, views=views)
        assert abs(sum(weights.values()) - 1.0) < 1e-4

    def test_covariance_shrinkage(self):
        from quantsim.portfolio.optimization import compute_covariance

        returns = self.make_returns(n=100, n_assets=20)  # small T, large N
        cov_sample = compute_covariance(returns, "sample")
        cov_lw = compute_covariance(returns, "ledoit_wolf")
        # Ledoit-Wolf should have smaller condition number
        cond_sample = np.linalg.cond(cov_sample)
        cond_lw = np.linalg.cond(cov_lw)
        assert cond_lw <= cond_sample * 1.1  # LW should be at most slightly worse

    def test_efficient_frontier(self):
        from quantsim.portfolio.optimization import MeanVarianceOptimizer

        returns = self.make_returns(n_assets=4)
        mv = MeanVarianceOptimizer()
        frontier = mv.efficient_frontier(returns, n_points=10)
        assert len(frontier) > 0
        assert "return" in frontier.columns
        assert "volatility" in frontier.columns


class TestOptimizationOverlayManager:
    def test_overlay_manager_initializes(self):
        from quantsim.core.event_queue import EventQueue
        from quantsim.portfolio.manager import OptimizationOverlayManager
        from quantsim.portfolio.optimization import HierarchicalRiskParity
        from quantsim.portfolio.portfolio import Portfolio
        from quantsim.risk.risk_manager import RiskManager

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
        from quantsim.core.event_queue import EventQueue
        from quantsim.core.events import BarEvent
        from quantsim.portfolio.manager import OptimizationOverlayManager
        from quantsim.portfolio.optimization import HierarchicalRiskParity
        from quantsim.portfolio.portfolio import Portfolio
        from quantsim.risk.risk_manager import RiskManager

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

        # Simulate bars from two different months
        # The rebalance cache needs enough returns first
        for month in [1, 2]:
            for day in range(1, 22):
                bar = BarEvent(
                    timestamp=datetime(2022, month, day),
                    asset_id="SPY",
                    open=400,
                    high=402,
                    low=398,
                    close=400,
                    volume=5_000_000,
                    adj_close=400.0,
                )
                mgr.on_bar(bar)  # should not raise
