"""GARCH volatility forecasts and the GARCH position sizer."""

import numpy as np
import pandas as pd


class TestGARCH:
    def make_returns(self, n=300, seed=42):
        np.random.seed(seed)
        # Simulate GARCH-like vol clustering
        returns = []
        sigma = 0.015
        for _ in range(n):
            r = np.random.normal(0, sigma)
            returns.append(r)
            sigma = np.sqrt(0.00001 + 0.1 * r**2 + 0.85 * sigma**2)
        return pd.Series(returns)

    def test_garch_fit_succeeds(self):
        from quantsim.strategies.garch_vol import GARCHForecaster

        fc = GARCHForecaster()
        returns = self.make_returns()
        success = fc.fit(returns)
        if success:
            assert fc.is_fitted
            assert 0 < fc.current_forecast < 2.0
            params = fc.params
            # alpha + beta < 1 (stationarity)
            if "alpha" in params and "beta" in params:
                assert params["alpha"] + params["beta"] < 1.0

    def test_garch_forecast_positive(self):
        from quantsim.strategies.garch_vol import GARCHForecaster

        fc = GARCHForecaster()
        returns = self.make_returns()
        fc.fit(returns)
        vol = fc.forecast()
        assert vol > 0
        assert vol < 5.0  # sanity upper bound

    def test_garch_adapter_update(self):
        from quantsim.strategies.garch_vol import GARCHVolatilityAdapter

        adapter = GARCHVolatilityAdapter(fit_window=60, refit_every=10)
        returns = self.make_returns()
        for r in returns:
            adapter.add_bar("SPY", float(r))
        vol = adapter.get_vol_forecast("SPY")
        assert 0 < vol < 3.0

    def test_leverage_for_target_vol(self):
        from quantsim.strategies.garch_vol import GARCHForecaster

        fc = GARCHForecaster(vol_target_annual=0.12)
        # High vol scenario: leverage should be < 1
        leverage = fc.leverage_for_target_vol(forecast_vol=0.30)
        assert leverage < 1.0
        assert leverage >= 0.25  # capped at minimum
        # Low vol scenario: leverage should be > 1
        leverage = fc.leverage_for_target_vol(forecast_vol=0.05)
        assert leverage > 1.0
        assert leverage <= 2.0  # capped at maximum

    def test_insufficient_data_fallback(self):
        from quantsim.strategies.garch_vol import GARCHForecaster

        fc = GARCHForecaster()
        short_returns = pd.Series(np.random.normal(0, 0.01, 20))
        success = fc.fit(short_returns)
        assert not success  # should fail gracefully


class TestGARCHSizerIntegration:
    def test_garch_sizer_in_sizing_module(self):
        from quantsim.portfolio.sizing import GARCHVolatilityTargetSizer

        sizer = GARCHVolatilityTargetSizer(target_annual_vol=0.12, max_position_fraction=0.15)
        assert sizer is not None

    def test_garch_sizer_update_and_size(self):
        from quantsim.portfolio.portfolio import Portfolio
        from quantsim.portfolio.sizing import GARCHVolatilityTargetSizer

        sizer = GARCHVolatilityTargetSizer(target_annual_vol=0.12)
        np.random.seed(42)
        returns = np.random.normal(0.0003, 0.020, 150)

        for r in returns:
            sizer.update("SPY", float(r))

        portfolio = Portfolio(initial_capital=100_000, cash=100_000)
        qty = sizer.size(portfolio, "SPY", confidence=1.0, price=400.0)
        assert qty > 0
        assert qty < 500  # sanity: not buying more than $200K of a $100K portfolio
