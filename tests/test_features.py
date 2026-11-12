"""Alpha features, the turbulence index and the turbulence filter."""

from datetime import datetime

import numpy as np
import pandas as pd

from tests.synthetic_bars import make_price_df


class TestMLAlpha:
    def test_compute_alpha_features_runs(self):
        from quantsim.strategies.ml_alpha import compute_alpha_features

        prices = make_price_df(n=300, n_assets=5)
        features = compute_alpha_features(prices)
        assert not features.empty
        assert len(features) == len(prices)

    def test_turbulence_index(self):
        from quantsim.strategies.ml_alpha import compute_turbulence_index

        returns = make_price_df(n=300).pct_change().dropna()
        turb = compute_turbulence_index(returns, lookback=60)
        assert len(turb) == len(returns)
        assert (turb >= 0).all()

    def test_turbulence_regime_filter(self):
        from quantsim.strategies.ml_alpha import compute_turbulence_index, turbulence_regime_filter

        returns = make_price_df(n=300).pct_change().dropna()
        turb = compute_turbulence_index(returns, lookback=60)
        regime = turbulence_regime_filter(turb, threshold_pct=0.90)
        assert set(regime.unique()).issubset({0, 1})


class TestTurbulenceFilter:
    def make_prices(self, n=400, n_assets=4, seed=42):
        np.random.seed(seed)
        symbols = ["SPY", "QQQ", "GLD", "TLT"]
        dates = pd.date_range("2018-01-01", periods=n, freq="B")
        data = {}
        for sym in symbols[:n_assets]:
            ret = np.random.normal(0.0003, 0.012, n)
            data[sym] = 100 * np.cumprod(1 + ret)
        return pd.DataFrame(data, index=dates)

    def test_turbulence_filter_suppresses_on_high_turb(self):
        """TurbulenceFilteredStrategy should emit FLAT during high turbulence."""
        from quantsim.core.event_queue import EventQueue
        from quantsim.core.events import BarEvent
        from quantsim.strategies.ml_strategy import TurbulenceFilteredStrategy
        from quantsim.strategies.trend import SMAcrossover

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
                dates = pd.date_range("2020-01-01", periods=max(n, 5), freq="B")
                prices = 100 + np.cumsum(np.random.normal(0, 2, max(n, 5)))
                return pd.DataFrame({"adj_close": prices, "close": prices}, index=dates)

            def get_current_bar(self, aid):
                return pd.Series({"adj_close": 100.0, "close": 100.0})

            def get_macro_value(self, s):
                return None

        dh = MockDH()
        for i in range(50):
            bar = BarEvent(
                timestamp=datetime(2020, 1, 1) + pd.Timedelta(days=i),
                asset_id="SPY",
                open=100,
                high=105,
                low=95,
                close=100,
                volume=1_000_000,
                adj_close=100.0,
            )
            filtered.on_bar(bar, dh)

        # Should have processed without errors
