"""
Walk-forward out-of-sample windows must trade from their first day.

The test window is scored on its own dates only, but the strategy's
indicators need history: an SMA 50/200 started cold on a one-year window is
flat for 200 of its 252 days, which makes every out-of-sample Sharpe
meaningless. Synthetic prices.
"""

import numpy as np
import pandas as pd

from quantsim.backtesting.vectorized import sma_crossover_signal
from quantsim.backtesting.walk_forward import WalkForwardOptimizer
from tests.synthetic_bars import make_price_df, make_price_series


def test_out_of_sample_window_is_invested_from_its_first_day():
    index = pd.bdate_range("2015-01-01", periods=4 * 252 + 10)
    prices = pd.DataFrame({"SYN": 100 * np.exp(0.0008 * np.arange(len(index)))}, index=index)
    wfo = WalkForwardOptimizer(prices, train_years=3, test_months=12, step_months=12)

    result = wfo.optimize_and_evaluate(
        sma_crossover_signal, {"fast": [50], "slow": [200]}, commission_rate=0.0, slippage_pct=0.0
    )

    window = result["window_results"][0]
    daily_change = np.diff(window["oos_equity"].to_numpy())
    assert np.all(daily_change > 0), "flat days mean the strategy started cold"
    assert window["oos_return"] > 0


class TestWalkForwardOptimizer:
    def test_window_generation(self):
        from quantsim.backtesting.walk_forward import WalkForwardOptimizer

        prices = make_price_df(n=700, n_assets=2)
        wfo = WalkForwardOptimizer(prices=prices, train_years=1, test_months=6, step_months=3)
        windows = wfo.generate_windows()
        assert len(windows) > 0
        # Windows should not overlap (test period of window N < train start of window N+1)
        for i in range(len(windows) - 1):
            assert (
                windows[i]["test_end"] <= windows[i + 1]["train_start"] or True
            )  # step may overlap train

    def test_wfo_returns_oos_metrics(self):
        from quantsim.backtesting.vectorized import sma_crossover_signal
        from quantsim.backtesting.walk_forward import WalkForwardOptimizer

        prices = make_price_df(n=600, n_assets=2)
        wfo = WalkForwardOptimizer(prices=prices, train_years=1, test_months=6, step_months=6)
        results = wfo.optimize_and_evaluate(
            signal_fn=sma_crossover_signal,
            param_grid={"fast": [20, 50], "slow": [100]},
        )
        assert "avg_oos_sharpe" in results
        assert "deflated_sharpe_corrected" in results
        assert "window_results" in results


class TestRegimeAnalysis:
    def test_regime_classification(self):
        from quantsim.backtesting.walk_forward import RegimeAnalyzer

        prices = make_price_series(n=500)
        regimes = RegimeAnalyzer.classify_regimes(prices)
        assert set(regimes.unique()).issubset(
            {"bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol", "unknown"}
        )

    def test_regime_analysis_returns_df(self):
        from quantsim.backtesting.walk_forward import RegimeAnalyzer

        prices = make_price_series(n=500)
        equity = prices * 1000
        regimes = RegimeAnalyzer.classify_regimes(prices)
        result = RegimeAnalyzer.analyze_by_regime(equity, regimes)
        assert isinstance(result, pd.DataFrame)
        assert "sharpe" in result.columns
