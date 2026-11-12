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
