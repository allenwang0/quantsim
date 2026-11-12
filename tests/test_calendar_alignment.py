"""
Multi-asset bars must line up on real exchange sessions, and a price that
was forward-filled because an asset did not trade must never trigger a
trade at that stale price. Synthetic prices.
"""

import numpy as np
import pandas as pd

from quantsim.backtesting.vectorized import VectorizedBacktester, _simulate_portfolio_numba


def signal_from_frame(frame):
    def signal_fn(prices, returns):
        return frame.reindex(prices.index).fillna(0)

    return signal_fn


def test_rows_follow_the_exchange_calendar():
    index = pd.date_range("2020-06-29", "2020-07-08", freq="B")  # includes July 3 holiday
    prices = pd.DataFrame({"A": np.linspace(100, 110, len(index))}, index=index)

    backtester = VectorizedBacktester(prices, calendar="NYSE")

    assert pd.Timestamp("2020-07-03") not in backtester.prices.index
    assert len(backtester.prices) == len(index) - 1


def test_forward_filled_bar_does_not_trade():
    index = pd.date_range("2021-03-01", periods=10, freq="B")
    prices = pd.DataFrame(
        {"A": np.linspace(100, 109, 10), "B": np.linspace(50, 59, 10)}, index=index
    )
    prices.loc[index[4], "B"] = np.nan  # B did not trade on bar 4
    signals = pd.DataFrame(0.0, index=index, columns=["A", "B"])
    signals.iloc[3:] = 1.0  # wanted from bar 3's close, executes on bar 4

    backtester = VectorizedBacktester(prices, commission_rate=0.0, slippage_pct=0.0)
    _, positions, _ = _simulate_portfolio_numba(
        backtester.prices.to_numpy(),
        signals.to_numpy(),
        100_000.0,
        0.0,
        0.0,
        0.1,
        np.inf,
        backtester.stale,
    )

    assert positions[4, 0] > 0  # A traded on its real bar 4
    assert positions[4, 1] == 0  # B's bar 4 is a forward-filled copy of bar 3
    assert positions[5, 1] > 0  # B enters on its next real bar
    np.testing.assert_allclose(
        positions[5, 1], 100_000 * 0.1 / prices.loc[index[5], "B"], rtol=0.01
    )

    result = backtester.run(signal_from_frame(signals))
    assert result["_meta"]["trade_count"] == 2
