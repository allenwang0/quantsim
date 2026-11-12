"""
Vectorized simulator contract.

Signal functions return the position wanted at bar t's close using data
through t. The simulator alone applies the one-bar delay and trades only when
the wanted direction changes, so trade counts and cost drag are real.
Expected signals here are recomputed from each indicator's definition, not by
calling the function under test. All prices are seeded synthetic walks.
"""

import numpy as np
import pandas as pd
import pytest

from quantsim.backtesting.vectorized import (
    _simulate_portfolio_numba,
    bollinger_signal,
    donchian_signal,
    equal_weight_rebalance_signal,
    momentum_signal,
    rsi_signal,
    sma_crossover_signal,
)


def random_walk(n_bars, n_assets=1, seed=7, daily_vol=0.01):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, daily_vol, (n_bars, n_assets))
    data = 100 * np.exp(np.cumsum(steps, axis=0))
    index = pd.date_range("2015-01-01", periods=n_bars, freq="B")
    return pd.DataFrame(data, index=index, columns=[f"SYN{i}" for i in range(n_assets)])


def test_held_position_is_not_resized_every_bar():
    """
    Before the fix, target size was recomputed from remaining cash each bar,
    so a constant +1 signal traded 249 times in 250 bars and paid costs each time.
    """
    prices = random_walk(251).to_numpy()
    signals = np.ones_like(prices)
    capital, commission, slippage, size = 100_000.0, 0.001, 0.0005, 0.10

    equity, _, trade_count = _simulate_portfolio_numba(
        prices, signals, capital, commission, slippage, size
    )

    entry, last = prices[1, 0], prices[-1, 0]
    units = capital * size / entry
    expected = capital - capital * size * (1 + commission + slippage) + units * last
    assert trade_count == 1
    assert equity[-1] == pytest.approx(expected, rel=1e-9)


def test_rebalance_threshold_resizes_drifted_position():
    prices = np.array([[100.0], [100.0], [200.0], [200.0]])
    signals = np.ones_like(prices)

    _, positions, trade_count = _simulate_portfolio_numba(
        prices, signals, 100_000.0, 0.0, 0.0, 0.10, 0.5
    )

    assert trade_count == 2
    assert positions[-1, 0] == pytest.approx(110_000 * 0.10 / 200.0)


def rsi(series, n):
    delta = series.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = (-delta).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + avg_gain / avg_loss.replace(0, 1e-10))


def expected_sma(p):
    return (p.rolling(5).mean() > p.rolling(20).mean()).astype(float)


def expected_rsi(p):
    r = p.apply(rsi, args=(14,))
    return ((r < 45) & (p > p.rolling(200).mean())).astype(float)


def expected_momentum(p):
    past_return = p.shift(5) / p.shift(65) - 1
    return (past_return > 0).astype(float)


def expected_bollinger(p):
    mid, sd = p.rolling(20).mean(), p.rolling(20).std()
    return (p < mid - sd).astype(float) - (p > mid + sd).astype(float)


def expected_donchian(p):
    prior_high = p.rolling(20).max().shift(1)
    prior_low = p.rolling(20).min().shift(1)
    return (p > prior_high).astype(float) - (p < prior_low).astype(float)


def expected_equal_weight(p):
    return pd.DataFrame(1.0, index=p.index, columns=p.columns)


CASES = [
    (sma_crossover_signal, {"fast": 5, "slow": 20}, expected_sma),
    (rsi_signal, {"oversold": 45}, expected_rsi),
    (momentum_signal, {"lookback": 60, "skip": 5}, expected_momentum),
    (bollinger_signal, {"window": 20, "k": 1.0}, expected_bollinger),
    (donchian_signal, {"period": 20}, expected_donchian),
    (equal_weight_rebalance_signal, {}, expected_equal_weight),
]


@pytest.mark.parametrize(
    "signal_fn,kwargs,expected_fn", CASES, ids=lambda c: getattr(c, "__name__", "")
)
def test_position_follows_raw_signal_by_exactly_one_bar(signal_fn, kwargs, expected_fn):
    prices = random_walk(600, n_assets=2, seed=11)
    raw = expected_fn(prices).fillna(0).to_numpy()
    assert np.abs(raw).sum() > 0, "fixture never triggers this signal"

    signals = signal_fn(prices, prices.pct_change().fillna(0), **kwargs)
    _, positions, _ = _simulate_portfolio_numba(
        prices.to_numpy(), signals.to_numpy(dtype=np.float64), 100_000.0, 0.0, 0.0, 0.10
    )

    np.testing.assert_array_equal(np.sign(positions[1:]), raw[:-1])
