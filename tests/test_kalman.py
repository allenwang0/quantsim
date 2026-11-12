"""
Kalman hedge-ratio contract for PairsTradingStrategy.

These are the same cases as statarb's tests/test_kalman.py (constant beta,
beta step at bar 100, price-level invariance) plus the event-engine case:
the strategy sees one bar per leg per date and must update the filter once
per date, with both legs' prices from that date.
Prices are seeded synthetic random walks.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from quantsim.core.event_queue import EventQueue
from quantsim.core.events import BarEvent
from quantsim.strategies.mean_reversion import PairsTradingStrategy


class LatestCloseHandler:
    """Minimal data handler: returns the last close fed for each asset."""

    def __init__(self):
        self.closes = {}

    def get_latest_bars(self, asset_id, n=1):
        if asset_id not in self.closes:
            return pd.DataFrame()
        return pd.DataFrame({"close": [self.closes[asset_id]]})


def walk(n, seed=5):
    rng = np.random.default_rng(seed)
    return 50 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))


def run_pair(prices_a, prices_b):
    strategy = PairsTradingStrategy("A", "B", EventQueue(), z_window=20)
    updates = []
    original = strategy._kalman_update

    def counting_update(price_a, price_b):
        updates.append((price_a, price_b))
        return original(price_a, price_b)

    strategy._kalman_update = counting_update
    handler = LatestCloseHandler()
    betas = []
    start = datetime(2021, 1, 4, 21)
    for i, (a, b) in enumerate(zip(prices_a, prices_b)):
        ts = start + timedelta(days=i)
        for asset, price in (("A", a), ("B", b)):
            handler.closes[asset] = price
            event = BarEvent(
                timestamp=ts,
                asset_id=asset,
                open=price,
                high=price,
                low=price,
                close=price,
                adj_close=price,
                volume=1,
            )
            strategy.on_bar(event, handler)
        betas.append(strategy._kalman_beta)
    return strategy, updates, np.array(betas)


def test_constant_ratio_updates_once_per_date_with_same_day_prices():
    b = walk(100)
    a = 2.0 * b

    strategy, updates, _ = run_pair(a, b)

    assert len(updates) == 100
    assert all(pa == pytest.approx(2.0 * pb) for pa, pb in updates)
    assert strategy._kalman_beta == pytest.approx(2.0, abs=1e-3)


def test_beta_step_at_bar_100_is_tracked():
    b = walk(200)
    beta = np.where(np.arange(200) < 100, 1.0, 2.0)
    a = beta * b

    _, _, betas = run_pair(a, b)

    assert betas[99] == pytest.approx(1.0, abs=1e-3)
    assert betas[-1] == pytest.approx(2.0, abs=1e-2)


def test_hedge_ratio_does_not_depend_on_price_level():
    rng = np.random.default_rng(9)
    b = walk(150)
    a = 1.5 * b + rng.normal(0, 0.2, 150)

    _, _, betas = run_pair(a, b)
    _, _, scaled_betas = run_pair(100 * a, 100 * b)

    np.testing.assert_allclose(scaled_betas, betas, rtol=1e-9)
