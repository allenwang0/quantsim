"""
VolatilityTargetingStrategy must change position size, not just compute a
leverage number: a calm asset gets its base signal scaled up, a volatile one
scaled down, within [min_leverage, max_leverage]. Synthetic prices.
"""

import numpy as np
import pandas as pd
import pytest

from quantsim.core.event_queue import EventQueue
from quantsim.core.events import BarEvent, Direction
from quantsim.strategies.momentum_factor import VolatilityTargetingStrategy
from quantsim.strategies.trend import Strategy


class AlwaysLong(Strategy):
    def __init__(self, event_queue):
        super().__init__("AlwaysLong", ["SYN"], event_queue, warmup_bars=0)

    def on_bar(self, event, data_handler):
        self._emit_signal(event.asset_id, Direction.LONG, event.timestamp, confidence=0.5)


class FixedBars:
    def __init__(self, closes):
        self.frame = pd.DataFrame({"close": closes})

    def get_latest_bars(self, asset_id, n=1):
        return self.frame.iloc[-n:]


@pytest.mark.parametrize("daily_vol,expected", [(0.002, 2.0), (0.05, 0.25)])
def test_signal_confidence_scaled_by_capped_leverage(daily_vol, expected):
    rng = np.random.default_rng(4)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, daily_vol, 40)))
    queue = EventQueue()
    overlay = VolatilityTargetingStrategy(["SYN"], queue, AlwaysLong(EventQueue()))

    overlay.on_bar(BarEvent(asset_id="SYN", close=closes[-1]), FixedBars(closes))

    signal = queue.get()
    assert queue.empty()
    assert signal.confidence == pytest.approx(0.5 * expected)
    assert signal.metadata["vol_target_leverage"] == pytest.approx(expected)
