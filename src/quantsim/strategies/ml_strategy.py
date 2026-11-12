"""
TurbulenceFilteredStrategy: moves any strategy to cash when market
turbulence is in the top decile of its history.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from quantsim.core.event_queue import EventQueue
from quantsim.core.events import BarEvent, Direction
from quantsim.data.data_handler import DataHandler
from quantsim.strategies.trend import Strategy

logger = logging.getLogger(__name__)


class TurbulenceFilteredStrategy(Strategy):
    """
    Wrapper that applies a turbulence index filter to any strategy.

    When market turbulence (Mahalanobis distance of returns) exceeds
    the 90th percentile of its historical distribution, the strategy
    goes to cash. This is the FinRL-inspired risk-off mechanism.

    Usage:
        base = SMAcrossover(asset_ids=["SPY"], event_queue=eq)
        safe = TurbulenceFilteredStrategy(base_strategy=base, ...)
    """

    def __init__(
        self,
        base_strategy: Strategy,
        event_queue: EventQueue,
        lookback_bars: int = 252,
        turbulence_threshold_pct: float = 0.90,
    ):
        super().__init__(
            strategy_id=f"TurbFilter_{base_strategy.strategy_id}",
            asset_ids=base_strategy.asset_ids,
            event_queue=event_queue,
            warmup_bars=base_strategy.warmup_bars + lookback_bars,
        )
        self.base = base_strategy
        self.base._queue = event_queue  # share queue
        self.lookback = lookback_bars
        self.threshold_pct = turbulence_threshold_pct

        self._return_history: List[Dict[str, float]] = []
        self._turbulence_history: List[float] = []
        self._in_turbulence: bool = False
        self._last_asset_prices: Dict[str, float] = {}

    def on_bar(self, event: BarEvent, data_handler: DataHandler) -> None:
        asset_id = event.asset_id
        self._tick(asset_id)

        # Track cross-sectional returns for turbulence computation
        if asset_id in self._last_asset_prices:
            prev = self._last_asset_prices[asset_id]
            if prev > 0:
                ret = (event.adj_close - prev) / prev
                if len(self._return_history) == 0 or asset_id not in self._return_history[-1]:
                    self._return_history.append({})
                self._return_history[-1][asset_id] = ret

        self._last_asset_prices[asset_id] = event.adj_close

        # Only compute turbulence once per bar (on first asset trigger)
        if asset_id == self.asset_ids[0] and len(self._return_history) > self.lookback:
            self._update_turbulence()

        # If in turbulence regime: suppress signals, emit FLAT for all positions
        if self._in_turbulence:
            self._emit_signal(
                asset_id=asset_id,
                direction=Direction.FLAT,
                timestamp=event.timestamp,
                signal_type="turbulence_exit",
                metadata={"turbulence_active": True},
            )
            return

        # Normal regime: delegate to base strategy
        self.base.on_bar(event, data_handler)

    def _update_turbulence(self) -> None:
        """Compute current turbulence and update regime flag."""
        if len(self._return_history) < self.lookback:
            return

        recent = self._return_history[-self.lookback :]
        # Build return matrix: rows=dates, cols=assets
        all_assets = list(self.asset_ids)
        matrix = []
        for day_rets in recent:
            row = [day_rets.get(a, 0.0) for a in all_assets]
            matrix.append(row)

        arr = np.array(matrix)
        if arr.shape[0] < 30 or arr.shape[1] < 2:
            return

        mu = arr[:-1].mean(axis=0)
        cov = np.cov(arr[:-1].T) + np.eye(len(all_assets)) * 1e-8

        current = arr[-1]
        try:
            cov_inv = np.linalg.pinv(cov)
            diff = current - mu
            turb = float(diff @ cov_inv @ diff.T)
        except Exception:
            turb = 0.0

        self._turbulence_history.append(turb)

        # Determine threshold from history
        if len(self._turbulence_history) >= 50:
            threshold = np.percentile(self._turbulence_history, self.threshold_pct * 100)
            self._in_turbulence = turb > threshold
        else:
            self._in_turbulence = False
