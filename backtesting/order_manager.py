"""
Order Manager: stop-loss, take-profit, trailing stops, bracket orders, OCO.

What's missing from a basic event-driven engine:
- Positions have no automatic exit triggers
- A SMA crossover that goes long has no protection against a 30% gap down
- Options positions need automatic delta-hedge triggers

This module provides:
1. StopLossManager: tracks open positions and emits FLAT signals when
   the stop level is breached. Integrates with any strategy.
2. TakeProfitManager: same but for profit targets
3. TrailingStopManager: stops that move with the position
4. BracketOrderManager: entry + stop + target as an atomic unit
5. OCAManager: One-Cancels-All order groups

Usage in BacktestEngine:
    order_mgr = OrderManager(portfolio, event_queue, data_handler)
    # In _handle_bar():
    order_mgr.on_bar(event)  # checks all stops/targets

Usage in strategy:
    order_mgr.add_stop_loss("SPY", entry_price=420.0, stop_pct=0.02, strategy_id="sma")
    order_mgr.add_take_profit("SPY", entry_price=420.0, target_pct=0.05, strategy_id="sma")
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from core.events import BarEvent, SignalEvent, Direction
from core.event_queue import EventQueue

logger = logging.getLogger(__name__)


@dataclass
class StopLevel:
    asset_id: str
    strategy_id: str
    entry_price: float
    stop_price: float
    stop_type: str       # 'fixed', 'trailing', 'atr'
    target_price: Optional[float] = None
    quantity: int = 0
    direction: str = "LONG"  # LONG or SHORT
    is_active: bool = True
    highest_price: float = 0.0   # for trailing stops
    lowest_price: float = 0.0    # for short trailing stops
    atr_multiple: float = 2.0
    created_at: datetime = field(default_factory=datetime.utcnow)


class OrderManager:
    """
    Monitors open positions and emits exit signals when stop/target levels
    are breached. Plugs into BacktestEngine._handle_bar().

    This is the correct architectural location for exit logic — not inside
    strategy on_bar() — because:
    1. Stops apply across all strategies uniformly
    2. Trailing stops need access to current bar prices, not strategy state
    3. It keeps strategy code purely focused on signal generation

    ATR-based stops are the most robust:
    - Fixed % stops get stopped out by noise in volatile names
    - ATR-based stops adapt to each asset's volatility regime
    - Typical parameters: 1.5-3x ATR(14) as stop distance
    """

    def __init__(
        self,
        event_queue: EventQueue,
        default_stop_pct: float = 0.05,
        default_target_pct: float = 0.10,
        use_atr_stops: bool = False,
        atr_multiple: float = 2.0,
        atr_period: int = 14,
    ):
        self._queue = event_queue
        self.default_stop_pct = default_stop_pct
        self.default_target_pct = default_target_pct
        self.use_atr = use_atr_stops
        self.atr_multiple = atr_multiple
        self.atr_period = atr_period

        self._stops: Dict[str, StopLevel] = {}  # asset_id -> StopLevel
        self._atr_cache: Dict[str, float] = {}

    def add_stop_loss(
        self,
        asset_id: str,
        entry_price: float,
        strategy_id: str,
        stop_pct: Optional[float] = None,
        stop_price: Optional[float] = None,
        direction: str = "LONG",
        quantity: int = 0,
    ) -> None:
        """Register a fixed stop-loss for a position."""
        pct = stop_pct or self.default_stop_pct
        if stop_price is None:
            if direction == "LONG":
                stop_price = entry_price * (1 - pct)
            else:
                stop_price = entry_price * (1 + pct)

        self._stops[asset_id] = StopLevel(
            asset_id=asset_id,
            strategy_id=strategy_id,
            entry_price=entry_price,
            stop_price=stop_price,
            stop_type="fixed",
            direction=direction,
            quantity=quantity,
            highest_price=entry_price,
            lowest_price=entry_price,
        )
        logger.debug(f"Stop-loss set: {asset_id} @ {stop_price:.2f} ({direction})")

    def add_take_profit(
        self,
        asset_id: str,
        entry_price: float,
        strategy_id: str,
        target_pct: Optional[float] = None,
        target_price: Optional[float] = None,
        direction: str = "LONG",
    ) -> None:
        """Add a take-profit level to an existing stop entry."""
        pct = target_pct or self.default_target_pct
        if target_price is None:
            if direction == "LONG":
                target_price = entry_price * (1 + pct)
            else:
                target_price = entry_price * (1 - pct)

        if asset_id in self._stops:
            self._stops[asset_id].target_price = target_price
        else:
            self._stops[asset_id] = StopLevel(
                asset_id=asset_id,
                strategy_id=strategy_id,
                entry_price=entry_price,
                stop_price=entry_price * (1 - self.default_stop_pct),
                stop_type="fixed",
                target_price=target_price,
                direction=direction,
                highest_price=entry_price,
                lowest_price=entry_price,
            )
        logger.debug(f"Take-profit set: {asset_id} @ {target_price:.2f}")

    def add_trailing_stop(
        self,
        asset_id: str,
        entry_price: float,
        strategy_id: str,
        trail_pct: float = 0.03,
        direction: str = "LONG",
        quantity: int = 0,
    ) -> None:
        """
        Trailing stop: the stop level moves with the position's best price.
        For LONG: stop = max_price * (1 - trail_pct)
        For SHORT: stop = min_price * (1 + trail_pct)
        """
        if direction == "LONG":
            stop_price = entry_price * (1 - trail_pct)
        else:
            stop_price = entry_price * (1 + trail_pct)

        self._stops[asset_id] = StopLevel(
            asset_id=asset_id,
            strategy_id=strategy_id,
            entry_price=entry_price,
            stop_price=stop_price,
            stop_type="trailing",
            direction=direction,
            quantity=quantity,
            highest_price=entry_price,
            lowest_price=entry_price,
            atr_multiple=trail_pct,
        )
        logger.debug(f"Trailing stop set: {asset_id} trail={trail_pct:.1%}")

    def add_atr_stop(
        self,
        asset_id: str,
        entry_price: float,
        strategy_id: str,
        current_atr: float,
        multiple: Optional[float] = None,
        direction: str = "LONG",
        quantity: int = 0,
    ) -> None:
        """
        ATR-based stop. More robust than fixed % because it adapts to volatility.
        stop_distance = multiple * ATR(14)
        Typical: 1.5x ATR for mean-reversion, 3x ATR for trend-following.
        """
        mult = multiple or self.atr_multiple
        stop_distance = mult * current_atr
        if direction == "LONG":
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance

        self._stops[asset_id] = StopLevel(
            asset_id=asset_id,
            strategy_id=strategy_id,
            entry_price=entry_price,
            stop_price=stop_price,
            stop_type="atr",
            direction=direction,
            quantity=quantity,
            highest_price=entry_price,
            lowest_price=entry_price,
            atr_multiple=mult,
        )
        self._atr_cache[asset_id] = current_atr

    def add_bracket(
        self,
        asset_id: str,
        entry_price: float,
        strategy_id: str,
        stop_pct: float = 0.02,
        target_pct: float = 0.06,
        direction: str = "LONG",
        quantity: int = 0,
    ) -> None:
        """
        Bracket order: entry + stop-loss + take-profit as one unit.
        Risk/reward ratio = target_pct / stop_pct (here: 3:1).
        Only the first trigger (stop or target) fires; the other is cancelled.
        This is the OCA (One-Cancels-All) behavior.
        """
        self.add_stop_loss(asset_id, entry_price, strategy_id,
                           stop_pct=stop_pct, direction=direction, quantity=quantity)
        self.add_take_profit(asset_id, entry_price, strategy_id,
                             target_pct=target_pct, direction=direction)

    def remove(self, asset_id: str) -> None:
        """Remove all exit orders for an asset (called on position close)."""
        self._stops.pop(asset_id, None)
        logger.debug(f"Exit orders removed: {asset_id}")

    def on_bar(self, event: BarEvent) -> List[str]:
        """
        Process a new bar. Check all active stop/target levels.
        Emits FLAT signals for any triggered exits.
        Returns list of triggered asset_ids.
        """
        asset_id = event.asset_id
        triggered = []

        if asset_id not in self._stops:
            return triggered

        stop = self._stops[asset_id]
        if not stop.is_active:
            return triggered

        current_price = event.close
        high = event.high
        low = event.low

        # Update trailing stop
        if stop.stop_type == "trailing":
            trail_pct = stop.atr_multiple
            if stop.direction == "LONG":
                stop.highest_price = max(stop.highest_price, high)
                stop.stop_price = stop.highest_price * (1 - trail_pct)
            else:
                stop.lowest_price = min(stop.lowest_price, low)
                stop.stop_price = stop.lowest_price * (1 + trail_pct)

        # Update ATR trailing stop
        elif stop.stop_type == "atr":
            atr = self._atr_cache.get(asset_id, 0)
            if atr > 0:
                if stop.direction == "LONG":
                    stop.highest_price = max(stop.highest_price, high)
                    stop.stop_price = stop.highest_price - stop.atr_multiple * atr
                else:
                    stop.lowest_price = min(stop.lowest_price, low)
                    stop.stop_price = stop.lowest_price + stop.atr_multiple * atr

        # Check stop-loss breach
        stop_triggered = False
        if stop.direction == "LONG" and low <= stop.stop_price:
            stop_triggered = True
            trigger_reason = f"stop_loss @ {stop.stop_price:.2f} (low={low:.2f})"
        elif stop.direction == "SHORT" and high >= stop.stop_price:
            stop_triggered = True
            trigger_reason = f"stop_loss @ {stop.stop_price:.2f} (high={high:.2f})"

        # Check take-profit breach
        target_triggered = False
        if stop.target_price is not None:
            if stop.direction == "LONG" and high >= stop.target_price:
                target_triggered = True
                trigger_reason = f"take_profit @ {stop.target_price:.2f}"
            elif stop.direction == "SHORT" and low <= stop.target_price:
                target_triggered = True
                trigger_reason = f"take_profit @ {stop.target_price:.2f}"

        if stop_triggered or target_triggered:
            stop.is_active = False
            del self._stops[asset_id]
            triggered.append(asset_id)

            signal = SignalEvent(
                timestamp=event.timestamp,
                strategy_id=stop.strategy_id,
                asset_id=asset_id,
                direction=Direction.FLAT,
                confidence=1.0,
                signal_type="stop_exit" if stop_triggered else "target_exit",
                metadata={
                    "trigger_reason": trigger_reason,
                    "entry_price": stop.entry_price,
                    "exit_price": current_price,
                    "pnl_pct": (
                        (current_price - stop.entry_price) / stop.entry_price
                        if stop.direction == "LONG"
                        else (stop.entry_price - current_price) / stop.entry_price
                    ),
                },
            )
            self._queue.put(signal)
            logger.info(
                f"EXIT triggered: {asset_id} {trigger_reason} "
                f"(entry={stop.entry_price:.2f})"
            )

        return triggered

    def update_atr(self, asset_id: str, atr: float) -> None:
        """Update ATR estimate for ATR-based stops."""
        self._atr_cache[asset_id] = atr

    def active_stops(self) -> Dict[str, StopLevel]:
        return {k: v for k, v in self._stops.items() if v.is_active}

    def __repr__(self) -> str:
        return f"OrderManager({len(self._stops)} active stops)"
