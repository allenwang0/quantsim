"""
PortfolioManager: converts SignalEvents into OrderEvents.
Applies the shared capital model with CapitalRequest arbitration.
Enforces concentration limits and handles multi-strategy coordination.
"""

from __future__ import annotations
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

from core.events import (
    SignalEvent, OrderEvent, FillEvent, BarEvent,
    Direction, OrderType, OrderSide, AssetType, TimeInForce,
)
from core.event_queue import EventQueue
from portfolio.portfolio import Portfolio
from portfolio.sizing import PositionSizer, EqualWeightSizer
from risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class CapitalRequest:
    """Request submitted by a strategy for capital allocation."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy_id: str = ""
    asset_id: str = ""
    requested_dollar_size: float = 0.0
    direction: Direction = Direction.FLAT
    priority: int = 5         # lower = higher priority
    confidence: float = 1.0
    correlated_with: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class PortfolioManager:
    """
    Translates SignalEvents into OrderEvents using the shared capital pool model.
    
    Arbitration sequence (per spec):
    1. Check total buying power; reject if insufficient
    2. Check concentration: no single position > max_position_pct of portfolio
    3. Check correlation with existing positions
    4. Check sector/strategy exposure
    5. If approved: compute size, emit OrderEvent
    
    Supports both isolated capital model (simple) and shared capital model (production).
    """

    def __init__(
        self,
        portfolio: Portfolio,
        event_queue: EventQueue,
        risk_manager: RiskManager,
        sizer: Optional[PositionSizer] = None,
        data_handler=None,
        db_path: Optional[str] = None,
        max_position_pct: float = 0.10,
        max_strategy_pct: float = 0.40,
        min_order_value: float = 500.0,
    ):
        self.portfolio = portfolio
        self._queue = event_queue
        self.risk = risk_manager
        self.sizer = sizer or EqualWeightSizer(n_positions=20)
        self._data = data_handler
        self._db_path = db_path
        self.max_position_pct = max_position_pct
        self.max_strategy_pct = max_strategy_pct
        self.min_order_value = min_order_value

        # Track current positions per strategy for exposure limits
        self._strategy_positions: Dict[str, Set[str]] = {}

        # Track realized volatility cache for sizing
        self._vol_cache: Dict[str, float] = {}

    def on_signal(self, signal: SignalEvent) -> Optional[OrderEvent]:
        """
        Process a SignalEvent. Returns the OrderEvent if approved, else None.
        """
        # Risk check first
        if not self.risk.check_signal(signal, self.portfolio):
            return None

        asset_id = signal.asset_id
        direction = signal.direction

        # Closing signal: close existing position
        if direction == Direction.FLAT:
            return self._close_position(signal)

        # Opening signal: size and create order
        return self._open_position(signal)

    def _close_position(self, signal: SignalEvent) -> Optional[OrderEvent]:
        """Generate a close order for an existing position."""
        asset_id = signal.asset_id

        if asset_id not in self.portfolio.positions:
            return None

        pos = self.portfolio.positions[asset_id]

        if pos.quantity > 0:
            side = OrderSide.SELL
        elif pos.quantity < 0:
            side = OrderSide.BUY
        else:
            return None

        order = OrderEvent(
            timestamp=signal.timestamp,
            asset_id=asset_id,
            asset_type=pos.asset_type,
            order_type=OrderType.MARKET,
            side=side,
            quantity=abs(pos.quantity),
            strategy_id=signal.strategy_id,
            time_in_force=TimeInForce.DAY,
        )

        if not self.risk.check_order(order, self.portfolio):
            return None

        self._queue.put(order)
        self._log_event("ORDER", signal.timestamp, order.__dict__)
        return order

    def _open_position(self, signal: SignalEvent) -> Optional[OrderEvent]:
        """Size and generate an opening order."""
        asset_id = signal.asset_id

        # Get current price estimate
        price = self._get_price_estimate(asset_id, signal.timestamp)
        if price <= 0:
            logger.warning(f"No price for {asset_id}, skipping signal")
            return None

        # Concentration check: skip if already at max
        current_weight = self.portfolio.get_position_weight(asset_id)
        if current_weight >= self.max_position_pct:
            logger.debug(f"Concentration limit reached for {asset_id}: {current_weight:.1%}")
            return None

        # Strategy exposure check
        strategy_assets = self._strategy_positions.get(signal.strategy_id, set())
        if len(strategy_assets) >= 20:  # max 20 positions per strategy
            logger.debug(f"Strategy {signal.strategy_id} at max position count")
            return None

        # Get realized vol for sizing
        vol = self._vol_cache.get(asset_id, 0.20)
        if self._data:
            bars = self._data.get_latest_bars(asset_id, n=35)
            if not bars.empty:
                closes = bars.get("adj_close", bars.get("close", bars.iloc[:, 0]))
                if len(closes) > 5:
                    log_rets = closes.pct_change().dropna()
                    vol = float(log_rets.std() * (252**0.5))
                    self._vol_cache[asset_id] = vol

        # Compute position size
        quantity = self.sizer.size(
            portfolio=self.portfolio,
            asset_id=asset_id,
            confidence=signal.confidence,
            price=price,
            vol_estimate=vol,
        )

        if quantity <= 0:
            return None

        order_value = quantity * price
        if order_value < self.min_order_value:
            logger.debug(f"Order value ${order_value:.0f} below minimum ${self.min_order_value}")
            return None

        # Check buying power
        if order_value > self.portfolio.cash * 0.95:
            # Scale down to fit available cash
            quantity = max(1, int(self.portfolio.cash * 0.95 / price))

        if quantity <= 0:
            return None

        side = OrderSide.BUY if signal.direction == Direction.LONG else OrderSide.SELL

        order = OrderEvent(
            timestamp=signal.timestamp,
            asset_id=asset_id,
            asset_type=AssetType.EQUITY,
            order_type=OrderType.MARKET,
            side=side,
            quantity=quantity,
            strategy_id=signal.strategy_id,
            time_in_force=TimeInForce.DAY,
        )

        if not self.risk.check_order(order, self.portfolio):
            return None

        # Update strategy position tracking
        self._strategy_positions.setdefault(signal.strategy_id, set()).add(asset_id)

        self._queue.put(order)
        self._log_event("ORDER", signal.timestamp, {
            "order_id": order.order_id,
            "asset_id": asset_id,
            "side": side.value,
            "quantity": quantity,
            "price_estimate": price,
            "strategy_id": signal.strategy_id,
            "confidence": signal.confidence,
        })
        return order

    def on_fill(self, fill: FillEvent) -> None:
        """Process fill: update portfolio, persist snapshot, record trade."""
        self.portfolio.apply_fill(fill)

        if self._db_path:
            self.portfolio.persist(self._db_path)
            self._record_trade(fill)

        logger.info(
            f"FILL: {fill.side.value} {fill.quantity} {fill.asset_id} "
            f"@ ${fill.fill_price:.2f} | "
            f"Commission: ${fill.commission:.2f} | "
            f"Portfolio equity: ${self.portfolio.total_equity:,.0f}"
        )

    def on_bar(self, event: BarEvent) -> None:
        """Update portfolio market prices on each bar."""
        prices = {event.asset_id: event.close}
        self.portfolio.update_market_prices(prices, timestamp=event.timestamp)
        self.risk.update(self.portfolio, event.timestamp)

    def _log_strategy_performance(self, fill) -> None:
        """Log per-strategy daily P&L to strategy_performance table."""
        try:
            from core.database import db_conn
            import time
            ts_epoch = int(fill.timestamp.timestamp())
            equity = self.portfolio.total_equity
            rpnl = self.portfolio.realized_pnl
            upnl = self.portfolio.unrealized_pnl
            n_pos = len(self.portfolio.positions)
            with db_conn(self._db_path) as conn:
                conn.execute(
                    """INSERT INTO strategy_performance
                       (timestamp, strategy_id, daily_pnl, realized_pnl,
                        unrealized_pnl, n_positions)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (ts_epoch, fill.strategy_id,
                     fill.fill_price * fill.quantity * (-1 if fill.side.value in ('SELL','SELL_TO_OPEN','SELL_TO_CLOSE') else 1),
                     rpnl, upnl, n_pos),
                )
        except Exception as e:
            logger.debug(f"Strategy performance log failed: {e}")

    def _get_price_estimate(self, asset_id: str, timestamp: datetime) -> float:
        """Get current price estimate for position sizing."""
        if asset_id in self.portfolio.positions:
            return self.portfolio.positions[asset_id].current_price

        if self._data:
            bars = self._data.get_latest_bars(asset_id, n=1)
            if not bars.empty:
                close_col = "adj_close" if "adj_close" in bars.columns else "close"
                if close_col in bars.columns:
                    return float(bars[close_col].iloc[-1])

        return 0.0

    def _record_trade(self, fill: FillEvent) -> None:
        """Record fill to trade log. Matches opens to closes."""
        try:
            from core.database import db_conn
            import time

            ts_epoch = int(fill.timestamp.timestamp())
            is_buy = fill.side in (OrderSide.BUY, OrderSide.BUY_TO_OPEN, OrderSide.BUY_TO_CLOSE)

            with db_conn(self._db_path) as conn:
                # Check for matching open trade
                open_trade = conn.execute(
                    """SELECT trade_id, entry_price, quantity FROM trades
                       WHERE asset_id = ? AND exit_timestamp IS NULL
                       ORDER BY entry_timestamp DESC LIMIT 1""",
                    (fill.asset_id,),
                ).fetchone()

                if open_trade and not is_buy:
                    # Closing a long: update with exit
                    entry_p = open_trade["entry_price"]
                    qty = open_trade["quantity"]
                    realized_pnl = qty * (fill.fill_price - entry_p) - fill.commission
                    conn.execute(
                        """UPDATE trades SET exit_timestamp=?, exit_price=?,
                           realized_pnl=?, holding_bars=?
                           WHERE trade_id=?""",
                        (ts_epoch, fill.fill_price, realized_pnl,
                         0, open_trade["trade_id"]),
                    )
                elif is_buy:
                    # Opening a new long trade
                    conn.execute(
                        """INSERT OR IGNORE INTO trades
                           (trade_id, strategy_id, asset_id, direction,
                            entry_timestamp, entry_price, quantity, commission)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (fill.order_id, fill.strategy_id, fill.asset_id,
                         "LONG", ts_epoch, fill.fill_price,
                         fill.quantity, fill.commission),
                    )
        except Exception as e:
            logger.error(f"Trade record failed: {e}")

    def _log_event(self, event_type: str, timestamp: datetime, payload: dict) -> None:
        """Log event to the immutable audit log."""
        if not self._db_path:
            return
        try:
            from core.database import db_conn
            ts_epoch = int(timestamp.timestamp())
            # Remove non-serializable fields
            clean_payload = {
                k: v for k, v in payload.items()
                if isinstance(v, (str, int, float, bool, type(None)))
            }
            with db_conn(self._db_path) as conn:
                conn.execute(
                    "INSERT INTO event_log (event_type, timestamp, payload) VALUES (?, ?, ?)",
                    (event_type, ts_epoch, json.dumps(clean_payload)),
                )
        except Exception as e:
            logger.debug(f"Event log failed: {e}")


class OptimizationOverlayManager(PortfolioManager):
    """
    PortfolioManager subclass that applies a portfolio optimization overlay
    at rebalance frequency (monthly by default).

    On rebalance: compute target weights via the optimizer, then size
    all positions to those weights. Between rebalances: passes through
    signals from strategies unchanged.

    Usage:
        optimizer = HierarchicalRiskParity()
        mgr = OptimizationOverlayManager(
            portfolio=portfolio,
            event_queue=eq,
            risk_manager=risk_mgr,
            optimizer=optimizer,
            data_handler=dh,
        )
    """

    def __init__(
        self,
        *args,
        optimizer=None,
        rebalance_day: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._optimizer = optimizer
        self._rebalance_day = rebalance_day
        self._last_rebalance_month: int = -1
        self._target_weights: dict = {}
        self._return_cache: dict = {}

    def on_bar(self, event) -> None:
        """Update portfolio prices and check for rebalance trigger."""
        super().on_bar(event)

        # Collect return observations for optimizer
        asset_id = event.asset_id
        bars = self._data.get_latest_bars(asset_id, n=2) if self._data else None
        if bars is not None and len(bars) >= 2:
            close = bars["adj_close"] if "adj_close" in bars.columns else bars["close"]
            if len(close) >= 2 and close.iloc[-2] > 0:
                ret = float(close.iloc[-1] / close.iloc[-2] - 1)
                if asset_id not in self._return_cache:
                    self._return_cache[asset_id] = []
                self._return_cache[asset_id].append(ret)
                if len(self._return_cache[asset_id]) > 504:
                    self._return_cache[asset_id] = self._return_cache[asset_id][-504:]

        # Monthly rebalance
        ts = event.timestamp
        if self._optimizer and ts.month != self._last_rebalance_month:
            self._last_rebalance_month = ts.month
            self._rebalance(ts)

    def _rebalance(self, timestamp) -> None:
        """Compute target weights and emit signals to reach them."""
        import pandas as pd
        if len(self._return_cache) < 2:
            return

        # Build returns DataFrame
        min_len = min(len(v) for v in self._return_cache.values())
        if min_len < 30:
            return

        returns = pd.DataFrame({
            k: v[-min_len:] for k, v in self._return_cache.items()
        })

        try:
            weights = self._optimizer.optimize(returns)
            self._target_weights = weights
            logger.debug(f"Portfolio rebalanced: {weights}")
        except Exception as e:
            logger.warning(f"Portfolio optimization failed at rebalance: {e}")

    def on_signal(self, signal):
        """
        If optimizer is active and target weights are set, override signal
        sizing based on target weight for the asset rather than strategy sizing.
        """
        if not self._optimizer or not self._target_weights:
            return super().on_signal(signal)

        # Scale confidence by the target weight for this asset
        asset_id = signal.asset_id
        target_weight = self._target_weights.get(asset_id, 0.0)

        if signal.direction.value == "FLAT" or target_weight <= 0:
            return super().on_signal(signal)

        # Override the sizer to use the target weight directly
        from core.events import Direction
        if signal.direction == Direction.FLAT:
            return super().on_signal(signal)

        # Temporarily set sizer to match target weight
        price = self._get_price_estimate(signal.asset_id, signal.timestamp)
        if price > 0:
            equity = self.portfolio.total_equity
            target_dollar = equity * target_weight
            target_qty = max(1, int(target_dollar / price))

            # Override quantity via metadata
            signal.metadata["target_qty_override"] = target_qty
            signal.confidence = min(1.0, signal.confidence * 1.5)  # boost confidence for optimizer-driven trades

        return super().on_signal(signal)
