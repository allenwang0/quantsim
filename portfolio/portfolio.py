"""
Portfolio management layer.
Tracks positions, cash, P&L. Supports both isolated and shared capital models.
Persisted to SQLite on every FillEvent.
"""

from __future__ import annotations
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from core.events import FillEvent, OrderSide, AssetType

logger = logging.getLogger(__name__)


@dataclass
class Position:
    asset_id: str
    asset_type: AssetType
    quantity: int                    # negative = short
    average_cost: float
    current_price: float
    entry_date: datetime
    strategy_id: str
    realized_pnl: float = 0.0
    # Options-specific
    option_symbol: Optional[str] = None
    expiration: Optional[str] = None
    strike: Optional[float] = None
    right: Optional[str] = None      # 'C' or 'P'
    # Greeks (updated on each bar)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

    @property
    def multiplier(self) -> int:
        return 100 if self.asset_type == AssetType.OPTION else 1

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price * self.multiplier

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.current_price - self.average_cost) * self.multiplier

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "asset_type": self.asset_type.value,
            "quantity": self.quantity,
            "average_cost": self.average_cost,
            "current_price": self.current_price,
            "entry_date": self.entry_date.isoformat(),
            "strategy_id": self.strategy_id,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "market_value": self.market_value,
            "option_symbol": self.option_symbol,
            "expiration": self.expiration,
            "strike": self.strike,
            "right": self.right,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
        }


@dataclass
class Portfolio:
    initial_capital: float
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Equity curve tracking for drawdown calculation
    peak_equity: float = 0.0
    _equity_history: List[Tuple[datetime, float]] = field(default_factory=list)

    def __post_init__(self):
        self.peak_equity = self.initial_capital

    @property
    def total_equity(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        return self.total_equity - self.initial_capital

    @property
    def current_drawdown(self) -> float:
        """Current drawdown as negative fraction from peak."""
        equity = self.total_equity
        if self.peak_equity <= 0:
            return 0.0
        return (equity - self.peak_equity) / self.peak_equity

    @property
    def portfolio_delta(self) -> float:
        """Aggregate delta across all options positions."""
        total = 0.0
        for pos in self.positions.values():
            if pos.asset_type == AssetType.OPTION:
                total += pos.delta * pos.quantity * 100
            else:
                total += pos.quantity  # equity delta = quantity
        return total

    @property
    def portfolio_vega(self) -> float:
        return sum(
            p.vega * p.quantity * 100
            for p in self.positions.values()
            if p.asset_type == AssetType.OPTION
        )

    @property
    def portfolio_gamma(self) -> float:
        return sum(
            p.gamma * p.quantity * 100
            for p in self.positions.values()
            if p.asset_type == AssetType.OPTION
        )

    @property
    def portfolio_theta(self) -> float:
        return sum(
            p.theta * p.quantity * 100
            for p in self.positions.values()
            if p.asset_type == AssetType.OPTION
        )

    def update_peak(self) -> None:
        equity = self.total_equity
        if equity > self.peak_equity:
            self.peak_equity = equity

    def record_equity(self) -> None:
        self.update_peak()
        # Normalize timestamp: strip timezone for consistent equity history
        ts = self.timestamp
        if hasattr(ts, 'replace') and hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        self._equity_history.append((ts, self.total_equity))

    def max_drawdown(self) -> float:
        """Compute max drawdown across full equity history."""
        if not self._equity_history:
            return 0.0
        equities = [e for _, e in self._equity_history]
        peak = equities[0]
        max_dd = 0.0
        for e in equities:
            if e > peak:
                peak = e
            dd = (e - peak) / peak if peak > 0 else 0
            if dd < max_dd:
                max_dd = dd
        return max_dd  # negative number

    def get_position_weight(self, asset_id: str) -> float:
        """Position as fraction of total equity."""
        equity = self.total_equity
        if equity <= 0 or asset_id not in self.positions:
            return 0.0
        return abs(self.positions[asset_id].market_value) / equity

    def to_snapshot_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_equity": self.total_equity,
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "current_drawdown": self.current_drawdown,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
        }

    def apply_fill(self, fill: FillEvent) -> None:
        """
        Update portfolio state from a fill. Handles both opening and closing positions.
        Computes realized P&L on closes.
        """
        is_buy = fill.side in (
            OrderSide.BUY, OrderSide.BUY_TO_OPEN, OrderSide.BUY_TO_CLOSE
        )
        qty_signed = fill.quantity if is_buy else -fill.quantity
        multiplier = 100 if fill.asset_type == AssetType.OPTION else 1
        cost = fill.fill_price * fill.quantity * multiplier

        asset_key = fill.option_symbol or fill.asset_id

        if asset_key not in self.positions:
            # Opening a new position
            if qty_signed == 0:
                return
            self.positions[asset_key] = Position(
                asset_id=fill.asset_id,
                asset_type=fill.asset_type,
                quantity=qty_signed,
                average_cost=fill.fill_price,
                current_price=fill.fill_price,
                entry_date=fill.timestamp,
                strategy_id=fill.strategy_id,
                option_symbol=fill.option_symbol,
                expiration=fill.expiration,
                strike=fill.strike,
                right=fill.right,
            )
            # Deduct cash
            if is_buy:
                self.cash -= cost + fill.commission
            else:
                self.cash += cost - fill.commission
        else:
            pos = self.positions[asset_key]
            old_qty = pos.quantity
            new_qty = old_qty + qty_signed

            if new_qty == 0:
                # Closing position entirely
                realized = pos.quantity * (fill.fill_price - pos.average_cost) * multiplier
                pos.realized_pnl += realized
                self.realized_pnl += realized

                # Cash settlement
                if is_buy:
                    self.cash -= cost + fill.commission
                else:
                    self.cash += cost - fill.commission

                del self.positions[asset_key]

            elif (old_qty > 0 and qty_signed > 0) or (old_qty < 0 and qty_signed < 0):
                # Adding to position - update average cost
                total_cost_basis = pos.average_cost * abs(old_qty) + fill.fill_price * fill.quantity
                pos.quantity = new_qty
                pos.average_cost = total_cost_basis / abs(new_qty)

                if is_buy:
                    self.cash -= cost + fill.commission
                else:
                    self.cash += cost - fill.commission

            else:
                # Partial close - reduce position
                close_qty = min(abs(qty_signed), abs(old_qty))
                realized = (
                    close_qty * (fill.fill_price - pos.average_cost) * multiplier
                    * (1 if old_qty > 0 else -1)
                )
                pos.realized_pnl += realized
                self.realized_pnl += realized
                pos.quantity = new_qty

                if is_buy:
                    self.cash -= fill.fill_price * fill.quantity * multiplier + fill.commission
                else:
                    self.cash += fill.fill_price * fill.quantity * multiplier - fill.commission

        self.timestamp = fill.timestamp

    def update_market_prices(self, prices: Dict[str, float], timestamp=None) -> None:
        """Update current_price for all positions. Called on each BarEvent."""
        for asset_key, pos in self.positions.items():
            lookup_key = pos.asset_id
            if lookup_key in prices:
                pos.current_price = prices[lookup_key]
        if timestamp is not None:
            self.timestamp = timestamp
        self.record_equity()

    def update_greeks(self, asset_key: str, delta: float, gamma: float,
                       theta: float, vega: float) -> None:
        """Update Greeks for a specific options position.
        Called by the engine on each bar after repricing options.
        """
        if asset_key in self.positions:
            pos = self.positions[asset_key]
            pos.delta = delta
            pos.gamma = gamma
            pos.theta = theta
            pos.vega = vega

    def expire_options(self, as_of_date) -> list:
        """
        Remove expired options positions and book P&L.
        Call this on each bar to handle options that expired worthless
        or were exercised/assigned.
        
        Returns list of expired position asset_keys.
        """
        from datetime import date as date_type
        expired = []
        for asset_key, pos in list(self.positions.items()):
            if pos.asset_type.value != 'OPTION':
                continue
            if pos.expiration is None:
                continue
            # Parse expiration string to date
            exp = pos.expiration
            if isinstance(exp, str):
                try:
                    from datetime import datetime
                    exp = datetime.strptime(exp[:10], '%Y-%m-%d').date()
                except Exception:
                    continue
            
            # Get current date
            if hasattr(as_of_date, 'date'):
                cur_date = as_of_date.date()
            elif isinstance(as_of_date, date_type):
                cur_date = as_of_date
            else:
                continue
            
            if cur_date >= exp:
                # Option expired: realize P&L
                realized = pos.unrealized_pnl
                self.realized_pnl += realized
                del self.positions[asset_key]
                expired.append(asset_key)
        return expired

    def persist(self, db_path: str) -> None:
        """Persist portfolio snapshot to SQLite."""
        try:
            from core.database import db_conn
            snapshot = self.to_snapshot_dict()
            ts_epoch = int(self.timestamp.timestamp())
            with db_conn(db_path) as conn:
                conn.execute(
                    """INSERT INTO portfolio_snapshots
                       (timestamp, total_equity, cash, realized_pnl, unrealized_pnl, payload)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (ts_epoch, self.total_equity, self.cash,
                     self.realized_pnl, self.unrealized_pnl,
                     json.dumps(snapshot)),
                )
        except Exception as e:
            logger.error(f"Portfolio persist failed: {e}")
