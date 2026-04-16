"""
Core event types for the event-driven backtesting and paper trading engine.
All modules communicate exclusively through these events via the EventQueue.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
import uuid


class EventType(Enum):
    BAR = "BAR"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    RISK = "RISK"


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"
    BUY_TO_OPEN = "BUY_TO_OPEN"
    BUY_TO_CLOSE = "BUY_TO_CLOSE"
    SELL_TO_OPEN = "SELL_TO_OPEN"
    SELL_TO_CLOSE = "SELL_TO_CLOSE"


class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class TimeInForce(Enum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class AssetType(Enum):
    EQUITY = "EQUITY"
    ETF = "ETF"
    OPTION = "OPTION"
    CRYPTO = "CRYPTO"


class RiskEventType(Enum):
    MAX_DRAWDOWN_BREACH = "MAX_DRAWDOWN_BREACH"
    CONCENTRATION_BREACH = "CONCENTRATION_BREACH"
    GREEKS_BREACH = "GREEKS_BREACH"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"


# Priority for heapq: lower number = higher priority at same timestamp
EVENT_PRIORITY = {
    EventType.FILL: 0,
    EventType.RISK: 1,
    EventType.ORDER: 2,
    EventType.SIGNAL: 3,
    EventType.BAR: 4,
}


@dataclass
class BarEvent:
    event_type: EventType = field(default=EventType.BAR, init=False)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    asset_id: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    adj_close: float = 0.0  # computed at query time from raw + adjustment factors
    asset_type: AssetType = AssetType.EQUITY

    def __lt__(self, other):
        return (self.timestamp, EVENT_PRIORITY[self.event_type]) < (
            other.timestamp, EVENT_PRIORITY[other.event_type]
        )


@dataclass
class SignalEvent:
    event_type: EventType = field(default=EventType.SIGNAL, init=False)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    strategy_id: str = ""
    asset_id: str = ""
    direction: Direction = Direction.FLAT
    confidence: float = 1.0           # [0, 1]
    signal_type: str = ""             # 'trend', 'mean_reversion', 'momentum', etc.
    holding_period_estimate: int = 1  # expected bars to hold
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        return (self.timestamp, EVENT_PRIORITY[self.event_type]) < (
            other.timestamp, EVENT_PRIORITY[other.event_type]
        )


@dataclass
class OrderEvent:
    event_type: EventType = field(default=EventType.ORDER, init=False)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    asset_id: str = ""
    asset_type: AssetType = AssetType.EQUITY
    order_type: OrderType = OrderType.MARKET
    side: OrderSide = OrderSide.BUY
    quantity: int = 0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trailing_amount: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    strategy_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: Optional[float] = None
    legs: Optional[List["OrderEvent"]] = None  # multi-leg options
    # Options-specific
    option_symbol: Optional[str] = None
    expiration: Optional[str] = None
    strike: Optional[float] = None
    right: Optional[str] = None  # 'C' or 'P'

    def __lt__(self, other):
        return (self.timestamp, EVENT_PRIORITY[self.event_type]) < (
            other.timestamp, EVENT_PRIORITY[other.event_type]
        )


@dataclass
class FillEvent:
    event_type: EventType = field(default=EventType.FILL, init=False)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    order_id: str = ""
    asset_id: str = ""
    asset_type: AssetType = AssetType.EQUITY
    side: OrderSide = OrderSide.BUY
    quantity: int = 0
    fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    strategy_id: str = ""
    # Options-specific
    option_symbol: Optional[str] = None
    expiration: Optional[str] = None
    strike: Optional[float] = None
    right: Optional[str] = None

    @property
    def total_cost(self) -> float:
        multiplier = 100 if self.asset_type == AssetType.OPTION else 1
        base = self.quantity * self.fill_price * multiplier
        if self.side in (OrderSide.BUY, OrderSide.BUY_TO_OPEN, OrderSide.BUY_TO_CLOSE):
            return base + self.commission
        return -base + self.commission

    def __lt__(self, other):
        return (self.timestamp, EVENT_PRIORITY[self.event_type]) < (
            other.timestamp, EVENT_PRIORITY[other.event_type]
        )


@dataclass
class RiskEvent:
    event_type: EventType = field(default=EventType.RISK, init=False)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    risk_type: RiskEventType = RiskEventType.MAX_DRAWDOWN_BREACH
    message: str = ""
    severity: str = "WARNING"  # WARNING, CRITICAL
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        return (self.timestamp, EVENT_PRIORITY[self.event_type]) < (
            other.timestamp, EVENT_PRIORITY[other.event_type]
        )


Event = BarEvent | SignalEvent | OrderEvent | FillEvent | RiskEvent
