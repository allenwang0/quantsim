"""
Execution handlers: simulate fills for backtesting, route to Tradier for paper trading.

Four slippage models, four fill models, options-specific bid-ask costs.
The BacktestExecutionHandler is the ONLY module that differs from paper trading.
"""

from __future__ import annotations
import logging
import random
import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict
import numpy as np

from core.events import (
    OrderEvent, FillEvent, OrderSide, OrderType, AssetType, EventType
)
from core.event_queue import EventQueue

logger = logging.getLogger(__name__)


# ── Slippage Models ────────────────────────────────────────────────────────────

class SlippageModel(ABC):
    @abstractmethod
    def compute_slippage(
        self,
        price: float,
        quantity: int,
        bar_volume: int,
        adv_20: float,
        asset_type: AssetType,
    ) -> float:
        """Returns slippage in dollars per share (always positive)."""
        ...


class NoSlippage(SlippageModel):
    """Model 1: Fill at close price. Unrealistic ceiling."""
    def compute_slippage(self, price, quantity, bar_volume, adv_20, asset_type):
        return 0.0


class FixedSpreadSlippage(SlippageModel):
    """Model 2: Fixed spread in basis points. Simple parametric model."""

    def __init__(self, spread_bps_large: float = 5.0, spread_bps_small: float = 20.0,
                 large_cap_threshold_adv: float = 1_000_000):
        self.spread_bps_large = spread_bps_large
        self.spread_bps_small = spread_bps_small
        self.threshold = large_cap_threshold_adv

    def compute_slippage(self, price, quantity, bar_volume, adv_20, asset_type):
        bps = self.spread_bps_large if adv_20 > self.threshold else self.spread_bps_small
        return price * bps / 10_000


class VolumeProportionalSlippage(SlippageModel):
    """
    Model 3 (RECOMMENDED DEFAULT): Almgren-Chriss inspired market impact.
    slippage = price * k * (order_size / ADV)^0.5
    
    Orders < 1% of ADV: roughly bid-ask spread only.
    Orders > 10% of ADV: significant price impact.
    """

    def __init__(self, k: float = 0.1):
        self.k = k

    def compute_slippage(self, price, quantity, bar_volume, adv_20, asset_type):
        if adv_20 <= 0:
            adv_20 = max(bar_volume, 1)
        participation = quantity / adv_20
        return price * self.k * np.sqrt(participation)


class OptionsBidAskSlippage(SlippageModel):
    """
    Options-specific: half the bid-ask spread per leg.
    For illiquid options, this alone can consume most of the premium collected.
    """

    def __init__(self, spread_pct_liquid: float = 0.02, spread_pct_illiquid: float = 0.20):
        self.spread_liquid = spread_pct_liquid
        self.spread_illiquid = spread_pct_illiquid

    def compute_slippage(self, price, quantity, bar_volume, adv_20, asset_type):
        is_liquid = adv_20 > 100 and bar_volume > 10
        spread_pct = self.spread_liquid if is_liquid else self.spread_illiquid
        return price * spread_pct / 2  # half-spread per leg


# ── Commission Models ──────────────────────────────────────────────────────────

class CommissionModel(ABC):
    @abstractmethod
    def compute_commission(self, quantity: int, fill_price: float, asset_type: AssetType) -> float:
        ...


class ZeroCommission(CommissionModel):
    """IBKR Lite / Schwab / Fidelity for equities."""
    def compute_commission(self, quantity, fill_price, asset_type):
        return 0.0


class PerShareCommission(CommissionModel):
    """IBKR Pro default: $0.005/share, $1.00 minimum."""
    def __init__(self, rate: float = 0.005, minimum: float = 1.0):
        self.rate = rate
        self.minimum = minimum

    def compute_commission(self, quantity, fill_price, asset_type):
        if asset_type == AssetType.OPTION:
            return max(self.minimum, quantity * 0.65)  # IBKR options
        return max(self.minimum, quantity * self.rate)


class PerContractOptionsCommission(CommissionModel):
    """Standard options commissions: $0.65/contract."""
    def __init__(self, rate: float = 0.65, minimum: float = 1.00):
        self.rate = rate
        self.minimum = minimum

    def compute_commission(self, quantity, fill_price, asset_type):
        if asset_type == AssetType.OPTION:
            return max(self.minimum, quantity * self.rate)
        return 0.0  # equities free


# ── Fill Models ────────────────────────────────────────────────────────────────

class FillModel:
    """
    Determines the fill price for a given order and bar data.
    Default: NEXT_BAR_OPEN (correct for daily-bar strategies).
    """

    IMMEDIATE_CLOSE = "IMMEDIATE_CLOSE"   # fills at close (unrealistic baseline)
    NEXT_BAR_OPEN = "NEXT_BAR_OPEN"      # correct default for daily strategies
    VWAP = "VWAP"                         # (o + h + l + c) / 4
    PARTIAL = "PARTIAL"                   # probabilistic partial fills

    def __init__(self, mode: str = NEXT_BAR_OPEN, partial_threshold: float = 0.10):
        self.mode = mode
        self.partial_threshold = partial_threshold

    def compute_fill(
        self,
        order: OrderEvent,
        current_bar: Optional[Dict],
        next_bar: Optional[Dict],
        adv_20: float = 1_000_000,
    ) -> Optional[Dict]:
        """
        Returns fill dict: {price, quantity, partial} or None if no fill.
        """
        if not next_bar and self.mode != self.IMMEDIATE_CLOSE:
            return None
        if not current_bar and self.mode == self.IMMEDIATE_CLOSE:
            return None

        if self.mode == self.IMMEDIATE_CLOSE:
            bar = current_bar
            price = bar.get("adj_close", bar.get("close", 0))
        elif self.mode == self.NEXT_BAR_OPEN:
            bar = next_bar
            price = bar.get("open", bar.get("adj_open", 0))
        elif self.mode == self.VWAP:
            bar = next_bar
            o, h, l, c = (bar.get(k, 0) for k in ("open", "high", "low", "close"))
            price = (o + h + l + c) / 4
        else:
            bar = next_bar
            price = bar.get("open", 0)

        # Limit order logic: buy fills only when low <= limit; sell when high >= limit
        if order.order_type == OrderType.LIMIT and order.limit_price:
            if order.side in (OrderSide.BUY, OrderSide.BUY_TO_OPEN, OrderSide.BUY_TO_CLOSE):
                if bar.get("low", 0) > order.limit_price:
                    return None  # price never reached limit
                price = min(price, order.limit_price)
            else:
                if bar.get("high", 0) < order.limit_price:
                    return None
                price = max(price, order.limit_price)

        # Partial fill model
        quantity = order.quantity
        if self.mode == self.PARTIAL and adv_20 > 0:
            participation = quantity / adv_20
            if participation > self.partial_threshold:
                fill_fraction = random.uniform(0.3, 1.0)
                quantity = max(1, int(quantity * fill_fraction))

        return {"price": price, "quantity": quantity, "partial": quantity < order.quantity}


# ── Execution Handlers ─────────────────────────────────────────────────────────

class ExecutionHandler(ABC):
    @abstractmethod
    def execute_order(self, order: OrderEvent) -> Optional[FillEvent]:
        ...


class BacktestExecutionHandler(ExecutionHandler):
    """
    Simulates fills for backtesting. Immediately computes FillEvent.
    
    This is the ONLY module that differs from paper trading.
    Everything else is identical between modes.
    """

    def __init__(
        self,
        data_handler,  # HistoricalDataHandler
        event_queue: EventQueue,
        slippage_model: SlippageModel = None,
        commission_model: CommissionModel = None,
        fill_model: FillModel = None,
    ):
        self._data = data_handler
        self._queue = event_queue
        self._slippage = slippage_model or VolumeProportionalSlippage()
        self._commission = commission_model or ZeroCommission()
        self._fill_model = fill_model or FillModel(FillModel.NEXT_BAR_OPEN)

        # ADV cache: updated when we process bars
        self._adv_cache: Dict[str, float] = {}

    def execute_order(self, order: OrderEvent) -> Optional[FillEvent]:
        """
        Given an OrderEvent, simulate a fill using the current bar data.
        Next-bar-open fill by default: signal at T executes at T+1 open.
        """
        asset_id = order.asset_id

        # Current bar (for immediate-close fills)
        current_bar = self._data.get_current_bar(asset_id)
        current_bar_dict = None
        if current_bar is not None:
            current_bar_dict = {
                "open": float(current_bar.get("adj_open", current_bar.get("open", 0))),
                "high": float(current_bar.get("adj_high", current_bar.get("high", 0))),
                "low": float(current_bar.get("adj_low", current_bar.get("low", 0))),
                "close": float(current_bar.get("adj_close", current_bar.get("close", 0))),
                "volume": int(current_bar.get("volume", 0)),
            }

        # For next-bar-open, we need the next bar
        # In the event-driven model, we schedule the fill for the NEXT bar
        # For simplicity in the backtester, we use the current bar's open as T+1 open
        # (this is valid because we're always one step behind in the event loop)
        if current_bar_dict is None:
            return None

        # Get 20-day ADV for slippage model
        adv = self._adv_cache.get(asset_id, 0)
        if adv == 0:
            recent = self._data.get_latest_bars(asset_id, n=20)
            if not recent.empty and "volume" in recent.columns:
                adv = float(recent["volume"].mean())
            adv = max(adv, 1)
            self._adv_cache[asset_id] = adv

        fill_result = self._fill_model.compute_fill(
            order=order,
            current_bar=current_bar_dict,
            next_bar=current_bar_dict,  # same bar for daily backtesting
            adv_20=adv,
        )

        if fill_result is None:
            return None

        price = fill_result["price"]
        quantity = fill_result["quantity"]

        if price <= 0:
            return None

        # Compute slippage
        slippage = self._slippage.compute_slippage(
            price=price,
            quantity=quantity,
            bar_volume=current_bar_dict.get("volume", 0),
            adv_20=adv,
            asset_type=order.asset_type,
        )

        # Apply slippage direction: buys pay more, sells receive less
        is_buy = order.side in (OrderSide.BUY, OrderSide.BUY_TO_OPEN, OrderSide.BUY_TO_CLOSE)
        adjusted_price = price + slippage if is_buy else price - slippage
        adjusted_price = max(0.01, adjusted_price)

        # Options: apply bid-ask spread cost (half-spread per leg)
        if order.asset_type == AssetType.OPTION:
            options_slippage = OptionsBidAskSlippage()
            ba_cost = options_slippage.compute_slippage(adjusted_price, quantity, 0, adv, order.asset_type)
            adjusted_price = adjusted_price + ba_cost if is_buy else adjusted_price - ba_cost

        commission = self._commission.compute_commission(quantity, adjusted_price, order.asset_type)

        fill = FillEvent(
            order_id=order.order_id,
            asset_id=order.asset_id,
            asset_type=order.asset_type,
            side=order.side,
            quantity=quantity,
            fill_price=adjusted_price,
            commission=commission,
            slippage=slippage,
            strategy_id=order.strategy_id,
            option_symbol=order.option_symbol,
            expiration=order.expiration,
            strike=order.strike,
            right=order.right,
            timestamp=order.timestamp,
        )

        self._queue.put(fill)
        return fill
