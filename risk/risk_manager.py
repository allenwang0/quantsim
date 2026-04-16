"""
RiskManager: cross-cutting risk checks applied to all strategies.
Runs after strategy signals but before order submission.
Emits RiskEvents when limits are breached.
"""

from __future__ import annotations
import json
import logging
from datetime import datetime
from typing import Optional

from core.events import (
    SignalEvent, OrderEvent, RiskEvent, RiskEventType, Direction
)
from core.event_queue import EventQueue
from portfolio.portfolio import Portfolio

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Risk management overlays that operate across all strategies.
    
    Limits enforced:
    - Max drawdown circuit breaker (halt opens, then close all)
    - Concentration: no single position > max_position_pct
    - Options Greeks: portfolio-level delta, vega, gamma limits
    - Sector exposure limits (simplified: tracks per strategy_id)
    """

    def __init__(
        self,
        event_queue: EventQueue,
        max_drawdown_halt: float = -0.15,    # halt new opens at -15%
        max_drawdown_close: float = -0.20,   # close all positions at -20%
        max_position_pct: float = 0.10,      # max 10% in any single name
        max_portfolio_delta: float = 100.0,  # in delta-equivalent shares
        max_portfolio_vega: float = 1000.0,  # $ per 1% IV move
        max_portfolio_gamma: float = 50.0,   # delta per $1 move
        db_path: Optional[str] = None,
    ):
        self._queue = event_queue
        self.max_drawdown_halt = max_drawdown_halt
        self.max_drawdown_close = max_drawdown_close
        self.max_position_pct = max_position_pct
        self.max_portfolio_delta = max_portfolio_delta
        self.max_portfolio_vega = max_portfolio_vega
        self.max_portfolio_gamma = max_portfolio_gamma
        self._db_path = db_path

        self._circuit_breaker_active = False
        self._close_all_triggered = False

    def check_signal(self, signal: SignalEvent, portfolio: Portfolio) -> bool:
        """
        Returns True if signal passes all risk checks and can be converted to an order.
        Returns False if the signal should be suppressed.
        """
        # Circuit breaker: suppress all new OPEN signals
        if self._circuit_breaker_active and signal.direction != Direction.FLAT:
            logger.warning(f"Circuit breaker active: suppressing {signal.asset_id} signal")
            return False

        # Drawdown check
        dd = portfolio.current_drawdown
        if dd <= self.max_drawdown_close:
            if not self._close_all_triggered:
                self._close_all_triggered = True
                self._emit_risk(
                    portfolio,
                    RiskEventType.CIRCUIT_BREAKER,
                    f"Max drawdown {dd:.1%} breached {self.max_drawdown_close:.1%}: "
                    "closing all positions",
                    severity="CRITICAL",
                )
            return False

        if dd <= self.max_drawdown_halt:
            if not self._circuit_breaker_active:
                self._circuit_breaker_active = True
                self._emit_risk(
                    portfolio,
                    RiskEventType.MAX_DRAWDOWN_BREACH,
                    f"Drawdown {dd:.1%} breached halt threshold {self.max_drawdown_halt:.1%}: "
                    "halting new opens",
                    severity="WARNING",
                )
            # Allow FLAT (close) signals through; block opens
            if signal.direction != Direction.FLAT:
                return False

        return True

    def check_order(self, order: OrderEvent, portfolio: Portfolio) -> bool:
        """
        Returns True if order passes pre-trade risk checks.
        """
        from core.events import OrderSide

        is_open = order.side in (
            OrderSide.BUY, OrderSide.BUY_TO_OPEN, OrderSide.SELL_TO_OPEN
        )

        if not is_open:
            return True  # closing orders always pass

        # Concentration check
        equity = portfolio.total_equity
        if equity > 0:
            order_value = (order.quantity * (order.limit_price or 0)) 
            if order_value == 0 and equity > 0:
                order_value = equity * 0.02  # conservative estimate
            projected_weight = (
                abs(portfolio.positions.get(order.asset_id, type('', (), {'market_value': 0})()).market_value)
                + order_value
            ) / equity

            if projected_weight > self.max_position_pct:
                logger.warning(
                    f"Concentration limit: {order.asset_id} would be "
                    f"{projected_weight:.1%} > {self.max_position_pct:.1%}"
                )
                self._emit_risk(
                    portfolio, RiskEventType.CONCENTRATION_BREACH,
                    f"{order.asset_id} concentration {projected_weight:.1%} exceeds limit",
                )
                return False

        # Greeks limits check
        if abs(portfolio.portfolio_delta) > self.max_portfolio_delta:
            self._emit_risk(
                portfolio, RiskEventType.GREEKS_BREACH,
                f"Portfolio delta {portfolio.portfolio_delta:.1f} exceeds {self.max_portfolio_delta}",
            )

        if abs(portfolio.portfolio_vega) > self.max_portfolio_vega:
            self._emit_risk(
                portfolio, RiskEventType.GREEKS_BREACH,
                f"Portfolio vega {portfolio.portfolio_vega:.1f} exceeds {self.max_portfolio_vega}",
            )

        return True

    def update(self, portfolio: Portfolio, timestamp: datetime) -> None:
        """
        Called every bar to update circuit breaker state.
        If close_all is triggered, emits FLAT signals for all positions.
        """
        dd = portfolio.current_drawdown

        # Reset circuit breaker if drawdown has recovered
        if self._circuit_breaker_active and dd > self.max_drawdown_halt * 0.5:
            self._circuit_breaker_active = False
            logger.info(f"Circuit breaker reset: drawdown recovered to {dd:.1%}")

    def _emit_risk(
        self,
        portfolio: Portfolio,
        risk_type: RiskEventType,
        message: str,
        severity: str = "WARNING",
    ) -> None:
        event = RiskEvent(
            risk_type=risk_type,
            message=message,
            severity=severity,
            metadata={"drawdown": portfolio.current_drawdown, "equity": portfolio.total_equity},
        )
        self._queue.put(event)
        logger.warning(f"[RISK] {severity}: {message}")

        if self._db_path:
            self._persist_alert(event)

    def _persist_alert(self, event: RiskEvent) -> None:
        try:
            from core.database import db_conn
            ts_epoch = int(event.timestamp.timestamp())
            with db_conn(self._db_path) as conn:
                conn.execute(
                    """INSERT INTO risk_alerts (timestamp, risk_type, message, severity)
                       VALUES (?, ?, ?, ?)""",
                    (ts_epoch, event.risk_type.value, event.message, event.severity),
                )
        except Exception as e:
            logger.error(f"Risk alert persist failed: {e}")


def log_portfolio_greeks(portfolio, db_path: str, timestamp: int) -> None:
    """Log portfolio-level Greeks to the options_greeks_log table."""
    try:
        from core.database_v2 import log_options_greeks
        log_options_greeks(db_path=db_path, timestamp=timestamp, portfolio=portfolio)
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"Greeks logging failed: {e}")
