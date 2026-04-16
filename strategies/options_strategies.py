"""
Options strategies. All use reconstructed chains for backtesting.
Live paper trading uses Tradier chains.

Critical: model full bid-ask spread costs on every leg of every entry and exit.
For a 4-leg iron condor: 8 half-spreads total. This alone consumes the majority
of premium collected for illiquid underlyings.

Scope: liquid underlyings only (SPY, QQQ, IWM, GLD, TLT, top 20 equities).
"""

from __future__ import annotations
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from core.events import BarEvent, SignalEvent, Direction
from core.event_queue import EventQueue
from data.data_handler import DataHandler
from data.options import (
    reconstruct_chain, find_option_by_delta, compute_iv_rank,
    estimate_iv_from_history, get_expected_move,
)
from strategies.trend import Strategy

logger = logging.getLogger(__name__)


class OptionsStrategyBase(Strategy):
    """
    Base class for options strategies. Handles chain access (reconstructed or live).
    """

    def __init__(
        self,
        asset_id: str,
        event_queue: EventQueue,
        strategy_id: str,
        min_ivr: float = 30.0,
        target_dte_min: int = 21,
        target_dte_max: int = 45,
        exit_dte: int = 21,
        exit_profit_pct: float = 0.50,
    ):
        super().__init__(
            strategy_id=strategy_id,
            asset_ids=[asset_id],
            event_queue=event_queue,
            warmup_bars=252 + 30,
        )
        self.underlying = asset_id
        self.min_ivr = min_ivr
        self.target_dte_min = target_dte_min
        self.target_dte_max = target_dte_max
        self.exit_dte = exit_dte
        self.exit_profit_pct = exit_profit_pct
        self._open_positions: Dict[str, Dict] = {}  # option_symbol -> position info
        self._iv_history: List[float] = []

    def _get_chain(
        self,
        data_handler: DataHandler,
        as_of: datetime,
        bars: pd.DataFrame,
    ) -> List[Dict]:
        """Get reconstructed options chain for backtesting."""
        closes = bars["adj_close"] if "adj_close" in bars.columns else bars["close"]
        risk_free = data_handler.get_macro_value("DGS3MO") or 5.0
        risk_free /= 100.0  # convert from percentage

        iv_series = estimate_iv_from_history(closes)
        current_iv = float(iv_series.iloc[-1]) if not iv_series.empty else 0.20

        self._iv_history.append(current_iv)
        if len(self._iv_history) > 252:
            self._iv_history = self._iv_history[-252:]

        iv_series_hist = pd.Series(self._iv_history)
        ivr, iv_pct = compute_iv_rank(iv_series_hist, current_iv)

        as_of_date = as_of.date() if hasattr(as_of, 'date') else as_of

        chain = reconstruct_chain(
            underlying=self.underlying,
            as_of_date=as_of_date,
            prices_df=bars,
            risk_free_rate=risk_free,
            iv_series=pd.Series([current_iv], index=[pd.Timestamp(as_of_date)]),
            expirations_dte=[self.target_dte_min, self.target_dte_max, 60],
        )

        # Annotate chain with IVR
        for q in chain:
            q["ivr"] = ivr
            q["iv_percentile"] = iv_pct

        return chain, ivr, current_iv


class CoveredCallStrategy(OptionsStrategyBase):
    """
    Covered call: long 100 shares + short 0.30-delta call, 30-45 DTE.
    
    Entry condition: LOW_VOL regime (IVR > 30).
    Exit: close short call at 50% max profit or 21 DTE.
    
    Economics: collects premium in exchange for capped upside.
    Transaction cost: 1 leg open + 1 leg close = 2 half-spreads.
    """

    def __init__(
        self,
        asset_id: str,
        event_queue: EventQueue,
        short_delta: float = 0.30,
    ):
        super().__init__(
            asset_id=asset_id,
            event_queue=event_queue,
            strategy_id=f"CoveredCall_{asset_id}",
            min_ivr=30.0,
        )
        self.short_delta = short_delta
        self._has_equity = False
        self._active_short_call: Optional[Dict] = None
        self._premium_collected: float = 0.0

    def on_bar(self, event: BarEvent, data_handler: DataHandler) -> None:
        if event.asset_id != self.underlying:
            return

        self._tick(event.asset_id)
        if not self._is_warmed_up(event.asset_id):
            return

        bars = data_handler.get_latest_bars(self.underlying, n=252 + 30)
        if bars.empty:
            return

        # Always hold the underlying equity
        if not self._has_equity:
            self._has_equity = True
            self._emit_signal(
                self.underlying, Direction.LONG, event.timestamp,
                signal_type="covered_call_equity",
                metadata={"note": "underlying for covered call"},
            )
            return

        chain, ivr, current_iv = self._get_chain(data_handler, event.timestamp, bars)

        # Check if we should enter a new short call
        if self._active_short_call is None and ivr >= self.min_ivr:
            # Find option closest to target delta for target DTE
            target_exp_calls = [q for q in chain if q["right"] == "C"]
            target_call = find_option_by_delta(
                target_exp_calls, self.short_delta, "C"
            )

            if target_call and target_call["is_liquid"] and target_call["bid"] > 0.10:
                self._active_short_call = target_call
                self._premium_collected = target_call["bid"]  # sell at bid

                self._emit_signal(
                    self.underlying, Direction.SHORT, event.timestamp,
                    confidence=min(1.0, ivr),
                    signal_type="covered_call_open",
                    metadata={
                        "option_symbol": target_call.get("option_symbol", ""),
                        "strike": target_call["strike"],
                        "expiration": str(target_call.get("expiration", "")),
                        "delta": target_call["delta"],
                        "premium": target_call["bid"],
                        "ivr": ivr,
                    },
                )

        # Check exit conditions for active short call
        elif self._active_short_call is not None:
            call = self._active_short_call
            exp_date = call.get("expiration")
            if exp_date:
                if isinstance(exp_date, str):
                    exp_date = datetime.strptime(exp_date, "%Y-%m-%d").date()
                dte = (exp_date - event.timestamp.date()).days if hasattr(event.timestamp, 'date') else 30

                # Exit at 21 DTE or 50% profit
                current_value = call.get("last", call.get("ask", 0))
                profit_pct = (self._premium_collected - current_value) / self._premium_collected if self._premium_collected > 0 else 0

                if dte <= self.exit_dte or profit_pct >= self.exit_profit_pct:
                    self._active_short_call = None
                    self._emit_signal(
                        self.underlying, Direction.FLAT, event.timestamp,
                        signal_type="covered_call_close",
                        metadata={"profit_pct": profit_pct, "dte": dte},
                    )


class IronCondorStrategy(OptionsStrategyBase):
    """
    Iron condor: short OTM put spread + short OTM call spread.
    
    Short strikes at 0.16-delta (~1 standard deviation).
    Entry: LOW_VOL regime, IVR > 50.
    
    Exit rules:
    1. Close at 50% max profit
    2. Close at 200% loss on one spread (stop loss)
    3. Close at 21 DTE (gamma risk increases sharply)
    
    CRITICAL TRANSACTION COST WARNING:
    4 legs * open + 4 legs * close = 8 half-spreads.
    For options with spread = 20% of mid, this consumes 80% of premium.
    Only backtest on liquid underlyings (SPY, QQQ, IWM).
    """

    def __init__(
        self,
        asset_id: str,
        event_queue: EventQueue,
        short_delta: float = 0.16,
        wing_width_pct: float = 0.03,
        min_ivr: float = 50.0,
    ):
        super().__init__(
            asset_id=asset_id,
            event_queue=event_queue,
            strategy_id=f"IronCondor_{asset_id}",
            min_ivr=min_ivr,
        )
        self.short_delta = short_delta
        self.wing_width_pct = wing_width_pct
        self._active_condor: Optional[Dict] = None
        self._net_credit: float = 0.0
        self._max_loss_per_spread: float = 0.0

    def on_bar(self, event: BarEvent, data_handler: DataHandler) -> None:
        if event.asset_id != self.underlying:
            return

        self._tick(event.asset_id)
        if not self._is_warmed_up(event.asset_id):
            return

        bars = data_handler.get_latest_bars(self.underlying, n=252 + 30)
        if bars.empty:
            return

        chain, ivr, current_iv = self._get_chain(data_handler, event.timestamp, bars)

        if self._active_condor is None:
            # Entry: LOW_VOL regime, IVR > 50
            if ivr < self.min_ivr:
                return

            current_close = float(event.close)

            # Find short strikes at 0.16-delta
            short_call = find_option_by_delta(chain, self.short_delta, "C")
            short_put = find_option_by_delta(chain, -self.short_delta, "P")

            if not short_call or not short_put:
                return

            # Long strikes as wings (further OTM)
            call_wing_strike = short_call["strike"] * (1 + self.wing_width_pct)
            put_wing_strike = short_put["strike"] * (1 - self.wing_width_pct)

            # Find closest wing options
            call_wings = [q for q in chain if q["right"] == "C" and q["strike"] >= call_wing_strike]
            put_wings = [q for q in chain if q["right"] == "P" and q["strike"] <= put_wing_strike]

            if not call_wings or not put_wings:
                return

            long_call = min(call_wings, key=lambda q: abs(q["strike"] - call_wing_strike))
            long_put = max(put_wings, key=lambda q: abs(q["strike"] - put_wing_strike))

            # Calculate net credit (sell short options, buy long options)
            # Model bid-ask spread costs: pay half-spread on each of 4 legs
            sc_credit = short_call["bid"]  # sell at bid
            sp_credit = short_put["bid"]
            lc_cost = long_call["ask"]     # buy at ask
            lp_cost = long_put["ask"]

            net_credit = sc_credit + sp_credit - lc_cost - lp_cost

            if net_credit <= 0.10:  # minimum credit threshold
                return

            spread_width = min(
                abs(long_call["strike"] - short_call["strike"]),
                abs(short_put["strike"] - long_put["strike"]),
            )
            max_loss = spread_width - net_credit

            self._active_condor = {
                "short_call": short_call,
                "long_call": long_call,
                "short_put": short_put,
                "long_put": long_put,
                "net_credit": net_credit,
                "max_loss": max_loss,
                "entry_date": event.timestamp,
                "expiration": short_call.get("expiration"),
            }
            self._net_credit = net_credit
            self._max_loss_per_spread = max_loss

            self._emit_signal(
                self.underlying, Direction.FLAT, event.timestamp,
                signal_type="iron_condor_open",
                metadata={
                    "net_credit": net_credit,
                    "max_loss": max_loss,
                    "short_call_strike": short_call["strike"],
                    "short_put_strike": short_put["strike"],
                    "ivr": ivr,
                    "warning": "8 half-spreads on open+close; model transaction costs explicitly",
                },
            )

        else:
            # Check exit conditions
            condor = self._active_condor
            exp_date = condor.get("expiration")
            dte = 30  # default

            if exp_date:
                if isinstance(exp_date, str):
                    try:
                        exp_date_obj = datetime.strptime(exp_date, "%Y-%m-%d").date()
                        curr_date = event.timestamp.date() if hasattr(event.timestamp, 'date') else event.timestamp
                        dte = (exp_date_obj - curr_date).days
                    except Exception:
                        pass

            # Approximate current condor value from current IV vs entry IV
            # In a full implementation, re-price all 4 legs from current chain
            approx_current_value = self._net_credit * (dte / 45.0)
            profit_pct = (self._net_credit - approx_current_value) / self._net_credit if self._net_credit > 0 else 0

            should_close = (
                profit_pct >= self.exit_profit_pct or  # 50% profit target
                dte <= self.exit_dte or                  # 21 DTE rule
                profit_pct <= -2.0                       # 200% loss stop
            )

            if should_close:
                self._active_condor = None
                self._emit_signal(
                    self.underlying, Direction.FLAT, event.timestamp,
                    signal_type="iron_condor_close",
                    metadata={"profit_pct": profit_pct, "dte": dte},
                )


class LongStraddleStrategy(OptionsStrategyBase):
    """
    Long straddle: long ATM call + long ATM put.
    
    Entry: HIGH_VOL regime (VIX > 25) or pre-earnings when actual move
    likely to exceed implied move. Long gamma, negative theta.
    
    Exit: close at 25-50% loss (theta decay) or 2x profit.
    
    Expected move = ATM_call_price + ATM_put_price.
    Enter only when you believe actual move > expected move.
    """

    def __init__(
        self,
        asset_id: str,
        event_queue: EventQueue,
        min_dte: int = 30,
        max_loss_pct: float = 0.40,
        profit_target_mult: float = 2.0,
    ):
        super().__init__(
            asset_id=asset_id,
            event_queue=event_queue,
            strategy_id=f"LongStraddle_{asset_id}",
            min_ivr=0.0,  # long vol strategy: low IVR is better entry
        )
        self.min_dte = min_dte
        self.max_loss_pct = max_loss_pct
        self.profit_target = profit_target_mult
        self._active_straddle: Optional[Dict] = None
        self._entry_cost: float = 0.0

    def on_bar(self, event: BarEvent, data_handler: DataHandler) -> None:
        if event.asset_id != self.underlying:
            return

        self._tick(event.asset_id)
        if not self._is_warmed_up(event.asset_id):
            return

        bars = data_handler.get_latest_bars(self.underlying, n=252 + 30)
        if bars.empty:
            return

        # Get VIX for regime check
        vix = data_handler.get_macro_value("VIXCLS")
        is_high_vol = vix and vix > 25

        chain, ivr, current_iv = self._get_chain(data_handler, event.timestamp, bars)

        if self._active_straddle is None and is_high_vol:
            # Find ATM call and put
            current_close = float(event.close)
            atm_call = find_option_by_delta(chain, 0.50, "C")
            atm_put = find_option_by_delta(chain, -0.50, "P")

            if not atm_call or not atm_put:
                return

            expected_move = get_expected_move(atm_call, atm_put)
            entry_cost = atm_call["ask"] + atm_put["ask"]  # buy at ask

            self._active_straddle = {
                "call": atm_call,
                "put": atm_put,
                "entry_cost": entry_cost,
                "expected_move": expected_move,
                "entry_date": event.timestamp,
            }
            self._entry_cost = entry_cost

            self._emit_signal(
                self.underlying, Direction.LONG, event.timestamp,
                signal_type="long_straddle_open",
                metadata={
                    "expected_move": expected_move,
                    "entry_cost": entry_cost,
                    "vix": vix,
                    "ivr": ivr,
                },
            )

        elif self._active_straddle is not None:
            # Simplified P&L: approximate based on underlying move from entry
            if bars.empty or len(bars) == 0:
                return

            closes = bars["adj_close"] if "adj_close" in bars.columns else bars["close"]
            entry_price = float(self._active_straddle.get("call", {}).get("last", closes.iloc[-1]))
            current_price = closes.iloc[-1]
            price_move = abs(current_price - entry_price)

            # Approximate straddle value = intrinsic + remaining time value
            approx_value = max(price_move - self._entry_cost * 0.3, self._entry_cost * 0.1)

            pnl_pct = (approx_value - self._entry_cost) / self._entry_cost if self._entry_cost > 0 else 0

            if pnl_pct <= -self.max_loss_pct or pnl_pct >= (self.profit_target - 1):
                self._active_straddle = None
                self._emit_signal(
                    self.underlying, Direction.FLAT, event.timestamp,
                    signal_type="long_straddle_close",
                    metadata={"pnl_pct": float(pnl_pct)},
                )
