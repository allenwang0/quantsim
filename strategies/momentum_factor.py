"""
Additional strategies: cross-sectional momentum, volatility targeting,
GARCH forecasting, factor-based (value, low-vol), dual momentum, regime detection.
"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from core.events import BarEvent, SignalEvent, Direction
from core.event_queue import EventQueue
from data.data_handler import DataHandler
from strategies.trend import Strategy

logger = logging.getLogger(__name__)


class CrossSectionalMomentum(Strategy):
    """
    Jegadeesh-Titman cross-sectional momentum.
    
    Rank universe by 12-1 month return. Long top quintile, flat (or short) bottom.
    Rebalance monthly.
    
    Transaction cost warning: high turnover destroys most of the gross alpha.
    Published Sharpe 0.8-1.2 → 0.3-0.5 after realistic transaction costs.
    Reference: Jegadeesh and Titman (1993).
    """

    def __init__(
        self,
        asset_ids: List[str],
        event_queue: EventQueue,
        lookback_months: int = 12,
        skip_months: int = 1,
        top_n: int = 5,
        bottom_n: int = 5,
        long_only: bool = True,
    ):
        super().__init__(
            strategy_id="XS_Momentum",
            asset_ids=asset_ids,
            event_queue=event_queue,
            warmup_bars=(lookback_months + skip_months) * 23,
        )
        self.lookback_bars = lookback_months * 21
        self.skip_bars = skip_months * 21
        self.top_n = top_n
        self.bottom_n = bottom_n
        self.long_only = long_only
        self._last_rebalance_month: int = -1
        self._current_longs: List[str] = []
        self._current_shorts: List[str] = []

    def on_bar(self, event: BarEvent, data_handler: DataHandler) -> None:
        asset_id = event.asset_id
        if asset_id != self.asset_ids[0]:  # Process once per bar (use first asset as trigger)
            return

        self._tick(asset_id)
        if not self._is_warmed_up(asset_id):
            return

        # Monthly rebalance only
        current_month = event.timestamp.month
        if self._last_rebalance_month == current_month:
            return
        self._last_rebalance_month = current_month

        # Rank all assets by 12-1 month return
        returns: Dict[str, float] = {}
        total_needed = self.lookback_bars + self.skip_bars + 5

        for aid in self.asset_ids:
            bars = data_handler.get_latest_bars(aid, n=total_needed)
            if len(bars) < self.lookback_bars + self.skip_bars:
                continue

            closes = bars["adj_close"] if "adj_close" in bars.columns else bars["close"]
            price_start = closes.iloc[-(self.lookback_bars + self.skip_bars)]
            price_end = closes.iloc[-self.skip_bars] if self.skip_bars > 0 else closes.iloc[-1]

            if price_start > 0:
                returns[aid] = (price_end / price_start) - 1.0

        if len(returns) < self.top_n + self.bottom_n:
            return

        sorted_assets = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        new_longs = [a for a, _ in sorted_assets[:self.top_n]]
        new_shorts = [a for a, _ in sorted_assets[-self.bottom_n:]] if not self.long_only else []

        # Close positions that are no longer in the portfolio
        for asset in self._current_longs:
            if asset not in new_longs:
                self._emit_signal(asset, Direction.FLAT, event.timestamp, signal_type="xsmom_exit")

        for asset in self._current_shorts:
            if asset not in new_shorts:
                self._emit_signal(asset, Direction.FLAT, event.timestamp, signal_type="xsmom_exit")

        # Open new positions
        for asset in new_longs:
            if asset not in self._current_longs:
                self._emit_signal(
                    asset, Direction.LONG, event.timestamp,
                    confidence=0.8,
                    signal_type="xsmom",
                    holding_period=21,
                    metadata={"momentum_return": returns.get(asset, 0)},
                )

        for asset in new_shorts:
            if asset not in self._current_shorts:
                self._emit_signal(
                    asset, Direction.SHORT, event.timestamp,
                    confidence=0.8,
                    signal_type="xsmom",
                    holding_period=21,
                    metadata={"momentum_return": returns.get(asset, 0)},
                )

        self._current_longs = new_longs
        self._current_shorts = new_shorts


class VolatilityTargetingStrategy(Strategy):
    """
    Scale position size so portfolio volatility = target_vol.
    This is a sizing overlay, not a direction strategy.
    
    target_leverage = target_annual_vol / realized_30day_vol
    Leverage capped at [0.25, 2.0].
    
    Known failure mode: leverage increases before a volatility regime change.
    """

    def __init__(
        self,
        asset_ids: List[str],
        event_queue: EventQueue,
        base_strategy: Strategy,
        target_annual_vol: float = 0.10,
        vol_window: int = 30,
        max_leverage: float = 2.0,
        min_leverage: float = 0.25,
    ):
        super().__init__(
            strategy_id=f"VolTarget_{base_strategy.strategy_id}",
            asset_ids=asset_ids,
            event_queue=event_queue,
            warmup_bars=vol_window + base_strategy.warmup_bars,
        )
        self.base_strategy = base_strategy
        self.target_vol = target_annual_vol
        self.vol_window = vol_window
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage

    def on_bar(self, event: BarEvent, data_handler: DataHandler) -> None:
        # Delegate to base strategy; size confidence proportional to vol-targeting leverage
        asset_id = event.asset_id
        self._tick(asset_id)

        bars = data_handler.get_latest_bars(asset_id, n=self.vol_window + 5)
        if not bars.empty and len(bars) >= self.vol_window:
            closes = bars["adj_close"] if "adj_close" in bars.columns else bars["close"]
            log_rets = np.log(closes / closes.shift(1)).dropna()
            realized_vol = float(log_rets.iloc[-self.vol_window:].std() * np.sqrt(252))

            if realized_vol > 0:
                leverage = np.clip(
                    self.target_vol / realized_vol,
                    self.min_leverage,
                    self.max_leverage,
                )
                # Adjust base strategy confidence by leverage
                # (implementation would intercept base strategy signals)

        self.base_strategy.on_bar(event, data_handler)


class VIXRegimeFilter(Strategy):
    """
    VIX-based regime filter applied as an overlay to all strategies.
    
    VIX_20day_avg > 25: reduce all position sizes by 50% (set confidence = 0.5)
    VIX_20day_avg > 35: close all positions (FLAT signal for all held positions)
    
    This is not a standalone strategy; it is a risk overlay.
    """

    HALT_THRESHOLD = 25.0
    CLOSE_THRESHOLD = 35.0

    def __init__(self, event_queue: EventQueue, asset_ids: List[str]):
        super().__init__(
            strategy_id="VIX_Regime",
            asset_ids=asset_ids,
            event_queue=event_queue,
            warmup_bars=20,
        )
        self._vix_history: List[float] = []
        self._regime: str = "LOW_VOL"

    def current_regime(self) -> str:
        return self._regime

    def get_confidence_multiplier(self) -> float:
        if self._regime == "HIGH_VOL_EXTREME":
            return 0.0
        elif self._regime == "HIGH_VOL":
            return 0.5
        return 1.0

    def on_bar(self, event: BarEvent, data_handler: DataHandler) -> None:
        vix = data_handler.get_macro_value("VIXCLS")
        if vix is None:
            return

        self._vix_history.append(vix)
        if len(self._vix_history) > 30:
            self._vix_history = self._vix_history[-30:]

        vix_20d_avg = np.mean(self._vix_history[-20:]) if len(self._vix_history) >= 20 else vix

        old_regime = self._regime
        if vix_20d_avg > self.CLOSE_THRESHOLD:
            self._regime = "HIGH_VOL_EXTREME"
        elif vix_20d_avg > self.HALT_THRESHOLD:
            self._regime = "HIGH_VOL"
        else:
            self._regime = "LOW_VOL"

        if self._regime == "HIGH_VOL_EXTREME" and old_regime != "HIGH_VOL_EXTREME":
            # Emit FLAT for all assets
            for asset_id in self.asset_ids:
                self._emit_signal(
                    asset_id, Direction.FLAT, event.timestamp,
                    signal_type="vix_regime_close",
                    metadata={"vix_20d": float(vix_20d_avg), "regime": self._regime},
                )


class DualMomentum(Strategy):
    """
    Antonacci (2014) Dual Momentum.
    
    Absolute momentum: if SPY 12-month return > T-bill rate → equity; else bonds.
    Relative momentum: choose between SPY and EFA (higher 12-month return).
    Combined: apply relative first, then absolute.
    Monthly rebalance.
    
    Reference: Antonacci (2014) "Dual Momentum Investing"
    """

    def __init__(
        self,
        event_queue: EventQueue,
        equity_assets: List[str] = None,
        bond_asset: str = "AGG",
        tbill_asset: str = "BIL",
    ):
        assets = equity_assets or ["SPY", "EFA"]
        all_assets = assets + [bond_asset, tbill_asset]
        super().__init__(
            strategy_id="DualMomentum",
            asset_ids=all_assets,
            event_queue=event_queue,
            warmup_bars=260,
        )
        self.equity_assets = assets
        self.bond_asset = bond_asset
        self.tbill_asset = tbill_asset
        self._last_rebalance_month: int = -1
        self._current_holding: str = ""

    def on_bar(self, event: BarEvent, data_handler: DataHandler) -> None:
        asset_id = event.asset_id
        if asset_id != self.asset_ids[0]:
            return

        self._tick(asset_id)
        if not self._is_warmed_up(asset_id):
            return

        current_month = event.timestamp.month
        if self._last_rebalance_month == current_month:
            return
        self._last_rebalance_month = current_month

        # Compute 12-month returns for all assets
        lookback = 252
        returns_12m: Dict[str, float] = {}
        for aid in self.equity_assets + [self.tbill_asset]:
            bars = data_handler.get_latest_bars(aid, n=lookback + 5)
            if len(bars) < lookback:
                continue
            closes = bars["adj_close"] if "adj_close" in bars.columns else bars["close"]
            ret = (closes.iloc[-1] / closes.iloc[-lookback]) - 1.0
            returns_12m[aid] = ret

        if not returns_12m:
            return

        # Step 1: Relative momentum - best performing equity asset
        equity_returns = {a: returns_12m.get(a, -99) for a in self.equity_assets}
        if equity_returns:
            best_equity = max(equity_returns, key=equity_returns.get)
            best_equity_return = equity_returns[best_equity]
        else:
            return

        # Step 2: Absolute momentum - compare best equity to T-bill rate
        tbill_return = returns_12m.get(self.tbill_asset, 0.0)
        new_holding = best_equity if best_equity_return > tbill_return else self.bond_asset

        if new_holding != self._current_holding:
            # Exit old holding
            if self._current_holding:
                self._emit_signal(
                    self._current_holding, Direction.FLAT, event.timestamp,
                    signal_type="dual_mom_exit",
                )

            # Enter new holding
            self._current_holding = new_holding
            self._emit_signal(
                new_holding, Direction.LONG, event.timestamp,
                signal_type="dual_momentum",
                holding_period=21,
                metadata={"equity_return": float(best_equity_return), "tbill": float(tbill_return)},
            )


class LowVolatilityFactor(Strategy):
    """
    Low volatility factor: long lowest-vol quintile of universe.
    Baker, Bradley, Wurgler (2011): low-vol stocks persistently outperform.
    Monthly rebalance.
    """

    def __init__(
        self,
        asset_ids: List[str],
        event_queue: EventQueue,
        vol_window: int = 252,
        top_n: int = 5,
    ):
        super().__init__(
            strategy_id="LowVol_Factor",
            asset_ids=asset_ids,
            event_queue=event_queue,
            warmup_bars=vol_window + 10,
        )
        self.vol_window = vol_window
        self.top_n = top_n
        self._last_rebalance_month: int = -1
        self._current_holdings: List[str] = []

    def on_bar(self, event: BarEvent, data_handler: DataHandler) -> None:
        asset_id = event.asset_id
        if asset_id != self.asset_ids[0]:
            return

        self._tick(asset_id)
        if not self._is_warmed_up(asset_id):
            return

        current_month = event.timestamp.month
        if self._last_rebalance_month == current_month:
            return
        self._last_rebalance_month = current_month

        vols: Dict[str, float] = {}
        for aid in self.asset_ids:
            bars = data_handler.get_latest_bars(aid, n=self.vol_window + 5)
            if len(bars) < self.vol_window // 2:
                continue
            closes = bars["adj_close"] if "adj_close" in bars.columns else bars["close"]
            log_rets = np.log(closes / closes.shift(1)).dropna()
            vol = float(log_rets.iloc[-self.vol_window:].std() * np.sqrt(252))
            if vol > 0:
                vols[aid] = vol

        if len(vols) < self.top_n:
            return

        sorted_assets = sorted(vols.items(), key=lambda x: x[1])
        new_holdings = [a for a, _ in sorted_assets[:self.top_n]]

        for asset in self._current_holdings:
            if asset not in new_holdings:
                self._emit_signal(asset, Direction.FLAT, event.timestamp, signal_type="lowvol_exit")

        for asset in new_holdings:
            if asset not in self._current_holdings:
                self._emit_signal(
                    asset, Direction.LONG, event.timestamp,
                    confidence=0.8,
                    signal_type="low_vol_factor",
                    holding_period=21,
                    metadata={"vol": vols.get(asset, 0)},
                )

        self._current_holdings = new_holdings


class BuyAndHold(Strategy):
    """
    Buy and hold strategy. Used for the walking skeleton validation.
    Buys all assets at start and holds.
    Validation target: SPY total return 2020-2023 within 0.1% annually.
    """

    def __init__(
        self,
        asset_ids: List[str],
        event_queue: EventQueue,
    ):
        super().__init__(
            strategy_id="BuyAndHold",
            asset_ids=asset_ids,
            event_queue=event_queue,
            warmup_bars=0,
        )
        self._bought: set = set()

    def on_bar(self, event: BarEvent, data_handler: DataHandler) -> None:
        asset_id = event.asset_id
        if asset_id in self._bought:
            return
        if asset_id not in self.asset_ids:
            return

        self._bought.add(asset_id)
        self._emit_signal(
            asset_id, Direction.LONG, event.timestamp,
            signal_type="buy_and_hold",
            holding_period=99999,
        )
