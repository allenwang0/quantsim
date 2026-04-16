"""
Strategy base class and trend-following strategy implementations.

All strategies:
- Receive BarEvents via on_bar()
- Emit SignalEvents via the event queue
- Are stateless except for indicator state (reconstructable from bar history)
- Never access future data

Trend strategies: SMA crossover, EMA crossover, MACD, Donchian, ADX-filtered, TSMOM
"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict
import numpy as np
import pandas as pd

from core.events import BarEvent, SignalEvent, Direction
from core.event_queue import EventQueue
from data.data_handler import DataHandler

logger = logging.getLogger(__name__)


class Strategy(ABC):
    """
    Abstract base class for all strategies.
    
    Lifecycle:
      on_bar(event, data_handler) → emits SignalEvent(s) if signal conditions met
    
    Invariant: only call data_handler.get_latest_bars() which enforces T-1 data access.
    Never use the current bar's price for signal generation.
    """

    def __init__(
        self,
        strategy_id: str,
        asset_ids: List[str],
        event_queue: EventQueue,
        warmup_bars: int = 252,
    ):
        self.strategy_id = strategy_id
        self.asset_ids = asset_ids
        self._queue = event_queue
        self.warmup_bars = warmup_bars
        self._bar_count: Dict[str, int] = {a: 0 for a in asset_ids}

    @abstractmethod
    def on_bar(self, event: BarEvent, data_handler: DataHandler) -> None:
        """Process a new bar event and potentially emit a signal."""
        ...

    def _emit_signal(
        self,
        asset_id: str,
        direction: Direction,
        timestamp: datetime,
        confidence: float = 1.0,
        signal_type: str = "",
        holding_period: int = 1,
        metadata: Optional[Dict] = None,
    ) -> None:
        signal = SignalEvent(
            timestamp=timestamp,
            strategy_id=self.strategy_id,
            asset_id=asset_id,
            direction=direction,
            confidence=confidence,
            signal_type=signal_type or self.__class__.__name__,
            holding_period_estimate=holding_period,
            metadata=metadata or {},
        )
        self._queue.put(signal)

    def _is_warmed_up(self, asset_id: str) -> bool:
        return self._bar_count.get(asset_id, 0) >= self.warmup_bars

    def _tick(self, asset_id: str) -> None:
        self._bar_count[asset_id] = self._bar_count.get(asset_id, 0) + 1


# ── Trend Following ────────────────────────────────────────────────────────────

class SMAcrossover(Strategy):
    """
    Simple Moving Average crossover. Long when fast > slow, flat otherwise.
    
    Parameters:
      fast: [10, 50], default 50
      slow: [100, 300], default 200
    
    Known failure mode: whipsaw in range-bound markets (ADX < 20).
    Validated reference: Faber (2007) TAA; expect Sharpe 0.4-0.7 on SPY 2000-2023.
    """

    def __init__(
        self,
        asset_ids: List[str],
        event_queue: EventQueue,
        fast: int = 50,
        slow: int = 200,
        long_only: bool = True,
    ):
        super().__init__(
            strategy_id=f"SMA_{fast}_{slow}",
            asset_ids=asset_ids,
            event_queue=event_queue,
            warmup_bars=slow + 5,
        )
        self.fast = fast
        self.slow = slow
        self.long_only = long_only
        self._positions: Dict[str, Direction] = {a: Direction.FLAT for a in asset_ids}

    def on_bar(self, event: BarEvent, data_handler: DataHandler) -> None:
        asset_id = event.asset_id
        if asset_id not in self.asset_ids:
            return

        self._tick(asset_id)
        if not self._is_warmed_up(asset_id):
            return

        bars = data_handler.get_latest_bars(asset_id, n=self.slow + 5)
        if len(bars) < self.slow:
            return

        closes = bars["adj_close"] if "adj_close" in bars.columns else bars["close"]

        sma_fast = closes.iloc[-self.fast:].mean()
        sma_slow = closes.iloc[-self.slow:].mean()

        prev_fast = closes.iloc[-(self.fast+1):-1].mean()
        prev_slow = closes.iloc[-(self.slow+1):-1].mean()

        current_direction = self._positions[asset_id]

        # Crossover detected
        if sma_fast > sma_slow and prev_fast <= prev_slow:
            if current_direction != Direction.LONG:
                self._positions[asset_id] = Direction.LONG
                self._emit_signal(
                    asset_id, Direction.LONG, event.timestamp,
                    signal_type="trend_crossover",
                    metadata={"sma_fast": sma_fast, "sma_slow": sma_slow},
                )
        elif sma_fast < sma_slow and prev_fast >= prev_slow:
            if current_direction != Direction.FLAT:
                self._positions[asset_id] = Direction.FLAT
                self._emit_signal(
                    asset_id, Direction.FLAT, event.timestamp,
                    signal_type="trend_crossover",
                    metadata={"sma_fast": sma_fast, "sma_slow": sma_slow},
                )


class EMACrossover(Strategy):
    """
    Exponential Moving Average crossover.
    
    Parameters:
      fast_span: [8, 26], default 12
      slow_span: [21, 100], default 26
    """

    def __init__(
        self,
        asset_ids: List[str],
        event_queue: EventQueue,
        fast_span: int = 12,
        slow_span: int = 26,
        long_only: bool = True,
    ):
        super().__init__(
            strategy_id=f"EMA_{fast_span}_{slow_span}",
            asset_ids=asset_ids,
            event_queue=event_queue,
            warmup_bars=slow_span * 3,
        )
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.long_only = long_only
        self._positions: Dict[str, Direction] = {a: Direction.FLAT for a in asset_ids}

    def on_bar(self, event: BarEvent, data_handler: DataHandler) -> None:
        asset_id = event.asset_id
        if asset_id not in self.asset_ids:
            return

        self._tick(asset_id)
        if not self._is_warmed_up(asset_id):
            return

        bars = data_handler.get_latest_bars(asset_id, n=self.slow_span * 4)
        if len(bars) < self.slow_span * 2:
            return

        closes = bars["adj_close"] if "adj_close" in bars.columns else bars["close"]
        ema_fast = closes.ewm(span=self.fast_span, adjust=False).mean()
        ema_slow = closes.ewm(span=self.slow_span, adjust=False).mean()

        if len(ema_fast) < 2:
            return

        current_cross = ema_fast.iloc[-1] > ema_slow.iloc[-1]
        prev_cross = ema_fast.iloc[-2] > ema_slow.iloc[-2]
        current_direction = self._positions[asset_id]

        if current_cross and not prev_cross:
            if current_direction != Direction.LONG:
                self._positions[asset_id] = Direction.LONG
                self._emit_signal(asset_id, Direction.LONG, event.timestamp, signal_type="ema_cross")
        elif not current_cross and prev_cross:
            if current_direction != Direction.FLAT:
                self._positions[asset_id] = Direction.FLAT
                self._emit_signal(asset_id, Direction.FLAT, event.timestamp, signal_type="ema_cross")


class MACDStrategy(Strategy):
    """
    MACD with signal line. Long when MACD crosses above signal.
    MACD = EMA(12) - EMA(26); Signal = EMA(9) of MACD.
    Daily bars only.
    """

    def __init__(
        self,
        asset_ids: List[str],
        event_queue: EventQueue,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ):
        super().__init__(
            strategy_id=f"MACD_{fast}_{slow}_{signal}",
            asset_ids=asset_ids,
            event_queue=event_queue,
            warmup_bars=slow * 3 + signal,
        )
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self._positions: Dict[str, Direction] = {a: Direction.FLAT for a in asset_ids}

    def on_bar(self, event: BarEvent, data_handler: DataHandler) -> None:
        asset_id = event.asset_id
        if asset_id not in self.asset_ids:
            return

        self._tick(asset_id)
        if not self._is_warmed_up(asset_id):
            return

        bars = data_handler.get_latest_bars(asset_id, n=self.slow * 4)
        if len(bars) < self.slow + self.signal:
            return

        closes = bars["adj_close"] if "adj_close" in bars.columns else bars["close"]
        ema_fast = closes.ewm(span=self.fast, adjust=False).mean()
        ema_slow = closes.ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()

        if len(macd_line) < 2:
            return

        bullish = macd_line.iloc[-1] > signal_line.iloc[-1]
        prev_bullish = macd_line.iloc[-2] > signal_line.iloc[-2]
        current_direction = self._positions[asset_id]

        histogram = (macd_line - signal_line).iloc[-1]

        if bullish and not prev_bullish:
            if current_direction != Direction.LONG:
                self._positions[asset_id] = Direction.LONG
                self._emit_signal(
                    asset_id, Direction.LONG, event.timestamp,
                    signal_type="macd",
                    metadata={"macd": float(macd_line.iloc[-1]), "histogram": float(histogram)},
                )
        elif not bullish and prev_bullish:
            if current_direction != Direction.FLAT:
                self._positions[asset_id] = Direction.FLAT
                self._emit_signal(asset_id, Direction.FLAT, event.timestamp, signal_type="macd")


class DonchianBreakout(Strategy):
    """
    Turtle Trading / Donchian channel breakout.
    
    Entry: close above highest high of N bars (long); below lowest low (short).
    Parameters: N in [20, 55]; default 20 (original Turtle entry period).
    
    Design intent: use on diversified basket (equity ETFs, bond ETFs, commodities).
    Running on single equities alone misses the multi-asset diversification.
    """

    def __init__(
        self,
        asset_ids: List[str],
        event_queue: EventQueue,
        entry_period: int = 20,
        exit_period: int = 10,
        long_only: bool = False,
    ):
        super().__init__(
            strategy_id=f"Donchian_{entry_period}",
            asset_ids=asset_ids,
            event_queue=event_queue,
            warmup_bars=entry_period + 5,
        )
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.long_only = long_only
        self._positions: Dict[str, Direction] = {a: Direction.FLAT for a in asset_ids}

    def on_bar(self, event: BarEvent, data_handler: DataHandler) -> None:
        asset_id = event.asset_id
        if asset_id not in self.asset_ids:
            return

        self._tick(asset_id)
        if not self._is_warmed_up(asset_id):
            return

        bars = data_handler.get_latest_bars(asset_id, n=self.entry_period + 5)
        if len(bars) < self.entry_period:
            return

        closes = bars["adj_close"] if "adj_close" in bars.columns else bars["close"]
        highs = bars["adj_high"] if "adj_high" in bars.columns else bars["high"]
        lows = bars["adj_low"] if "adj_low" in bars.columns else bars["low"]

        current_close = closes.iloc[-1]
        highest_high = highs.iloc[-self.entry_period:-1].max()
        lowest_low = lows.iloc[-self.entry_period:-1].min()

        exit_high = highs.iloc[-self.exit_period:-1].max()
        exit_low = lows.iloc[-self.exit_period:-1].min()

        current_direction = self._positions[asset_id]

        if current_close > highest_high:
            if current_direction != Direction.LONG:
                self._positions[asset_id] = Direction.LONG
                self._emit_signal(
                    asset_id, Direction.LONG, event.timestamp,
                    signal_type="donchian_breakout",
                    holding_period=self.entry_period * 2,
                    metadata={"breakout_level": highest_high},
                )
        elif current_close < lowest_low and not self.long_only:
            if current_direction != Direction.SHORT:
                self._positions[asset_id] = Direction.SHORT
                self._emit_signal(
                    asset_id, Direction.SHORT, event.timestamp,
                    signal_type="donchian_breakout",
                    holding_period=self.entry_period * 2,
                    metadata={"breakdown_level": lowest_low},
                )
        elif current_direction == Direction.LONG and current_close < exit_low:
            self._positions[asset_id] = Direction.FLAT
            self._emit_signal(asset_id, Direction.FLAT, event.timestamp, signal_type="donchian_exit")
        elif current_direction == Direction.SHORT and current_close > exit_high:
            self._positions[asset_id] = Direction.FLAT
            self._emit_signal(asset_id, Direction.FLAT, event.timestamp, signal_type="donchian_exit")


class ADXFilteredTrend(Strategy):
    """
    Apply any trend signal only when ADX(14) > threshold (default 25).
    ADX < threshold = range-bound = go to cash.
    Uses +DI/-DI for direction determination.
    """

    def __init__(
        self,
        asset_ids: List[str],
        event_queue: EventQueue,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        trend_fast: int = 50,
        trend_slow: int = 200,
    ):
        super().__init__(
            strategy_id=f"ADXFiltered_{adx_threshold}",
            asset_ids=asset_ids,
            event_queue=event_queue,
            warmup_bars=trend_slow + adx_period * 2,
        )
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.trend_fast = trend_fast
        self.trend_slow = trend_slow
        self._positions: Dict[str, Direction] = {a: Direction.FLAT for a in asset_ids}

    def _compute_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Compute ADX using Wilder's exponential smoothing."""
        n = self.adx_period

        high_diff = high.diff()
        low_diff = -low.diff()

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.ewm(alpha=1/n, adjust=False).mean()
        smoothed_pdm = pd.Series(plus_dm).ewm(alpha=1/n, adjust=False).mean()
        smoothed_mdm = pd.Series(minus_dm).ewm(alpha=1/n, adjust=False).mean()

        pdi = 100 * smoothed_pdm / atr.replace(0, np.nan)
        mdi = 100 * smoothed_mdm / atr.replace(0, np.nan)

        dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
        adx = dx.ewm(alpha=1/n, adjust=False).mean()
        return adx.fillna(0)

    def on_bar(self, event: BarEvent, data_handler: DataHandler) -> None:
        asset_id = event.asset_id
        if asset_id not in self.asset_ids:
            return

        self._tick(asset_id)
        if not self._is_warmed_up(asset_id):
            return

        bars = data_handler.get_latest_bars(asset_id, n=self.trend_slow + 30)
        if len(bars) < self.trend_slow + self.adx_period * 2:
            return

        closes = bars["adj_close"] if "adj_close" in bars.columns else bars["close"]
        highs = bars.get("adj_high", bars.get("high", closes))
        lows = bars.get("adj_low", bars.get("low", closes))

        adx = self._compute_adx(highs, lows, closes)
        current_adx = adx.iloc[-1]

        sma_fast = closes.iloc[-self.trend_fast:].mean()
        sma_slow = closes.iloc[-self.trend_slow:].mean()
        current_direction = self._positions[asset_id]

        if current_adx > self.adx_threshold:
            # Trending: follow the trend
            if sma_fast > sma_slow and current_direction != Direction.LONG:
                self._positions[asset_id] = Direction.LONG
                self._emit_signal(
                    asset_id, Direction.LONG, event.timestamp,
                    signal_type="adx_trend",
                    metadata={"adx": float(current_adx)},
                )
            elif sma_fast < sma_slow and current_direction == Direction.LONG:
                self._positions[asset_id] = Direction.FLAT
                self._emit_signal(asset_id, Direction.FLAT, event.timestamp, signal_type="adx_exit")
        else:
            # Range-bound: go flat
            if current_direction != Direction.FLAT:
                self._positions[asset_id] = Direction.FLAT
                self._emit_signal(
                    asset_id, Direction.FLAT, event.timestamp,
                    signal_type="adx_ranging",
                    metadata={"adx": float(current_adx)},
                )


class TimeSeriesMomentum(Strategy):
    """
    Moskowitz, Ooi, Pedersen (2012): time-series momentum.
    Signal = sign of 12-1 month return (skip most recent month).
    
    Best on diversified ETF basket. Works by design on multiple asset classes.
    Parameters: 12 months lookback, 1 month skip (standard from published paper).
    Monthly rebalance.
    """

    def __init__(
        self,
        asset_ids: List[str],
        event_queue: EventQueue,
        lookback_months: int = 12,
        skip_months: int = 1,
        rebalance_day: int = 1,  # first trading day of month
    ):
        super().__init__(
            strategy_id="TSMOM_12_1",
            asset_ids=asset_ids,
            event_queue=event_queue,
            warmup_bars=(lookback_months + skip_months) * 23,
        )
        self.lookback_bars = lookback_months * 21
        self.skip_bars = skip_months * 21
        self._last_rebalance_month: Dict[str, int] = {}
        self._positions: Dict[str, Direction] = {a: Direction.FLAT for a in asset_ids}

    def on_bar(self, event: BarEvent, data_handler: DataHandler) -> None:
        asset_id = event.asset_id
        if asset_id not in self.asset_ids:
            return

        self._tick(asset_id)
        if not self._is_warmed_up(asset_id):
            return

        # Monthly rebalance only
        current_month = event.timestamp.month
        if self._last_rebalance_month.get(asset_id) == current_month:
            return
        self._last_rebalance_month[asset_id] = current_month

        total_bars = self.lookback_bars + self.skip_bars + 5
        bars = data_handler.get_latest_bars(asset_id, n=total_bars)

        if len(bars) < self.lookback_bars + self.skip_bars:
            return

        closes = bars["adj_close"] if "adj_close" in bars.columns else bars["close"]

        # 12-1 month return: from 12 months ago to 1 month ago
        price_12m_ago = closes.iloc[-(self.lookback_bars + self.skip_bars)]
        price_1m_ago = closes.iloc[-self.skip_bars]

        momentum_return = (price_1m_ago / price_12m_ago) - 1.0

        current_direction = self._positions[asset_id]
        new_direction = Direction.LONG if momentum_return > 0 else Direction.SHORT

        if new_direction != current_direction:
            self._positions[asset_id] = new_direction
            self._emit_signal(
                asset_id, new_direction, event.timestamp,
                confidence=min(1.0, abs(momentum_return) * 5),
                signal_type="tsmom",
                holding_period=21,
                metadata={"momentum_12_1": float(momentum_return)},
            )
