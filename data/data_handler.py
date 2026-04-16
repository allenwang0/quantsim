"""
DataHandler: abstract interface over data access.
HistoricalDataHandler implements it for backtesting (SQLite reads).
LiveDataHandler (paper_trading module) implements it for live mode.

The critical invariant: get_latest_bars() NEVER returns the current bar.
Signals generated using data through T-1 execute at T+1's open.
"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict
import pandas as pd

from core.event_queue import EventQueue
from core.events import BarEvent, AssetType

logger = logging.getLogger(__name__)


class DataHandler(ABC):
    """
    Abstract base class for all data handlers.
    Both backtesting and paper trading implement this interface.
    Strategy code never knows which implementation is active.
    """

    @abstractmethod
    def get_latest_bars(
        self,
        asset_id: str,
        n: int = 1,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """
        Returns N most recent bars STRICTLY BEFORE the current bar timestamp.
        Enforces point-in-time correctness: no look-ahead.
        """
        ...

    @abstractmethod
    def get_current_bar(self, asset_id: str) -> Optional[pd.Series]:
        """Returns the current (most recent completed) bar for execution price."""
        ...

    @abstractmethod
    def get_macro_value(self, series_id: str) -> Optional[float]:
        """Returns the most recently released macro value as of current timestamp."""
        ...

    @property
    @abstractmethod
    def current_datetime(self) -> datetime:
        """The engine's current clock time."""
        ...

    @property
    @abstractmethod
    def universe(self) -> List[str]:
        """List of asset IDs in the current universe."""
        ...


class HistoricalDataHandler(DataHandler):
    """
    DataHandler for backtesting. Reads from SQLite.
    Advances one trading day at a time via update_bars().
    """

    def __init__(
        self,
        asset_ids: List[str],
        start: datetime,
        end: datetime,
        event_queue: EventQueue,
        warmup_bars: int = 252,
        db_path: Optional[str] = None,
    ):
        from core.database import DB_PATH
        from data.ingestion import get_bars, get_trading_calendar, get_latest_macro_as_of

        self._asset_ids = asset_ids
        self._start = start
        self._end = end
        self._event_queue = event_queue
        self._warmup_bars = warmup_bars
        self._db_path = db_path or DB_PATH
        self._get_latest_macro = get_latest_macro_as_of

        # Pre-load all bar data into memory for performance
        # This is safe since we're working with daily bars (not tick data)
        logger.info(f"Loading historical bars for {len(asset_ids)} assets...")
        self._bar_cache: Dict[str, pd.DataFrame] = {}
        for asset_id in asset_ids:
            df = get_bars(asset_id, start, end, adjusted=True, db_path=self._db_path)
            if not df.empty:
                self._bar_cache[asset_id] = df
            else:
                logger.warning(f"No data for {asset_id} in range {start}-{end}")

        # Trading calendar for stepping through time
        trading_days = get_trading_calendar(
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )
        self._trading_days = pd.DatetimeIndex(trading_days).normalize()
        self._current_index = 0

        logger.info(f"DataHandler ready: {len(self._bar_cache)} assets, "
                    f"{len(self._trading_days)} trading days")

    @property
    def current_datetime(self) -> datetime:
        if self._current_index < len(self._trading_days):
            return self._trading_days[self._current_index].to_pydatetime()
        return self._end

    @property
    def universe(self) -> List[str]:
        return list(self._bar_cache.keys())

    def has_more_data(self) -> bool:
        return self._current_index < len(self._trading_days)

    def update_bars(self) -> bool:
        """
        Advance one trading day. Fires BarEvents for all assets with data on this day.
        Returns False when all data has been consumed.
        """
        if not self.has_more_data():
            return False

        current_ts = self._trading_days[self._current_index]

        bars_fired = 0
        for asset_id, df in self._bar_cache.items():
            # Find the bar closest to current_ts
            if current_ts not in df.index:
                # Try normalized comparison
                matching = df.index[df.index.normalize() == current_ts.normalize()]
                if len(matching) == 0:
                    continue
                row = df.loc[matching[0]]
            else:
                row = df.loc[current_ts]

            event = BarEvent(
                timestamp=current_ts.to_pydatetime(),
                asset_id=asset_id,
                open=float(row.get("open", row.get("adj_open", 0))),
                high=float(row.get("high", row.get("adj_high", 0))),
                low=float(row.get("low", row.get("adj_low", 0))),
                close=float(row.get("adj_close", row.get("close", 0))),
                volume=int(row.get("volume", 0)),
                adj_close=float(row.get("adj_close", row.get("close", 0))),
            )
            self._event_queue.put(event)
            bars_fired += 1

        self._current_index += 1
        return True

    def get_latest_bars(
        self,
        asset_id: str,
        n: int = 1,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """
        Returns N bars strictly before current_datetime.
        NEVER includes the current bar. This is the look-ahead bias prevention.
        """
        if asset_id not in self._bar_cache:
            return pd.DataFrame()

        df = self._bar_cache[asset_id]
        current_ts = self.current_datetime

        # Strictly before current timestamp
        ts_aware = pd.Timestamp(current_ts).tz_localize("UTC") if getattr(current_ts, "tzinfo", None) is None else pd.Timestamp(current_ts).tz_convert("UTC")
        prior = df[df.index < ts_aware]
        if len(prior) == 0:
            return pd.DataFrame()

        return prior.iloc[-n:].copy()

    def get_current_bar(self, asset_id: str) -> Optional[pd.Series]:
        """Returns the current bar (for execution price calculation)."""
        if asset_id not in self._bar_cache:
            return None

        df = self._bar_cache[asset_id]
        current_ts = self.current_datetime

        # Current bar: at or immediately before current timestamp
        ts_aware2 = pd.Timestamp(current_ts).tz_localize("UTC") if getattr(current_ts, "tzinfo", None) is None else pd.Timestamp(current_ts).tz_convert("UTC")
        at_or_before = df[df.index <= ts_aware2]
        if at_or_before.empty:
            return None
        return at_or_before.iloc[-1]

    def get_macro_value(self, series_id: str) -> Optional[float]:
        current_date = self.current_datetime.date()
        return self._get_latest_macro(series_id, current_date, self._db_path)

    def is_warmup_complete(self) -> bool:
        return self._current_index > self._warmup_bars
