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
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from quantsim.core.event_queue import EventQueue
from quantsim.core.events import BarEvent

logger = logging.getLogger(__name__)


def session_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Map daily bar timestamps to their session date (UTC midnight).

    Sources stamp a daily bar at the exchange close, at UTC midnight or at
    US midnight; all of these fall on the session's UTC calendar date.
    """
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert("UTC").normalize()


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
        from quantsim.core.database import DB_PATH
        from quantsim.data.ingestion import get_bars, get_latest_macro_as_of, get_trading_calendar

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
        first_session = session_dates(pd.DatetimeIndex([start]))[0]
        last_session = session_dates(pd.DatetimeIndex([end]))[0]
        for asset_id in asset_ids:
            df = get_bars(
                asset_id,
                start - timedelta(days=1),
                end + timedelta(days=1),
                adjusted=True,
                db_path=self._db_path,
            )
            if not df.empty:
                df.index = session_dates(df.index)
                df = df[~df.index.duplicated(keep="last")]
                in_range = (df.index >= first_session) & (df.index <= last_session)
                self._bar_cache[asset_id] = df[in_range]
            else:
                logger.warning(f"No data for {asset_id} in range {start}-{end}")

        # Trading calendar for stepping through time
        trading_days = get_trading_calendar(
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )
        self._trading_days = session_dates(pd.DatetimeIndex(trading_days))
        self._current_index = 0
        self._session: Optional[pd.Timestamp] = None

        logger.info(
            f"DataHandler ready: {len(self._bar_cache)} assets, "
            f"{len(self._trading_days)} trading days"
        )

    @property
    def current_datetime(self) -> datetime:
        """The session whose bars were fired last (the start date before the first bar)."""
        if self._session is None:
            return self._start
        return self._session.to_pydatetime()

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

        session = self._trading_days[self._current_index]
        self._session = session

        for asset_id, df in self._bar_cache.items():
            if session not in df.index:
                continue
            row = df.loc[session]
            event = BarEvent(
                timestamp=session.to_pydatetime(),
                asset_id=asset_id,
                open=float(row.get("open", row.get("adj_open", 0))),
                high=float(row.get("high", row.get("adj_high", 0))),
                low=float(row.get("low", row.get("adj_low", 0))),
                close=float(row.get("adj_close", row.get("close", 0))),
                volume=int(row.get("volume", 0)),
                adj_close=float(row.get("adj_close", row.get("close", 0))),
            )
            self._event_queue.put(event)

        self._current_index += 1
        return True

    def get_latest_bars(
        self,
        asset_id: str,
        n: int = 1,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """
        Returns the N most recent bars up to and including the current session.

        A daily bar is processed after its close, so its own close is known;
        orders created from it fill on the next session's bar.
        """
        if asset_id not in self._bar_cache or self._session is None:
            return pd.DataFrame()
        df = self._bar_cache[asset_id]
        known = df[df.index <= self._session]
        if known.empty:
            return pd.DataFrame()
        return known.iloc[-n:].copy()

    def get_current_bar(self, asset_id: str) -> Optional[pd.Series]:
        """Returns the asset's bar for the current session, or its last earlier bar."""
        bars = self.get_latest_bars(asset_id, n=1)
        if bars.empty:
            return None
        return bars.iloc[-1]

    def get_macro_value(self, series_id: str) -> Optional[float]:
        current_date = self.current_datetime.date()
        return self._get_latest_macro(series_id, current_date, self._db_path)

    def is_warmup_complete(self) -> bool:
        return self._current_index > self._warmup_bars
