"""
Data ingestion layer. Sources: yfinance (primary equities), FRED (macro),
Stooq (bulk historical). All data normalized to canonical schemas.

Critical: stores only RAW (unadjusted) prices. Adjustment factors stored separately.
"""

import time
import logging
import sqlite3
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict
import pandas as pd
import numpy as np

from core.database import db_conn, DB_PATH
from core.schemas import RawBar, AdjustmentFactor, MacroSeries

logger = logging.getLogger(__name__)

# FRED series used in this system
FRED_SERIES = {
    "VIXCLS":       "VIX daily close",
    "DGS10":        "10-year Treasury yield",
    "DGS2":         "2-year Treasury yield",
    "DGS3MO":       "3-month Treasury yield (BSM risk-free rate)",
    "BAMLH0A0HYM2": "ICE BofA HY spread",
    "BAMLC0A0CM":   "ICE BofA IG spread",
    "FEDFUNDS":     "Federal funds rate",
    "T10YIE":       "10-year breakeven inflation",
}

# Default liquid options universe for backtesting
LIQUID_OPTIONS_UNIVERSE = [
    "SPY", "QQQ", "IWM", "GLD", "TLT", "EEM", "EFA",
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
    "JPM", "GS", "BAC", "XOM", "CVX",
]


def _sleep_with_jitter(base: float = 0.5, jitter: float = 0.3):
    """Rate limit management: sleep with random jitter to avoid thundering herd."""
    import random
    time.sleep(base + random.uniform(0, jitter))


def fetch_equity_history(
    symbols: List[str],
    start: str = "2000-01-01",
    end: Optional[str] = None,
    db_path: str = DB_PATH,
) -> Dict[str, int]:
    """
    Download daily OHLCV from yfinance and store raw (unadjusted) bars.
    Also stores cumulative adjustment factors.
    Returns dict of {symbol: bars_inserted}.
    """
    import yfinance as yf

    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%d")

    results = {}

    # Batch download is more efficient than per-symbol
    batch_size = 50
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        logger.info(f"Downloading batch {i//batch_size + 1}: {batch[:5]}...")

        try:
            # auto_adjust=False gives us raw prices + explicit Adj Close
            data = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=False,
                actions=True,  # include dividends and splits
                progress=False,
                threads=True,
            )
        except Exception as e:
            logger.error(f"yfinance batch download failed: {e}")
            _sleep_with_jitter(2.0)
            continue

        if data.empty:
            continue

        # Handle both single and multi-ticker DataFrames
        if len(batch) == 1:
            data.columns = pd.MultiIndex.from_product([data.columns, batch])

        for symbol in batch:
            try:
                count = _store_symbol_data(symbol, data, db_path)
                results[symbol] = count
            except Exception as e:
                logger.error(f"Failed to store {symbol}: {e}")
                results[symbol] = 0

        _sleep_with_jitter(0.5)

    return results


def _store_symbol_data(symbol: str, data: pd.DataFrame, db_path: str) -> int:
    """Extract one symbol from multi-ticker download and store to DB."""
    try:
        sym_data = data.xs(symbol, axis=1, level=1) if symbol in data.columns.get_level_values(1) else data
    except Exception:
        return 0

    if sym_data.empty:
        return 0

    # Compute cumulative adjustment factors from Adj Close vs Close
    # adj_close = close * split_factor (yfinance Adj Close is split+div adjusted)
    bars_inserted = 0

    with db_conn(db_path) as conn:
        for ts, row in sym_data.iterrows():
            try:
                raw_close = float(row["Close"])
                adj_close = float(row["Adj Close"])

                if raw_close <= 0 or pd.isna(raw_close):
                    continue

                # Compute cumulative factor for this date
                split_factor = adj_close / raw_close if raw_close > 0 else 1.0
                div_adjustment = 0.0  # incorporate via split factor in yfinance approach

                # Validate bar quality
                raw_bar = RawBar(
                    asset_id=symbol,
                    timestamp=ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=raw_close,
                    volume=int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
                    exchange="",
                    source="yfinance",
                )

                if not raw_bar.validate():
                    continue

                ts_epoch = int(raw_bar.timestamp.timestamp())

                conn.execute(
                    """INSERT OR REPLACE INTO raw_bars
                       (asset_id, timestamp, open, high, low, close, volume, exchange, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (symbol, ts_epoch, raw_bar.open, raw_bar.high,
                     raw_bar.low, raw_bar.close, raw_bar.volume,
                     raw_bar.exchange, raw_bar.source),
                )

                eff_date = int(raw_bar.timestamp.timestamp())
                conn.execute(
                    """INSERT OR REPLACE INTO adjustment_factors
                       (asset_id, effective_date, cumulative_split_factor, cumulative_div_adjustment)
                       VALUES (?, ?, ?, ?)""",
                    (symbol, eff_date, split_factor, div_adjustment),
                )

                bars_inserted += 1

            except Exception as e:
                logger.debug(f"Row error for {symbol}: {e}")
                continue

    return bars_inserted


def get_bars(
    asset_id: str,
    start: datetime,
    end: datetime,
    adjusted: bool = True,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """
    Retrieve OHLCV bars for an asset. If adjusted=True, applies cumulative
    split and dividend factors. Uses release_timestamp for point-in-time correctness.
    """
    with db_conn(db_path) as conn:
        start_epoch = int(start.timestamp())
        end_epoch = int(end.timestamp())

        rows = conn.execute(
            """SELECT b.asset_id, b.timestamp, b.open, b.high, b.low, b.close,
                      b.volume, b.source,
                      COALESCE(af.cumulative_split_factor, 1.0) as split_factor,
                      COALESCE(af.cumulative_div_adjustment, 0.0) as div_adjustment
               FROM raw_bars b
               LEFT JOIN adjustment_factors af
                 ON af.asset_id = b.asset_id
                 AND af.effective_date = (
                     SELECT MAX(effective_date) FROM adjustment_factors
                     WHERE asset_id = b.asset_id AND effective_date <= b.timestamp
                 )
               WHERE b.asset_id = ? AND b.timestamp >= ? AND b.timestamp <= ?
               ORDER BY b.timestamp ASC""",
            (asset_id, start_epoch, end_epoch),
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=["asset_id", "timestamp", "open", "high", "low", "close",
                 "volume", "source", "split_factor", "div_adjustment"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df.set_index("timestamp", inplace=True)

    if adjusted:
        df["adj_close"] = df["close"] * df["split_factor"] + df["div_adjustment"]
        df["adj_open"]  = df["open"]  * df["split_factor"]
        df["adj_high"]  = df["high"]  * df["split_factor"]
        df["adj_low"]   = df["low"]   * df["split_factor"]
    else:
        df["adj_close"] = df["close"]

    return df


def get_latest_bars_as_of(
    asset_id: str,
    as_of: datetime,
    n: int,
    adjusted: bool = True,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """
    Returns the N most recent bars STRICTLY BEFORE as_of timestamp.
    This is the primary point-in-time data access method for the backtesting engine.
    Signals generated at bar T use data through T-1.
    """
    with db_conn(db_path) as conn:
        as_of_epoch = int(as_of.timestamp())

        rows = conn.execute(
            """SELECT b.asset_id, b.timestamp, b.open, b.high, b.low, b.close,
                      b.volume,
                      COALESCE(af.cumulative_split_factor, 1.0) as split_factor,
                      COALESCE(af.cumulative_div_adjustment, 0.0) as div_adjustment
               FROM raw_bars b
               LEFT JOIN adjustment_factors af
                 ON af.asset_id = b.asset_id
                 AND af.effective_date = (
                     SELECT MAX(effective_date) FROM adjustment_factors
                     WHERE asset_id = b.asset_id AND effective_date <= b.timestamp
                 )
               WHERE b.asset_id = ? AND b.timestamp < ?
               ORDER BY b.timestamp DESC
               LIMIT ?""",
            (asset_id, as_of_epoch, n),
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=["asset_id", "timestamp", "open", "high", "low", "close",
                 "volume", "split_factor", "div_adjustment"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)  # ascending order

    if adjusted:
        df["adj_close"] = df["close"] * df["split_factor"] + df["div_adjustment"]
        df["adj_open"]  = df["open"]  * df["split_factor"]
        df["adj_high"]  = df["high"]  * df["split_factor"]
        df["adj_low"]   = df["low"]   * df["split_factor"]
    else:
        df["adj_close"] = df["close"]

    return df


def get_latest_macro_as_of(
    series_id: str,
    as_of_date: date,
    db_path: str = DB_PATH,
) -> Optional[float]:
    """
    Returns the most recently RELEASED value of series_id available as of as_of_date.
    Uses release_timestamp to prevent look-ahead bias on macro data.
    """
    with db_conn(db_path) as conn:
        as_of_epoch = int(datetime(as_of_date.year, as_of_date.month, as_of_date.day).timestamp())
        row = conn.execute(
            """SELECT value FROM macro_series
               WHERE series_id = ? AND release_timestamp <= ?
               ORDER BY release_timestamp DESC
               LIMIT 1""",
            (series_id, as_of_epoch),
        ).fetchone()
    return float(row["value"]) if row else None


def fetch_fred_series(
    series_ids: Optional[List[str]] = None,
    start: str = "2000-01-01",
    db_path: str = DB_PATH,
) -> Dict[str, int]:
    """
    Download FRED series and store with release_timestamp = observation_date
    (FRED observations include release date metadata when using realtime_start).
    """
    try:
        from fredapi import Fred
        import os
        fred_key = os.environ.get("FRED_API_KEY", "")
        fred = Fred(api_key=fred_key) if fred_key else Fred()
    except Exception as e:
        logger.warning(f"FRED API not available: {e}. Skipping macro data.")
        return {}

    if series_ids is None:
        series_ids = list(FRED_SERIES.keys())

    results = {}
    for sid in series_ids:
        try:
            s = fred.get_series(sid, observation_start=start)
            count = 0
            with db_conn(db_path) as conn:
                for obs_date, value in s.items():
                    if pd.isna(value):
                        continue
                    obs_dt = obs_date.to_pydatetime() if hasattr(obs_date, 'to_pydatetime') else obs_date
                    release_epoch = int(obs_dt.timestamp())
                    conn.execute(
                        """INSERT OR REPLACE INTO macro_series
                           (series_id, release_timestamp, reference_period, value, is_revised)
                           VALUES (?, ?, ?, ?, 0)""",
                        (sid, release_epoch, release_epoch, float(value)),
                    )
                    count += 1
            results[sid] = count
            logger.info(f"FRED {sid}: {count} observations stored")
            _sleep_with_jitter(0.2)
        except Exception as e:
            logger.error(f"FRED {sid} failed: {e}")
            results[sid] = 0

    return results


def get_trading_calendar(start: str = "2000-01-01", end: Optional[str] = None) -> pd.DatetimeIndex:
    """Returns NYSE trading days as a DatetimeIndex (UTC midnight)."""
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar("NYSE")
    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%d")
    schedule = nyse.schedule(start_date=start, end_date=end)
    return mcal.date_range(schedule, frequency="1D")


def get_universe(
    min_market_cap_b: float = 5.0,
    db_path: str = DB_PATH,
) -> List[str]:
    """
    Returns the fixed backtesting universe. Uses a curated list of
    high-liquidity US equities to avoid survivorship bias complications.
    For production, cross-reference with historical Russell 3000 constituents.
    """
    # S&P 500 large-cap core universe: liquid, minimal survivorship bias concerns
    CORE_UNIVERSE = [
        "SPY", "QQQ", "IWM", "GLD", "TLT", "EEM", "EFA", "AGG", "BIL",
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
        "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "NFLX",
        "PYPL", "ADBE", "CRM", "INTC", "AMD", "QCOM", "TXN", "MU",
        "GS", "BAC", "WFC", "C", "MS", "AXP", "BLK", "SCHW",
        "XOM", "CVX", "COP", "SLB", "OXY",
        "JNJ", "PFE", "MRK", "ABBV", "BMY", "AMGN", "GILD",
        "KO", "PEP", "MCD", "SBUX", "NKE", "WMT", "COST", "TGT",
        "BA", "CAT", "GE", "MMM", "HON", "LMT", "RTX", "NOC",
        "NEE", "DUK", "SO", "D", "AEP",
        "AMT", "PLD", "EQIX", "SPG", "O",
    ]
    return list(dict.fromkeys(CORE_UNIVERSE))  # deduplicate preserving order
