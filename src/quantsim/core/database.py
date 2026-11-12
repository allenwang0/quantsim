"""
Database layer. SQLite for solo development.
All timestamps stored as Unix epoch INTEGER for fast range queries.
"""

import sqlite3
import os
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

DB_PATH = os.environ.get("QUANTSIM_DB", str(Path.home() / ".quantsim" / "quantsim.db"))


def ensure_db_dir():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    ensure_db_dir()
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")    # WAL: concurrent reads while writing
    conn.execute("PRAGMA synchronous=NORMAL")  # fsync on checkpoint only
    conn.execute("PRAGMA cache_size=-64000")   # 64MB page cache
    conn.execute("PRAGMA temp_store=MEMORY")   # temp tables in RAM
    conn.execute("PRAGMA mmap_size=268435456") # 256MB memory-mapped I/O
    conn.isolation_level = None  # autocommit mode; we manage transactions explicitly
    return conn


@contextmanager
def db_conn(db_path: str = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


SCHEMA_SQL = """
-- Raw unadjusted OHLCV bars. Never store adjusted prices here.
CREATE TABLE IF NOT EXISTS raw_bars (
    asset_id  TEXT    NOT NULL,
    timestamp INTEGER NOT NULL,
    open      REAL    NOT NULL,
    high      REAL    NOT NULL,
    low       REAL    NOT NULL,
    close     REAL    NOT NULL,
    volume    INTEGER NOT NULL,
    exchange  TEXT    DEFAULT '',
    source    TEXT    DEFAULT '',
    PRIMARY KEY (asset_id, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_raw_bars_time ON raw_bars(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_raw_bars_asset_time ON raw_bars(asset_id, timestamp DESC);

-- Cumulative split and dividend adjustment factors.
-- Apply at query time: adj_close = close * split_factor + div_adjustment
CREATE TABLE IF NOT EXISTS adjustment_factors (
    asset_id              TEXT    NOT NULL,
    effective_date        INTEGER NOT NULL,
    cumulative_split_factor REAL  NOT NULL DEFAULT 1.0,
    cumulative_div_adjustment REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (asset_id, effective_date)
);

-- Reconstructed (or live) options chain snapshots.
CREATE TABLE IF NOT EXISTS options_quotes (
    underlying_id  TEXT    NOT NULL,
    timestamp      INTEGER NOT NULL,
    expiration     INTEGER NOT NULL,
    strike         REAL    NOT NULL,
    right          TEXT    NOT NULL CHECK (right IN ('C','P')),
    bid            REAL,
    ask            REAL,
    last           REAL,
    volume         INTEGER DEFAULT 0,
    open_interest  INTEGER DEFAULT 0,
    iv             REAL,
    delta          REAL,
    gamma          REAL,
    theta          REAL,
    vega           REAL,
    rho            REAL,
    is_reconstructed INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (underlying_id, timestamp, expiration, strike, right)
);
CREATE INDEX IF NOT EXISTS idx_options_ts ON options_quotes(timestamp DESC);

-- FRED macro series with release date for point-in-time correctness.
CREATE TABLE IF NOT EXISTS macro_series (
    series_id        TEXT    NOT NULL,
    release_timestamp INTEGER NOT NULL,
    reference_period  INTEGER NOT NULL,
    value            REAL    NOT NULL,
    is_revised       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (series_id, release_timestamp)
);
CREATE INDEX IF NOT EXISTS idx_macro_series ON macro_series(series_id, release_timestamp DESC);

-- Immutable audit log of every event processed by the engine.
CREATE TABLE IF NOT EXISTS event_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT    NOT NULL,
    timestamp  INTEGER NOT NULL,
    payload    TEXT    NOT NULL   -- JSON
);
CREATE INDEX IF NOT EXISTS idx_event_log_time ON event_log(timestamp DESC);

-- Portfolio snapshots persisted on every FillEvent.
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   INTEGER NOT NULL,
    total_equity REAL   NOT NULL,
    cash        REAL    NOT NULL,
    realized_pnl REAL   NOT NULL,
    unrealized_pnl REAL NOT NULL,
    payload     TEXT    NOT NULL   -- JSON positions
);
CREATE INDEX IF NOT EXISTS idx_portfolio_time ON portfolio_snapshots(timestamp DESC);

-- Trade log: one row per closed trade.
CREATE TABLE IF NOT EXISTS trades (
    trade_id        TEXT    PRIMARY KEY,
    strategy_id     TEXT    NOT NULL,
    asset_id        TEXT    NOT NULL,
    direction       TEXT    NOT NULL,
    entry_timestamp INTEGER NOT NULL,
    exit_timestamp  INTEGER,
    entry_price     REAL    NOT NULL,
    exit_price      REAL,
    quantity        INTEGER NOT NULL,
    realized_pnl    REAL,
    commission      REAL    NOT NULL DEFAULT 0,
    holding_bars    INTEGER
);

-- Risk alerts table polled by the dashboard.
CREATE TABLE IF NOT EXISTS risk_alerts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    risk_type TEXT    NOT NULL,
    message   TEXT    NOT NULL,
    severity  TEXT    NOT NULL DEFAULT 'WARNING',
    dismissed INTEGER NOT NULL DEFAULT 0
);
"""


def init_db(db_path: str = DB_PATH) -> None:
    """Create all tables. Safe to call multiple times (IF NOT EXISTS)."""
    ensure_db_dir()
    conn = get_connection(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()
    print(f"[DB] Initialized: {db_path}")
