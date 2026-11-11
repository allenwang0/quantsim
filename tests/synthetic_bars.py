"""Write synthetic OHLC bars into a QuantSim SQLite database for engine tests."""
from core.database import db_conn
from core.database_v2 import init_full_db
from data.ingestion import get_trading_calendar


def write_bars(db_path, asset_id, dates, opens, closes):
    init_full_db(db_path)
    with db_conn(db_path) as conn:
        for date, o, c in zip(dates, opens, closes):
            ts = int(date.timestamp())
            conn.execute(
                "INSERT OR REPLACE INTO raw_bars "
                "(asset_id, timestamp, open, high, low, close, volume, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (asset_id, ts, o, max(o, c) * 1.001, min(o, c) * 0.999, c, 10_000_000, "synthetic"),
            )
            conn.execute(
                "INSERT OR REPLACE INTO adjustment_factors "
                "(asset_id, effective_date, cumulative_split_factor, cumulative_div_adjustment) "
                "VALUES (?, ?, 1.0, 0.0)",
                (asset_id, ts),
            )


def trading_days(n):
    days = get_trading_calendar("2020-01-02", "2025-12-31")
    return list(days[:n])
