"""Synthetic price helpers for tests: random-walk series and OHLC bars written to SQLite."""

import numpy as np
import pandas as pd

from quantsim.core.database import db_conn
from quantsim.core.database_v2 import init_full_db
from quantsim.data.ingestion import get_trading_calendar


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


def make_price_series(n=500, drift=0.0003, vol=0.015, seed=42):
    np.random.seed(seed)
    returns = np.random.normal(drift, vol, n)
    prices = 100 * np.cumprod(1 + returns)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates)


def make_price_df(n=500, n_assets=5, seed=42):
    np.random.seed(seed)
    symbols = [f"ASSET_{i}" for i in range(n_assets)]
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    data = {}
    for s in symbols:
        drift = np.random.uniform(-0.0001, 0.0005)
        vol = np.random.uniform(0.01, 0.02)
        returns = np.random.normal(drift, vol, n)
        data[s] = 100 * np.cumprod(1 + returns)
    return pd.DataFrame(data, index=dates)
