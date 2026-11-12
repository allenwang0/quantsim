"""
A daily bar must be stored under the same instant no matter where the
download ran. yfinance indexes daily bars by naive dates; converting those
with the machine's local time zone stored a Tokyo download a day early,
which shifted every fill by one session. Synthetic rows, no network.
"""

import os
import sqlite3
import time

import pandas as pd
import pytest

from quantsim.core.database_v2 import init_full_db
from quantsim.data.ingestion import _store_symbol_data


@pytest.fixture
def machine_timezone():
    original = os.environ.get("TZ")

    def set_tz(name):
        os.environ["TZ"] = name
        time.tzset()

    yield set_tz
    if original is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = original
    time.tzset()


@pytest.mark.parametrize("tz", ["Asia/Tokyo", "America/Los_Angeles", "UTC"])
def test_naive_daily_dates_are_stored_at_utc_midnight(tmp_path, machine_timezone, tz):
    machine_timezone(tz)
    db_path = str(tmp_path / "bars.db")
    init_full_db(db_path)
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    frame = pd.DataFrame(
        [[10.0, 11.0, 9.5, 10.5, 10.5, 1000]],
        index=pd.DatetimeIndex(["2021-03-01"]),
        columns=pd.MultiIndex.from_product([fields, ["SYN"]]),
    )

    _store_symbol_data("SYN", frame, db_path)

    with sqlite3.connect(db_path) as conn:
        (stored,) = conn.execute("SELECT timestamp FROM raw_bars WHERE asset_id='SYN'").fetchone()
    assert stored == int(pd.Timestamp("2021-03-01", tz="UTC").timestamp())
