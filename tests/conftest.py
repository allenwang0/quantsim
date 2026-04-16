"""
pytest configuration: custom marks, shared fixtures.
"""
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "live: marks tests that require live data/API access")
    config.addinivalue_line("markers", "slow: marks tests that take >5 seconds")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")

@pytest.fixture
def tmp_db(tmp_path):
    """Temporary database for test isolation."""
    import os
    db_path = str(tmp_path / "test.db")
    os.environ["QUANTSIM_DB"] = db_path
    from core.database_v2 import init_full_db
    init_full_db(db_path)
    yield db_path
    os.unlink(db_path)

@pytest.fixture
def synthetic_prices():
    """300 bars of synthetic price data for 5 assets."""
    import numpy as np
    import pandas as pd
    np.random.seed(42)
    n, n_assets = 300, 5
    symbols = [f"ASSET_{i}" for i in range(n_assets)]
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = {}
    for s in symbols:
        returns = np.random.normal(0.0003, 0.015, n)
        data[s] = pd.Series(100 * np.cumprod(1 + returns), index=dates, name=s)
    return pd.DataFrame(data)

@pytest.fixture
def synthetic_returns(synthetic_prices):
    return synthetic_prices.pct_change().dropna()

@pytest.fixture
def synthetic_db(tmp_path, synthetic_prices):
    """
    Fully populated test DB with synthetic OHLCV injected.
    Returns (db_path, prices_df).
    """
    import os
    db_path = str(tmp_path / "synth.db")
    os.environ["QUANTSIM_DB"] = db_path
    from core.database_v2 import init_full_db
    init_full_db(db_path)

    from core.database import db_conn
    for sym in synthetic_prices.columns:
        for ts, price in synthetic_prices[sym].items():
            ts_epoch = int(ts.timestamp())
            with db_conn(db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO raw_bars "
                    "(asset_id, timestamp, open, high, low, close, volume, source) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (sym, ts_epoch, price*0.999, price*1.005,
                     price*0.995, price, 1_000_000, "synthetic"),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO adjustment_factors "
                    "(asset_id, effective_date, cumulative_split_factor, cumulative_div_adjustment) "
                    "VALUES (?, ?, 1.0, 0.0)",
                    (sym, ts_epoch),
                )
    return db_path, synthetic_prices


@pytest.fixture
def backtest_engine(synthetic_db):
    """
    Pre-built BacktestEngine with synthetic data and BuyAndHold strategy.
    Use as a reusable test fixture for engine-level tests.
    """
    db_path, prices = synthetic_db
    from core.event_queue import EventQueue
    from strategies.momentum_factor import BuyAndHold
    from backtesting.engine import BacktestEngine
    import pandas as pd

    symbols = list(prices.columns)
    eq = EventQueue()
    strategy = BuyAndHold(asset_ids=symbols, event_queue=eq)

    start = prices.index[0].to_pydatetime()
    end = prices.index[-1].to_pydatetime()

    engine = BacktestEngine(
        strategies=[strategy],
        start=start,
        end=end,
        initial_capital=100_000,
        db_path=db_path,
        verbose=False,
    )
    return engine, db_path
