"""
Seeded synthetic prices: geometric Brownian motion with regime shifts.

Used by the tests, the benchmark and the published demo report so that no
downloaded market data is ever committed. Every output is labeled synthetic.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# (name, annual drift, annual volatility)
REGIMES: List[Tuple[str, float, float]] = [
    ("bull", 0.15, 0.14),
    ("bear", -0.25, 0.32),
    ("sideways", 0.0, 0.10),
]


def regime_path(n_bars: int, rng: np.random.Generator, mean_regime_bars: int = 189) -> np.ndarray:
    """Regime index per bar; regime lengths are geometric with the given mean."""
    regimes = np.empty(n_bars, dtype=np.int64)
    t, current = 0, 0
    while t < n_bars:
        length = int(rng.geometric(1.0 / mean_regime_bars))
        regimes[t : t + length] = current
        t += length
        current = int(rng.choice([r for r in range(len(REGIMES)) if r != current]))
    return regimes


def regime_gbm_prices(
    n_assets: int = 1,
    n_bars: int = 10 * TRADING_DAYS_PER_YEAR,
    seed: int = 0,
    start: str = "2014-01-02",
    start_price: float = 100.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Close prices (business-day index, columns SYN0..SYNn) and the regime name
    per bar. Assets share the regime path and are correlated at about 0.5.
    """
    rng = np.random.default_rng(seed)
    regimes = regime_path(n_bars, rng)
    drift = np.array([REGIMES[r][1] for r in regimes]) / TRADING_DAYS_PER_YEAR
    vol = np.array([REGIMES[r][2] for r in regimes]) / np.sqrt(TRADING_DAYS_PER_YEAR)

    common = rng.normal(0, 1, n_bars)
    idio = rng.normal(0, 1, (n_bars, n_assets))
    shocks = np.sqrt(0.5) * common[:, None] + np.sqrt(0.5) * idio
    log_returns = (drift - 0.5 * vol**2)[:, None] + vol[:, None] * shocks
    closes = start_price * np.exp(np.cumsum(log_returns, axis=0))

    index = pd.bdate_range(start, periods=n_bars)
    prices = pd.DataFrame(closes, index=index, columns=[f"SYN{i}" for i in range(n_assets)])
    regime_names = pd.Series([REGIMES[r][0] for r in regimes], index=index, name="regime")
    return prices, regime_names


def write_prices_to_db(db_path: str, prices: pd.DataFrame, volume: int = 10_000_000) -> None:
    """
    Store close prices as daily OHLCV bars for the event-driven engine.

    Each bar opens at the previous close, so a next-open fill and a
    next-close fill differ only by that session's move. Rows are labeled
    source='synthetic'.
    """
    from quantsim.core.database import db_conn
    from quantsim.core.database_v2 import init_full_db

    init_full_db(db_path)
    opens = prices.shift(1).fillna(prices.iloc[0])
    rows, factors = [], []
    for asset in prices.columns:
        for ts, o, c in zip(prices.index, opens[asset], prices[asset]):
            epoch = int(pd.Timestamp(ts).normalize().tz_localize("UTC").timestamp())
            rows.append((asset, epoch, o, max(o, c), min(o, c), c, volume, "synthetic"))
            factors.append((asset, epoch))
    with db_conn(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO raw_bars "
            "(asset_id, timestamp, open, high, low, close, volume, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.executemany(
            "INSERT OR REPLACE INTO adjustment_factors "
            "(asset_id, effective_date, cumulative_split_factor, cumulative_div_adjustment) "
            "VALUES (?, ?, 1.0, 0.0)",
            factors,
        )
