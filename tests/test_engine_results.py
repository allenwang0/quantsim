"""
Event-engine analytics must be computed on one equity value per session.

The engine wrote an equity snapshot every 21 bars and fed those monthly
points to analytics that annualize by sqrt(252) and count 252 points a
year, which inflated CAGR and mis-scaled Sharpe. Synthetic prices.
"""

import numpy as np
import pytest

from quantsim.backtesting.engine import BacktestEngine
from quantsim.core.event_queue import EventQueue
from quantsim.data.synthetic import regime_gbm_prices, write_prices_to_db
from quantsim.strategies.momentum_factor import BuyAndHold


def test_analytics_use_daily_equity(tmp_path):
    prices, _ = regime_gbm_prices(n_assets=1, n_bars=504, seed=3)
    db_path = str(tmp_path / "bars.db")
    write_prices_to_db(db_path, prices)
    engine = BacktestEngine(
        strategies=[BuyAndHold(["SYN0"], EventQueue())],
        start=prices.index[0].to_pydatetime(),
        end=prices.index[-1].to_pydatetime(),
        warmup_bars=0,
        db_path=db_path,
        verbose=False,
    )

    results = engine.run()

    sessions = results["_meta"]["bars_processed"]
    assert results["n_years"] == pytest.approx((sessions - 1) / 252, abs=2 / 252)
    years = results["n_years"]
    expected_cagr = (1 + results["total_return"]) ** (1 / years) - 1
    assert results["cagr"] == pytest.approx(expected_cagr, rel=1e-9)
    assert np.isfinite(results["sharpe_ratio"])
