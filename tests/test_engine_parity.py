"""
The event-driven and vectorized engines must agree on the same strategy.

One seeded synthetic series with open == close on every bar, so a fill at bar
T+1's open (event engine) and at bar T+1's price (vectorized) is the same
trade. The series starts with a decline so both engines begin flat.
"""

import numpy as np
import pandas as pd

from quantsim.backtesting.engine import BacktestEngine
from quantsim.backtesting.execution import FillModel, NoSlippage, ZeroCommission
from quantsim.backtesting.vectorized import _simulate_portfolio_numba, sma_crossover_signal
from quantsim.core.event_queue import EventQueue
from quantsim.core.events import OrderSide
from quantsim.portfolio.sizing import FixedFractionalSizer
from quantsim.strategies.trend import SMAcrossover
from tests.synthetic_bars import trading_days, write_bars

FAST, SLOW, SIZE, CAPITAL = 5, 20, 0.10, 100_000.0


def synthetic_closes(n_bars, seed=3):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0003, 0.012, n_bars)
    steps[:40] = -0.004
    return 100 * np.exp(np.cumsum(steps))


def test_sma_crossover_matches_across_engines(tmp_path):
    n_bars = 400
    dates = trading_days(n_bars)
    closes = synthetic_closes(n_bars)
    db_path = str(tmp_path / "parity.db")
    write_bars(db_path, "SYN", dates, closes, closes)

    strategy = SMAcrossover(["SYN"], EventQueue(), fast=FAST, slow=SLOW)
    engine = BacktestEngine(
        strategies=[strategy],
        start=dates[0].to_pydatetime(),
        end=dates[-1].to_pydatetime(),
        initial_capital=CAPITAL,
        sizer=FixedFractionalSizer(fraction=SIZE, max_position_fraction=SIZE),
        slippage_model=NoSlippage(),
        commission_model=ZeroCommission(),
        fill_mode=FillModel.NEXT_BAR_OPEN,
        warmup_bars=0,
        db_path=db_path,
        verbose=False,
    )
    fills = []
    handle_fill = engine._handle_fill
    engine._handle_fill = lambda event: (fills.append(event), handle_fill(event))
    engine.run()
    event_round_trips = sum(f.side == OrderSide.SELL for f in fills)
    event_return = engine.portfolio.total_equity / CAPITAL - 1

    prices = pd.DataFrame({"SYN": closes})
    signals = sma_crossover_signal(prices, prices.pct_change(), fast=FAST, slow=SLOW)
    equity, positions, _ = _simulate_portfolio_numba(
        prices.to_numpy(), signals.to_numpy(dtype=np.float64), CAPITAL, 0.0, 0.0, SIZE
    )
    held = positions[:, 0] != 0
    vector_round_trips = int(np.sum(held[:-1] & ~held[1:]))
    vector_return = equity[-1] / CAPITAL - 1

    assert event_round_trips >= 3
    assert event_round_trips == vector_round_trips
    assert abs(event_return - vector_return) < 0.005
