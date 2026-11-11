"""
A signal computed from bar T's close must not trade at a price from bar T.

The engine used to fill orders inside the bar that produced the signal, at
that bar's open: a price that was already in the past when the signal was
known. These tests use synthetic bars only.
"""
from datetime import datetime

import numpy as np
import pytest

from backtesting.engine import BacktestEngine
from backtesting.execution import FillModel, NoSlippage, ZeroCommission
from core.event_queue import EventQueue
from core.events import Direction
from strategies.momentum_factor import BuyAndHold
from strategies.trend import Strategy
from tests.synthetic_bars import trading_days, write_bars


class BuyOnBar(Strategy):
    """Goes long on the close of one chosen bar (0-based)."""

    def __init__(self, asset_ids, event_queue, signal_bar):
        super().__init__("BuyOnBar", asset_ids, event_queue, warmup_bars=0)
        self.signal_bar = signal_bar
        self.seen = 0

    def on_bar(self, event, data_handler):
        if self.seen == self.signal_bar:
            self._emit_signal(event.asset_id, Direction.LONG, event.timestamp)
        self.seen += 1


class KnowsTodaysClose(Strategy):
    """
    Long when bar T closed above its open, flat otherwise. Under same-bar
    fills at T's open this pockets every up day; filled at T+1's open it has
    no information about the next bar of a random walk.
    """

    def __init__(self, asset_ids, event_queue):
        super().__init__("KnowsTodaysClose", asset_ids, event_queue, warmup_bars=0)
        self.long = False

    def on_bar(self, event, data_handler):
        want_long = event.close > event.open
        if want_long != self.long:
            direction = Direction.LONG if want_long else Direction.FLAT
            self._emit_signal(event.asset_id, direction, event.timestamp)
            self.long = want_long


def run_engine(db_path, strategy_cls, dates, **kwargs):
    strategy = strategy_cls(["SYN"], EventQueue(), **kwargs)
    fills = []
    engine = BacktestEngine(
        strategies=[strategy],
        start=dates[0].to_pydatetime(),
        end=dates[-1].to_pydatetime(),
        initial_capital=100_000,
        slippage_model=NoSlippage(),
        commission_model=ZeroCommission(),
        fill_mode=FillModel.NEXT_BAR_OPEN,
        warmup_bars=0,
        db_path=db_path,
        verbose=False,
    )
    original = engine._handle_fill

    def record(event):
        fills.append(event)
        original(event)

    engine._handle_fill = record
    engine.run()
    return engine, fills


def test_signal_on_bar_2_fills_at_bar_3_open(tmp_path):
    dates = trading_days(5)
    closes = [100.0, 100.0, 110.0, 111.0, 112.0]
    opens = [100.0, 100.0, 110.0, 110.5, 111.5]
    db_path = str(tmp_path / "five.db")
    write_bars(db_path, "SYN", dates, opens, closes)

    _, fills = run_engine(db_path, BuyOnBar, dates, signal_bar=1)

    assert len(fills) == 1
    assert fills[0].fill_price == pytest.approx(opens[2] + fills[0].slippage)
    assert fills[0].timestamp.date() == dates[2].date()


def test_knowing_todays_close_gives_no_edge_on_random_walks(tmp_path):
    """
    Over 50 seeded random walks the close-peeking strategy's return minus
    buy-and-hold's must stay within 3 standard errors of zero (costs are zero
    here). With same-bar fills its mean excess return is many SEs positive.
    """
    n_bars, n_walks = 60, 50
    dates = trading_days(n_bars)
    excess = []
    for seed in range(n_walks):
        rng = np.random.default_rng(seed)
        gaps = rng.normal(0, 0.01, n_bars)
        intraday = rng.normal(0, 0.01, n_bars)
        opens, closes = [], []
        prev_close = 100.0
        for g, d in zip(gaps, intraday):
            o = prev_close * np.exp(g)
            c = o * np.exp(d)
            opens.append(o)
            closes.append(c)
            prev_close = c

        db_path = str(tmp_path / f"walk{seed}.db")
        write_bars(db_path, "SYN", dates, opens, closes)
        peek, _ = run_engine(db_path, KnowsTodaysClose, dates)
        hold, _ = run_engine(db_path, BuyAndHold, dates)
        excess.append(
            peek.portfolio.total_equity / 100_000 - hold.portfolio.total_equity / 100_000
        )

    excess = np.array(excess)
    standard_error = excess.std(ddof=1) / np.sqrt(n_walks)
    assert excess.mean() <= 3 * standard_error
