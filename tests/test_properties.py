"""
Accounting and optimizer invariants checked on generated inputs (hypothesis).

If any of these break, every equity curve and report built on them is wrong:
equity must equal cash plus marked positions and reconcile with P&L, fills
must land inside the bar that produced them, and long-only optimizer
weights must be a valid allocation.
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from quantsim.backtesting.execution import FillModel
from quantsim.core.events import AssetType, FillEvent, OrderEvent, OrderSide
from quantsim.portfolio.optimization import HierarchicalRiskParity, RiskParityOptimizer
from quantsim.portfolio.portfolio import Portfolio

trade = st.tuples(
    st.sampled_from([OrderSide.BUY, OrderSide.SELL]),
    st.integers(1, 500),
    st.floats(1.0, 500.0),
)


@settings(max_examples=300, deadline=None)
@given(trades=st.lists(trade, min_size=1, max_size=30), mark=st.floats(1.0, 500.0))
def test_equity_reconciles_with_cash_positions_and_pnl(trades, mark):
    portfolio = Portfolio(initial_capital=1_000_000.0, cash=1_000_000.0)
    for side, quantity, price in trades:
        portfolio.apply_fill(
            FillEvent(
                asset_id="SYN",
                asset_type=AssetType.EQUITY,
                side=side,
                quantity=quantity,
                fill_price=price,
                commission=0.0,
            )
        )
    portfolio.update_market_prices({"SYN": mark})

    marked = sum(p.quantity * p.current_price for p in portfolio.positions.values())
    assert np.isclose(portfolio.total_equity, portfolio.cash + marked)
    assert np.isclose(
        portfolio.total_equity,
        portfolio.initial_capital + portfolio.realized_pnl + portfolio.unrealized_pnl,
        rtol=1e-9,
    )


bar = st.tuples(
    st.floats(1.0, 1000.0), st.floats(0.0, 0.2), st.floats(0.0, 1.0), st.floats(0.0, 1.0)
)


@settings(max_examples=300, deadline=None)
@given(
    bar=bar,
    mode=st.sampled_from([FillModel.NEXT_BAR_OPEN, FillModel.VWAP, FillModel.IMMEDIATE_CLOSE]),
    side=st.sampled_from([OrderSide.BUY, OrderSide.SELL]),
)
def test_fill_price_lies_inside_the_fill_bar(bar, mode, side):
    low, spread, open_pos, close_pos = bar
    high = low * (1 + spread)
    ohlc = {
        "open": low + open_pos * (high - low),
        "high": high,
        "low": low,
        "close": low + close_pos * (high - low),
        "volume": 1_000_000,
    }
    order = OrderEvent(asset_id="SYN", side=side, quantity=10)

    fill = FillModel(mode).compute_fill(order, current_bar=ohlc, next_bar=ohlc)

    assert low <= fill["price"] <= high


@settings(max_examples=40, deadline=None)
@given(n_assets=st.integers(2, 8), seed=st.integers(0, 10_000))
def test_long_only_optimizer_weights_are_an_allocation(n_assets, seed):
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.005, 0.03, n_assets)
    returns = pd.DataFrame(
        rng.normal(0, 1, (300, n_assets)) * vols, columns=[f"A{i}" for i in range(n_assets)]
    )

    for optimizer in (HierarchicalRiskParity(), RiskParityOptimizer()):
        weights = np.array(list(optimizer.optimize(returns).values()))
        assert np.all(weights >= -1e-12)
        assert abs(weights.sum() - 1.0) < 1e-9
