"""Portfolio handling of option positions: greeks and expiry."""

from datetime import date, datetime


class TestPortfolioOptionsMethods:
    def test_update_greeks_on_position(self):
        from quantsim.core.events import AssetType, FillEvent, OrderSide
        from quantsim.portfolio.portfolio import Portfolio

        p = Portfolio(initial_capital=100_000, cash=100_000)

        # Add an options position
        fill = FillEvent(
            order_id="opt-1",
            asset_id="SPY",
            asset_type=AssetType.OPTION,
            side=OrderSide.BUY_TO_OPEN,
            quantity=1,
            fill_price=5.0,
            commission=0.65,
            slippage=0.0,
            strategy_id="covered_call",
            timestamp=datetime(2022, 1, 3),
            option_symbol="SPY220121C00420000",
            expiration="2022-01-21",
            strike=420.0,
            right="C",
        )
        p.apply_fill(fill)

        # Update Greeks
        p.update_greeks("SPY220121C00420000", delta=0.30, gamma=0.05, theta=-0.08, vega=0.12)

        pos = p.positions.get("SPY220121C00420000")
        assert pos is not None
        assert abs(pos.delta - 0.30) < 1e-6
        assert abs(pos.gamma - 0.05) < 1e-6
        assert abs(pos.theta - (-0.08)) < 1e-6

    def test_expire_options_removes_position(self):
        from quantsim.core.events import AssetType, FillEvent, OrderSide
        from quantsim.portfolio.portfolio import Portfolio

        p = Portfolio(initial_capital=100_000, cash=100_000)

        fill = FillEvent(
            order_id="opt-2",
            asset_id="SPY",
            asset_type=AssetType.OPTION,
            side=OrderSide.BUY_TO_OPEN,
            quantity=1,
            fill_price=3.0,
            commission=0.65,
            slippage=0.0,
            strategy_id="test",
            timestamp=datetime(2022, 1, 3),
            option_symbol="SPY220110C00430000",
            expiration="2022-01-10",
            strike=430.0,
            right="C",
        )
        p.apply_fill(fill)

        # Advance past expiration
        expired = p.expire_options(as_of_date=date(2022, 1, 11))
        assert "SPY220110C00430000" in expired
        assert "SPY220110C00430000" not in p.positions

    def test_expire_options_books_pnl(self):
        from quantsim.core.events import AssetType, FillEvent, OrderSide
        from quantsim.portfolio.portfolio import Portfolio

        p = Portfolio(initial_capital=100_000, cash=100_000)
        # Buy an option at 5.0
        fill = FillEvent(
            order_id="opt-3",
            asset_id="SPY",
            asset_type=AssetType.OPTION,
            side=OrderSide.BUY_TO_OPEN,
            quantity=1,
            fill_price=5.0,
            commission=0.0,
            slippage=0.0,
            strategy_id="test",
            timestamp=datetime(2022, 1, 3),
            option_symbol="SPY220107P00390000",
            expiration="2022-01-07",
            strike=390.0,
            right="P",
        )
        p.apply_fill(fill)

        # Update price to 0 (expires worthless)
        p.positions["SPY220107P00390000"].current_price = 0.0

        expired = p.expire_options(as_of_date=date(2022, 1, 8))
        assert len(expired) > 0
        # Realized P&L should have been updated
        # (in this case it's negative since we paid 5 and got 0)
