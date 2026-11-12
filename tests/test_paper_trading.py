"""Paper trading handler behavior without Alpaca credentials."""

import os


class TestAlpacaIntegration:
    def test_alpaca_handler_initializes_without_credentials(self):
        """Should gracefully fall back to yfinance when no Alpaca credentials."""

        # Temporarily clear credentials
        orig_key = os.environ.pop("ALPACA_API_KEY", None)
        orig_secret = os.environ.pop("ALPACA_SECRET_KEY", None)

        try:
            from quantsim.core.event_queue import EventQueue
            from quantsim.paper_trading.alpaca_handler import AlpacaLiveDataHandler

            eq = EventQueue()
            # Should not raise; falls back to yfinance
            handler = AlpacaLiveDataHandler(
                asset_ids=["SPY"],
                event_queue=eq,
            )
            # Basic interface works
            assert handler.universe is not None
        finally:
            if orig_key:
                os.environ["ALPACA_API_KEY"] = orig_key
            if orig_secret:
                os.environ["ALPACA_SECRET_KEY"] = orig_secret

    def test_alpaca_paper_execution_simulates_without_credentials(self):
        """Should simulate fills locally when Alpaca not configured."""

        orig_key = os.environ.pop("ALPACA_API_KEY", None)

        try:
            from quantsim.core.event_queue import EventQueue
            from quantsim.core.events import AssetType, OrderEvent, OrderSide, OrderType
            from quantsim.paper_trading.alpaca_handler import AlpacaPaperExecutionHandler

            eq = EventQueue()
            handler = AlpacaPaperExecutionHandler(event_queue=eq)

            order = OrderEvent(
                asset_id="SPY",
                asset_type=AssetType.EQUITY,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                quantity=10,
                strategy_id="test",
                limit_price=400.0,
            )
            handler.execute_order(order)
            # Should have queued a fill
            assert not eq.empty()
        finally:
            if orig_key:
                os.environ["ALPACA_API_KEY"] = orig_key
