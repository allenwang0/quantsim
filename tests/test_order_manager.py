"""Stop-loss, take-profit and trailing-stop orders."""

from datetime import datetime


class TestOrderManager:
    def make_bar(self, asset_id, price, high=None, low=None, dt=None):
        from quantsim.core.events import BarEvent

        return BarEvent(
            timestamp=dt or datetime(2022, 1, 3),
            asset_id=asset_id,
            open=price * 0.999,
            high=high or price * 1.01,
            low=low or price * 0.99,
            close=price,
            volume=1_000_000,
            adj_close=price,
        )

    def test_stop_loss_triggers_below_price(self):
        from quantsim.backtesting.order_manager import OrderManager
        from quantsim.core.event_queue import EventQueue
        from quantsim.core.events import EventType

        eq = EventQueue()
        mgr = OrderManager(event_queue=eq)
        mgr.add_stop_loss(
            "SPY", entry_price=400.0, strategy_id="sma", stop_pct=0.02, direction="LONG"
        )

        # Bar where low goes below stop (400 * 0.98 = 392)
        bar = self.make_bar("SPY", price=391, low=389)
        triggered = mgr.on_bar(bar)
        assert "SPY" in triggered

        # Should have emitted a FLAT signal
        assert not eq.empty()
        event = eq.get()
        assert event.event_type == EventType.SIGNAL
        from quantsim.core.events import Direction

        assert event.direction == Direction.FLAT

    def test_stop_loss_does_not_trigger_above_price(self):
        from quantsim.backtesting.order_manager import OrderManager
        from quantsim.core.event_queue import EventQueue

        eq = EventQueue()
        mgr = OrderManager(event_queue=eq)
        mgr.add_stop_loss(
            "SPY", entry_price=400.0, strategy_id="sma", stop_pct=0.02, direction="LONG"
        )

        # Bar above stop
        bar = self.make_bar("SPY", price=405, low=402)
        triggered = mgr.on_bar(bar)
        assert "SPY" not in triggered
        assert eq.empty()

    def test_take_profit_triggers_above_target(self):
        from quantsim.backtesting.order_manager import OrderManager
        from quantsim.core.event_queue import EventQueue

        eq = EventQueue()
        mgr = OrderManager(event_queue=eq)
        mgr.add_take_profit(
            "SPY", entry_price=400.0, strategy_id="sma", target_pct=0.05, direction="LONG"
        )

        # Bar where high goes above target (400 * 1.05 = 420)
        bar = self.make_bar("SPY", price=422, high=425)
        triggered = mgr.on_bar(bar)
        assert "SPY" in triggered

    def test_trailing_stop_moves_with_price(self):
        from quantsim.backtesting.order_manager import OrderManager
        from quantsim.core.event_queue import EventQueue

        eq = EventQueue()
        mgr = OrderManager(event_queue=eq)
        mgr.add_trailing_stop(
            "SPY", entry_price=400.0, strategy_id="sma", trail_pct=0.03, direction="LONG"
        )

        # Price rises to 420 — stop should follow to 420 * 0.97 = 407.4
        bar_up = self.make_bar("SPY", price=420, high=420, low=418)
        mgr.on_bar(bar_up)

        stop = mgr._stops.get("SPY")
        assert stop is not None
        assert stop.stop_price > 400 * 0.97  # stop moved up

    def test_bracket_order_stop_triggers_cancels_target(self):
        from quantsim.backtesting.order_manager import OrderManager
        from quantsim.core.event_queue import EventQueue

        eq = EventQueue()
        mgr = OrderManager(event_queue=eq)
        mgr.add_bracket("SPY", entry_price=400.0, strategy_id="sma", stop_pct=0.02, target_pct=0.06)

        # Stop triggers
        bar = self.make_bar("SPY", price=391, low=389)
        triggered = mgr.on_bar(bar)
        assert "SPY" in triggered
        # Both stop and target should be removed
        assert "SPY" not in mgr._stops

    def test_atr_stop_computed_from_atr(self):
        from quantsim.backtesting.order_manager import OrderManager
        from quantsim.core.event_queue import EventQueue

        eq = EventQueue()
        mgr = OrderManager(event_queue=eq)
        atr = 5.0  # $5 ATR
        mgr.add_atr_stop(
            "SPY",
            entry_price=400.0,
            strategy_id="sma",
            current_atr=atr,
            multiple=2.0,
            direction="LONG",
        )

        stop = mgr._stops["SPY"]
        # Stop = 400 - 2 * 5 = 390
        assert abs(stop.stop_price - 390.0) < 0.01

    def test_remove_clears_stop(self):
        from quantsim.backtesting.order_manager import OrderManager
        from quantsim.core.event_queue import EventQueue

        eq = EventQueue()
        mgr = OrderManager(event_queue=eq)
        mgr.add_stop_loss("SPY", entry_price=400.0, strategy_id="sma")
        mgr.remove("SPY")
        assert "SPY" not in mgr._stops

    def test_short_position_stop_triggers_above(self):
        from quantsim.backtesting.order_manager import OrderManager
        from quantsim.core.event_queue import EventQueue

        eq = EventQueue()
        mgr = OrderManager(event_queue=eq)
        mgr.add_stop_loss(
            "SPY", entry_price=400.0, strategy_id="sma", stop_pct=0.02, direction="SHORT"
        )

        # Short position: stop above entry (400 * 1.02 = 408)
        bar = self.make_bar("SPY", price=410, high=412)
        triggered = mgr.on_bar(bar)
        assert "SPY" in triggered
