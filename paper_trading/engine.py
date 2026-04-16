"""
Live Paper Trading Engine

The PaperEngine is structurally identical to BacktestEngine.
ONLY two components differ:
  - DataHandler: AlpacaLiveDataHandler (polls Alpaca) vs HistoricalDataHandler
  - ExecutionHandler: AlpacaPaperExecutionHandler (routes to Alpaca) vs BacktestExecutionHandler

Everything else is shared: strategies, portfolio, risk manager, portfolio manager.
This is the central design principle of the system.

Concurrent architecture (per spec Section 1):
  Process 1 (this file, asyncio):
    Coroutine A: market data polling (every 60s during hours)
    Coroutine B: fill polling (every 5s)
    Coroutine C: event loop dispatcher
    Coroutine D: risk monitor
  
  Process 2 (dashboard/app.py, Streamlit):
    Reads SQLite every 2s; never writes; never blocks trading

Communication: SQLite as shared state between processes.
"""

from __future__ import annotations
import asyncio
import logging
import signal
import time
from datetime import datetime, date, timezone
from typing import List, Optional, Dict
import pytz

from core.event_queue import EventQueue
from core.events import EventType, BarEvent, SignalEvent, OrderEvent, FillEvent, RiskEvent
from core.database import init_db
from core.config import config
from portfolio.portfolio import Portfolio
from portfolio.manager import PortfolioManager
from portfolio.sizing import EqualWeightSizer
from backtesting.order_manager import OrderManager
from reporting.monitor import StrategyHealthMonitor
from risk.risk_manager import RiskManager
from strategies.registry import EnsembleEngine, StrategyRegistry

logger = logging.getLogger(__name__)

ET_TZ = pytz.timezone("US/Eastern")


class MarketHoursChecker:
    """Checks if the market is currently open (NYSE regular hours)."""

    @staticmethod
    def is_market_open() -> bool:
        """Returns True if NYSE regular session is open."""
        try:
            import pandas_market_calendars as mcal
            nyse = mcal.get_calendar("NYSE")
            now_et = datetime.now(ET_TZ)
            today = now_et.date()
            schedule = nyse.schedule(
                start_date=str(today), end_date=str(today)
            )
            if schedule.empty:
                return False
            market_open = schedule.iloc[0]["market_open"].to_pydatetime()
            market_close = schedule.iloc[0]["market_close"].to_pydatetime()
            now_utc = datetime.now(timezone.utc)
            return market_open <= now_utc <= market_close
        except Exception:
            # Fallback: check if weekday between 9:30-16:00 ET
            now_et = datetime.now(ET_TZ)
            if now_et.weekday() >= 5:
                return False
            return (
                (now_et.hour == 9 and now_et.minute >= 30) or
                (10 <= now_et.hour <= 15) or
                (now_et.hour == 16 and now_et.minute == 0)
            )

    @staticmethod
    def minutes_to_open() -> int:
        """Returns minutes until next market open (0 if open now)."""
        if MarketHoursChecker.is_market_open():
            return 0
        now_et = datetime.now(ET_TZ)
        # Next 9:30 AM ET
        next_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        if now_et >= next_open:
            # Already past open today (market is closed), wait until tomorrow
            from datetime import timedelta
            next_open += timedelta(days=1)
            while next_open.weekday() >= 5:
                next_open += timedelta(days=1)
        return max(0, int((next_open - now_et).total_seconds() / 60))


class PaperEngine:
    """
    Live paper trading engine.
    
    Usage:
        engine = PaperEngine(strategies=[sma_strategy])
        asyncio.run(engine.run())
    
    Or from CLI:
        python scripts/run_paper_trading.py --strategy sma --symbol SPY
    
    The engine runs indefinitely until interrupted (Ctrl+C or SIGTERM).
    State is persisted to SQLite continuously.
    Dashboard runs as a separate process reading from the same DB.
    """

    def __init__(
        self,
        strategies: List,
        initial_capital: Optional[float] = None,
        use_ensemble: bool = False,
        ensemble_method: str = "confidence_weighted",
        db_path: Optional[str] = None,
    ):
        self.strategies = strategies
        self.initial_capital = initial_capital or config.initial_capital
        self.use_ensemble = use_ensemble
        self.db_path = db_path or config.db_path

        init_db(self.db_path)

        # Collect universe
        universe = list(set(
            asset_id
            for s in strategies
            for asset_id in s.asset_ids
        ))

        # Event queue
        self.event_queue = EventQueue()

        # Live data handler (Alpaca or yfinance fallback)
        from paper_trading.alpaca_handler import AlpacaLiveDataHandler
        self.data_handler = AlpacaLiveDataHandler(
            asset_ids=universe,
            event_queue=self.event_queue,
            db_path=self.db_path,
        )

        # Portfolio
        self.portfolio = Portfolio(
            initial_capital=self.initial_capital,
            cash=self.initial_capital,
            timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
        )

        # Risk manager
        self.risk_manager = RiskManager(
            event_queue=self.event_queue,
            max_drawdown_halt=config.backtest.max_drawdown_halt,
            max_drawdown_close=config.backtest.max_drawdown_close,
            db_path=self.db_path,
        )

        # Portfolio manager
        self.portfolio_manager = PortfolioManager(
            portfolio=self.portfolio,
            event_queue=self.event_queue,
            risk_manager=self.risk_manager,
            sizer=EqualWeightSizer(n_positions=max(len(universe), 5)),
            data_handler=self.data_handler,
            db_path=self.db_path,
        )

        # Execution handler (Alpaca paper)
        from paper_trading.alpaca_handler import AlpacaPaperExecutionHandler
        self.execution_handler = AlpacaPaperExecutionHandler(
            event_queue=self.event_queue,
            db_path=self.db_path,
        )

        # Order manager (stops, targets, trailing stops)
        self.order_manager = OrderManager(event_queue=self.event_queue)

        # Strategy health monitor
        self.health_monitor = StrategyHealthMonitor(
            db_path=self.db_path,
            max_dd_warn=config.backtest.max_drawdown_halt,
            max_dd_critical=config.backtest.max_drawdown_close,
        )

        # Optional: ensemble coordinator
        if use_ensemble and len(strategies) > 1:
            self.ensemble = EnsembleEngine(
                strategies=strategies,
                main_queue=self.event_queue,
                aggregation_method=ensemble_method,
            )
        else:
            self.ensemble = None
            for s in strategies:
                s._queue = self.event_queue

        # Counters
        self._bars_processed = 0
        self._fills_processed = 0
        self._running = False

        logger.info(
            f"PaperEngine initialized: {len(universe)} assets, "
            f"capital=${self.initial_capital:,.0f}, "
            f"Alpaca={'configured' if config.alpaca_configured else 'fallback'}"
        )

    async def _poll_market_data(self) -> None:
        """
        Coroutine A: Poll for new bars.
        During market hours: every 60s (sufficient for daily-bar strategies).
        Outside market hours: sleep until 5 minutes before open.
        """
        while self._running:
            if MarketHoursChecker.is_market_open():
                logger.debug("Polling market data...")
                try:
                    self.data_handler.poll_latest_bars()
                except Exception as e:
                    logger.error(f"Market data poll error: {e}")
                await asyncio.sleep(config.live.poll_interval_seconds)
            else:
                mins = MarketHoursChecker.minutes_to_open()
                sleep_secs = max(60, (mins - 5) * 60) if mins > 5 else 60
                logger.info(
                    f"Market closed. {mins}m to open. Sleeping {sleep_secs//60}m."
                )
                await asyncio.sleep(min(sleep_secs, 3600))

    async def _poll_fills(self) -> None:
        """
        Coroutine B: Poll for order fills from Alpaca.
        Runs every 5 seconds regardless of market hours.
        """
        while self._running:
            try:
                self.execution_handler.poll_fills()
            except Exception as e:
                logger.error(f"Fill poll error: {e}")
            await asyncio.sleep(config.live.fill_poll_seconds)

    async def _event_loop(self) -> None:
        """
        Coroutine C: Process events from the queue.
        """
        while self._running:
            events_processed = 0
            while not self.event_queue.empty() and events_processed < 100:
                event = self.event_queue.get()
                self._dispatch(event)
                events_processed += 1
            await asyncio.sleep(0.1)  # yield control

    async def _log_status(self) -> None:
        """
        Coroutine D: Log portfolio status periodically.
        """
        while self._running:
            # Run health checks
            try:
                self.health_monitor.check_all(self.portfolio, datetime.now(timezone.utc).replace(tzinfo=None))
            except Exception:
                pass
            equity = self.portfolio.total_equity
            dd = self.portfolio.current_drawdown
            n_pos = len(self.portfolio.positions)
            logger.info(
                f"STATUS | Equity: ${equity:,.0f} | "
                f"Drawdown: {dd:.1%} | Positions: {n_pos} | "
                f"Fills: {self._fills_processed}"
            )
            await asyncio.sleep(300)  # every 5 minutes

    def _dispatch(self, event) -> None:
        if event.event_type == EventType.BAR:
            self._handle_bar(event)
        elif event.event_type == EventType.SIGNAL:
            self._handle_signal(event)
        elif event.event_type == EventType.ORDER:
            self._handle_order(event)
        elif event.event_type == EventType.FILL:
            self._handle_fill(event)
        elif event.event_type == EventType.RISK:
            self._handle_risk(event)

    def _handle_bar(self, event: BarEvent) -> None:
        self._bars_processed += 1
        self.portfolio_manager.on_bar(event)

        # Stop-loss / take-profit checks
        self.order_manager.on_bar(event)

        # Options expiration
        self.portfolio.expire_options(as_of_date=event.timestamp)

        if self.ensemble:
            self.ensemble.on_bar(event, self.data_handler)
        else:
            for strategy in self.strategies:
                if event.asset_id in strategy.asset_ids:
                    try:
                        strategy.on_bar(event, self.data_handler)
                    except Exception as e:
                        logger.error(f"Strategy error: {e}")

    def _handle_signal(self, event: SignalEvent) -> None:
        logger.info(
            f"SIGNAL: {event.strategy_id} | {event.direction.value} "
            f"{event.asset_id} | conf={event.confidence:.2f}"
        )
        self.portfolio_manager.on_signal(event)

    def _handle_order(self, event: OrderEvent) -> None:
        self.execution_handler.execute_order(event)

    def _handle_fill(self, event: FillEvent) -> None:
        self._fills_processed += 1
        self.portfolio_manager.on_fill(event)
        logger.info(
            f"FILL: {event.side.value} {event.quantity} {event.asset_id} "
            f"@ ${event.fill_price:.2f}"
        )

    def _handle_risk(self, event: RiskEvent) -> None:
        logger.warning(f"RISK: {event.risk_type.value} | {event.message}")

    async def run(self) -> None:
        """
        Start the paper trading engine.
        Runs until interrupted.
        """
        self._running = True
        logger.info("PaperEngine starting...")

        # Validate config
        warnings = config.validate()
        for w in warnings:
            logger.warning(f"Config: {w}")

        # Register shutdown handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._shutdown)

        # Run all coroutines concurrently
        await asyncio.gather(
            self._poll_market_data(),
            self._poll_fills(),
            self._event_loop(),
            self._log_status(),
        )

    def _shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down PaperEngine...")
        self._running = False

        # Persist final state
        self.portfolio.persist(self.db_path)

        # Log final summary
        logger.info(
            f"Final equity: ${self.portfolio.total_equity:,.0f} | "
            f"Realized P&L: ${self.portfolio.realized_pnl:,.0f} | "
            f"Total fills: {self._fills_processed}"
        )
