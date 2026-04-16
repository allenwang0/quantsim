"""
BacktestEngine: the main event loop.

Wires together DataHandler → StrategyEngine → PortfolioManager →
ExecutionHandler → ReportingEngine.

The engine is structurally identical to PaperEngine.
Only DataHandler and ExecutionHandler differ between modes.
"""

from __future__ import annotations
import logging
import time
from datetime import datetime
from typing import List, Optional, Dict, Type

from core.event_queue import EventQueue
from core.events import (
    EventType, BarEvent, SignalEvent, OrderEvent, FillEvent, RiskEvent
)
from core.database import init_db, DB_PATH
from data.data_handler import HistoricalDataHandler
from portfolio.portfolio import Portfolio
from portfolio.manager import PortfolioManager
from portfolio.sizing import PositionSizer, EqualWeightSizer, VolatilityTargetSizer
from risk.risk_manager import RiskManager
from backtesting.execution import (
    BacktestExecutionHandler, VolumeProportionalSlippage,
    ZeroCommission, PerShareCommission, FillModel
)
from strategies.trend import Strategy
from reporting.analytics import PerformanceAnalytics, load_equity_curve_from_db, load_trades_from_db
from backtesting.order_manager import OrderManager
from strategies.garch_vol import GARCHVolatilityAdapter
from strategies.registry import EnsembleEngine

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Event-driven backtesting engine.
    
    Architecture:
      DataHandler fires BarEvents → Strategies process BarEvents → emit SignalEvents
      → PortfolioManager converts to OrderEvents → ExecutionHandler simulates fills
      → FillEvents update Portfolio → ReportingEngine logs everything
    
    The loop processes events in strict chronological order with FillEvents
    having priority over BarEvents at the same timestamp.
    
    Walking skeleton validation: run BuyAndHold on SPY, expect total return
    within 0.1% annually of published SPY total return.
    """

    def __init__(
        self,
        strategies: List[Strategy],
        start: datetime,
        end: datetime,
        initial_capital: float = 100_000.0,
        sizer: Optional[PositionSizer] = None,
        slippage_model=None,
        commission_model=None,
        fill_mode: str = FillModel.NEXT_BAR_OPEN,
        warmup_bars: int = 252,
        db_path: str = DB_PATH,
        max_drawdown_halt: float = -0.15,
        max_drawdown_close: float = -0.20,
        verbose: bool = True,
        use_ensemble: bool = False,
        ensemble_method: str = "confidence_weighted",
    ):
        self.strategies = strategies
        self.start = start
        self.end = end
        self.initial_capital = initial_capital
        self.db_path = db_path
        self.verbose = verbose

        # Initialize database
        init_db(db_path)

        # Collect universe from all strategies
        universe = list(set(
            asset_id
            for strategy in strategies
            for asset_id in strategy.asset_ids
        ))

        # Event queue
        self.event_queue = EventQueue()

        # Data handler
        self.data_handler = HistoricalDataHandler(
            asset_ids=universe,
            start=start,
            end=end,
            event_queue=self.event_queue,
            warmup_bars=warmup_bars,
            db_path=db_path,
        )

        # CRITICAL: Rewire all strategy queues to the engine's event queue.
        # Strategies are constructed with a user-provided queue, but the engine
        # creates its own queue. Without this, signals go to the wrong queue.
        for strategy in strategies:
            strategy._queue = self.event_queue

        # Portfolio
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            cash=initial_capital,
            timestamp=start,
        )

        # Risk manager
        self.risk_manager = RiskManager(
            event_queue=self.event_queue,
            max_drawdown_halt=max_drawdown_halt,
            max_drawdown_close=max_drawdown_close,
            db_path=db_path,
        )

        # Portfolio manager
        self.portfolio_manager = PortfolioManager(
            portfolio=self.portfolio,
            event_queue=self.event_queue,
            risk_manager=self.risk_manager,
            sizer=sizer or EqualWeightSizer(n_positions=max(len(universe), 5)),
            data_handler=self.data_handler,
            db_path=db_path,
        )

        # Execution handler
        self.execution_handler = BacktestExecutionHandler(
            data_handler=self.data_handler,
            event_queue=self.event_queue,
            slippage_model=slippage_model or VolumeProportionalSlippage(),
            commission_model=commission_model or ZeroCommission(),
            fill_model=FillModel(fill_mode),
        )

        # Order manager (stop-loss, take-profit, trailing stops)
        self.order_manager = OrderManager(event_queue=self.event_queue)

        # GARCH volatility adapter (feeds VolatilityTargetSizer)
        self.garch_adapter = GARCHVolatilityAdapter(
            fit_window=252, refit_every=21
        )

        # Ensemble support: if use_ensemble=True, wrap strategies
        if use_ensemble and len(strategies) > 1:
            self.ensemble = EnsembleEngine(
                strategies=strategies,
                main_queue=self.event_queue,
                aggregation_method=ensemble_method,
            )
            self._use_ensemble = True
        else:
            self.ensemble = None
            self._use_ensemble = False

        # Analytics
        self.analytics = PerformanceAnalytics()

        # Counters
        self._bars_processed = 0
        self._signals_processed = 0
        self._orders_processed = 0
        self._fills_processed = 0

        logger.info(
            f"BacktestEngine initialized: {len(universe)} assets, "
            f"{start.date()} → {end.date()}, capital=${initial_capital:,.0f}"
        )

    def run(self) -> Dict:
        """
        Execute the backtest. Returns performance analytics dict.
        
        Event loop:
        1. Advance DataHandler one bar (fires BarEvents)
        2. Process all queued events (signals → orders → fills)
        3. Repeat until no more data
        """
        logger.info("Backtest starting...")
        t_start = time.time()

        while self.data_handler.has_more_data():
            # Advance one trading day; fires BarEvents into the queue
            self.data_handler.update_bars()

            # Process all events generated by this bar
            while not self.event_queue.empty():
                event = self.event_queue.get()
                self._dispatch(event)

        # Persist final portfolio state to DB
        self.portfolio.persist(self.db_path)

        elapsed = time.time() - t_start
        logger.info(
            f"Backtest complete in {elapsed:.1f}s: "
            f"{self._bars_processed} bars, "
            f"{self._signals_processed} signals, "
            f"{self._fills_processed} fills"
        )

        # Compute and return analytics
        return self._compute_results()

    def _dispatch(self, event) -> None:
        """Route event to the appropriate handler."""
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

        # Update portfolio market prices
        self.portfolio_manager.on_bar(event)

        # Persist equity snapshot every 21 bars (monthly frequency)
        if self._bars_processed % 21 == 0:
            self._write_equity_snapshot(event.timestamp)

        # Update GARCH vol model with latest return
        prev_bars = self.data_handler.get_latest_bars(event.asset_id, n=2)
        if not prev_bars.empty and len(prev_bars) >= 2:
            col = "adj_close" if "adj_close" in prev_bars.columns else "close"
            prev_close = float(prev_bars[col].iloc[-2])
            if prev_close > 0 and event.close > 0:
                ret = (event.close / prev_close) - 1
                self.garch_adapter.add_bar(event.asset_id, ret)

        # Check stop-loss / take-profit levels
        self.order_manager.on_bar(event)

        # Expire options past their expiration date
        self.portfolio.expire_options(as_of_date=event.timestamp)

        # Only fire strategy logic after warmup
        if not self.data_handler.is_warmup_complete():
            return

        # Route to ensemble or individual strategies
        if self._use_ensemble and self.ensemble:
            self.ensemble.on_bar(event, self.data_handler)
        else:
            for strategy in self.strategies:
                if event.asset_id in strategy.asset_ids:
                    try:
                        strategy.on_bar(event, self.data_handler)
                    except Exception as e:
                        logger.error(f"Strategy {strategy.strategy_id} error on {event.asset_id}: {e}")

    def _write_equity_snapshot(self, timestamp) -> None:
        """Write an equity snapshot to the DB. Called every 21 bars so
        load_equity_curve_from_db() has enough resolution for analytics."""
        try:
            ts_epoch = int(timestamp.timestamp()) if hasattr(timestamp, 'timestamp') else int(timestamp)
            equity = self.portfolio.total_equity
            cash = self.portfolio.cash
            rpnl = self.portfolio.realized_pnl
            upnl = self.portfolio.unrealized_pnl
            payload = '{}'
            from core.database import db_conn
            with db_conn(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO portfolio_snapshots "
                    "(timestamp, total_equity, cash, realized_pnl, unrealized_pnl, payload) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (ts_epoch, equity, cash, rpnl, upnl, payload)
                )
        except Exception:
            pass  # Never let DB writes crash the backtest

    def _handle_signal(self, event: SignalEvent) -> None:
        self._signals_processed += 1
        if self.verbose:
            logger.debug(
                f"SIGNAL: {event.strategy_id} {event.direction.value} "
                f"{event.asset_id} conf={event.confidence:.2f}"
            )
        self.portfolio_manager.on_signal(event)

    def _handle_order(self, event: OrderEvent) -> None:
        self._orders_processed += 1
        self.execution_handler.execute_order(event)

    def _handle_fill(self, event: FillEvent) -> None:
        self._fills_processed += 1
        self.portfolio_manager.on_fill(event)

        if self.verbose:
            logger.info(
                f"FILL: {event.side.value} {event.quantity} {event.asset_id} "
                f"@ ${event.fill_price:.2f} | equity=${self.portfolio.total_equity:,.0f}"
            )

    def _handle_risk(self, event: RiskEvent) -> None:
        logger.warning(f"RISK EVENT: {event.risk_type.value} - {event.message}")

    def _compute_results(self) -> Dict:
        """Build the full performance analytics report."""
        equity_curve = load_equity_curve_from_db(self.db_path)

        # Treat DB curve as insufficient if it has <= 1 data point
        # OR if all values are identical (stale/broken snapshots)
        if len(equity_curve) <= 1 or equity_curve.std() == 0:
            equity_curve = pd.Series(dtype=float)

        if equity_curve.empty:
            # Build from portfolio history if DB snapshots are thin
            if self.portfolio._equity_history:
                timestamps, equities = zip(*self.portfolio._equity_history)
                # Normalize all timestamps to naive UTC for consistent indexing
                clean_ts = []
                for t in timestamps:
                    if hasattr(t, 'tzinfo') and t.tzinfo is not None:
                        t = t.replace(tzinfo=None)
                    clean_ts.append(t)
                equity_curve = pd.Series(
                    list(equities),
                    index=pd.DatetimeIndex(clean_ts),
                )
                # Deduplicate: keep last value per timestamp
                equity_curve = equity_curve[~equity_curve.index.duplicated(keep='last')]
                equity_curve = equity_curve.sort_index()

        trades_df = load_trades_from_db(self.db_path)

        if equity_curve.empty:
            logger.warning("No equity curve data. Did fills occur?")
            results = {
                "final_equity": self.portfolio.total_equity,
                "initial_equity": self.initial_capital,
                "total_return": (self.portfolio.total_equity / self.initial_capital) - 1,
                "n_fills": self._fills_processed,
                "warning": "No equity curve data; run with a longer period",
            }
        else:
            results = self.analytics.compute_all(
                equity_curve=equity_curve,
                trades=trades_df if not trades_df.empty else None,
            )

        results["_meta"] = {
            "start": str(self.start.date()),
            "end": str(self.end.date()),
            "initial_capital": self.initial_capital,
            "final_equity": self.portfolio.total_equity,
            "bars_processed": self._bars_processed,
            "fills_executed": self._fills_processed,
            "universe_size": len(self.data_handler.universe),
        }

        if self.verbose:
            self.analytics.print_summary(results)

        return results


# Need pandas for equity curve fallback
import pandas as pd
