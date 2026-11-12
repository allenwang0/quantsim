"""
QuantSim: event-driven and vectorized daily-bar backtesting.
Zero paid dependencies. Event-driven + vectorized engines.

Quick start:
    from quantsim import BacktestEngine, VectorizedBacktester
    from quantsim import SMAcrossover, HierarchicalRiskParity
    from quantsim import StrategyRegistry, config
"""

from quantsim.backtesting.engine import BacktestEngine

# Order management
from quantsim.backtesting.order_manager import OrderManager
from quantsim.backtesting.vectorized import (
    VectorizedBacktester,
    momentum_signal,
    sma_crossover_signal,
)
from quantsim.backtesting.walk_forward import RegimeAnalyzer, WalkForwardOptimizer
from quantsim.core.config import config
from quantsim.core.database import init_db
from quantsim.core.database_v2 import init_full_db, migrate_v2
from quantsim.core.event_queue import EventQueue
from quantsim.core.events import BarEvent, Direction, FillEvent, OrderEvent, SignalEvent
from quantsim.data.data_handler import HistoricalDataHandler
from quantsim.data.ingestion import fetch_equity_history, fetch_fred_series, get_bars, get_universe
from quantsim.portfolio.manager import PortfolioManager
from quantsim.portfolio.optimization import (
    BlackLittermanOptimizer,
    HierarchicalRiskParity,
    MeanVarianceOptimizer,
    PortfolioOptimizationStrategy,
    RiskParityOptimizer,
)
from quantsim.portfolio.portfolio import Portfolio, Position
from quantsim.portfolio.sizing import (
    EqualWeightSizer,
    FixedFractionalSizer,
    KellySizer,
    VolatilityTargetSizer,
)
from quantsim.reporting.advanced import AdvancedAnalytics, generate_full_report
from quantsim.reporting.analytics import PerformanceAnalytics
from quantsim.reporting.monitor import StrategyHealthMonitor

# Reporting
from quantsim.reporting.tearsheet import generate_tearsheet
from quantsim.strategies.garch_vol import GARCHForecaster, GARCHVolatilityAdapter
from quantsim.strategies.mean_reversion import (
    BollingerBandMeanReversion,
    PairsTradingStrategy,
    RSIMeanReversion,
)
from quantsim.strategies.momentum_factor import (
    BuyAndHold,
    CrossSectionalMomentum,
    DualMomentum,
    LowVolatilityFactor,
)
from quantsim.strategies.registry import EnsembleEngine, SignalAggregator, StrategyRegistry
from quantsim.strategies.trend import (
    DonchianBreakout,
    EMACrossover,
    MACDStrategy,
    SMAcrossover,
    TimeSeriesMomentum,
)

__version__ = "3.0.0"
