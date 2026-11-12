"""
QuantSim: event-driven and vectorized daily-bar backtesting.
Zero paid dependencies. Event-driven + vectorized engines.

Quick start:
    from quantsim import BacktestEngine, VectorizedBacktester
    from quantsim import SMAcrossover, HierarchicalRiskParity
    from quantsim import StrategyRegistry, config
"""

from quantsim.backtesting.engine import BacktestEngine
from quantsim.backtesting.vectorized import VectorizedBacktester, sma_crossover_signal, momentum_signal
from quantsim.backtesting.walk_forward import WalkForwardOptimizer, RegimeAnalyzer

from quantsim.strategies.trend import SMAcrossover, EMACrossover, MACDStrategy, DonchianBreakout, TimeSeriesMomentum
from quantsim.strategies.mean_reversion import BollingerBandMeanReversion, RSIMeanReversion, PairsTradingStrategy
from quantsim.strategies.momentum_factor import CrossSectionalMomentum, DualMomentum, LowVolatilityFactor, BuyAndHold
from quantsim.strategies.registry import StrategyRegistry, SignalAggregator, EnsembleEngine
from quantsim.strategies.garch_vol import GARCHForecaster, GARCHVolatilityAdapter

from quantsim.portfolio.portfolio import Portfolio, Position
from quantsim.portfolio.optimization import (
    HierarchicalRiskParity, RiskParityOptimizer,
    MeanVarianceOptimizer, BlackLittermanOptimizer,
    PortfolioOptimizationStrategy,
)
from quantsim.portfolio.sizing import FixedFractionalSizer, VolatilityTargetSizer, KellySizer, EqualWeightSizer
from quantsim.portfolio.manager import PortfolioManager

from quantsim.data.ingestion import fetch_equity_history, fetch_fred_series, get_bars, get_universe
from quantsim.data.options import reconstruct_chain, compute_greeks, compute_iv_rank
from quantsim.data.data_handler import HistoricalDataHandler

from quantsim.reporting.analytics import PerformanceAnalytics
from quantsim.reporting.advanced import AdvancedAnalytics, generate_full_report

from quantsim.core.events import BarEvent, SignalEvent, OrderEvent, FillEvent, Direction
from quantsim.core.event_queue import EventQueue
from quantsim.core.config import config
from quantsim.core.database import init_db
from quantsim.core.database_v2 import migrate_v2, init_full_db

# Order management
from quantsim.backtesting.order_manager import OrderManager

# Reporting
from quantsim.reporting.tearsheet import generate_tearsheet
from quantsim.reporting.monitor import StrategyHealthMonitor
from quantsim.reporting.advanced import generate_full_report

__version__ = "3.0.0"
