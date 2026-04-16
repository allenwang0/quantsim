"""
QuantSim - Production-grade quantitative trading simulator.
Zero paid dependencies. Event-driven + vectorized engines.

Quick start:
    from quantsim import BacktestEngine, VectorizedBacktester
    from quantsim import SMAcrossover, HierarchicalRiskParity
    from quantsim import StrategyRegistry, config
"""

from backtesting.engine import BacktestEngine
from backtesting.vectorized import VectorizedBacktester, sma_crossover_signal, momentum_signal
from backtesting.walk_forward import WalkForwardOptimizer, RegimeAnalyzer

from strategies.trend import SMAcrossover, EMACrossover, MACDStrategy, DonchianBreakout, TimeSeriesMomentum
from strategies.mean_reversion import BollingerBandMeanReversion, RSIMeanReversion, PairsTradingStrategy
from strategies.momentum_factor import CrossSectionalMomentum, DualMomentum, LowVolatilityFactor, BuyAndHold
from strategies.registry import StrategyRegistry, SignalAggregator, EnsembleEngine
from strategies.garch_vol import GARCHForecaster, GARCHVolatilityAdapter

from portfolio.portfolio import Portfolio, Position
from portfolio.optimization import (
    HierarchicalRiskParity, RiskParityOptimizer,
    MeanVarianceOptimizer, BlackLittermanOptimizer,
    PortfolioOptimizationStrategy,
)
from portfolio.sizing import FixedFractionalSizer, VolatilityTargetSizer, KellySizer, EqualWeightSizer
from portfolio.manager import PortfolioManager

from data.ingestion import fetch_equity_history, fetch_fred_series, get_bars, get_universe
from data.options import reconstruct_chain, compute_greeks, compute_iv_rank
from data.data_handler import HistoricalDataHandler

from reporting.analytics import PerformanceAnalytics
from reporting.advanced import AdvancedAnalytics, generate_full_report

from core.events import BarEvent, SignalEvent, OrderEvent, FillEvent, Direction
from core.event_queue import EventQueue
from core.config import config
from core.database import init_db
from core.database_v2 import migrate_v2, init_full_db

# Order management
from backtesting.order_manager import OrderManager

# Reporting
from reporting.tearsheet import generate_tearsheet
from reporting.monitor import StrategyHealthMonitor
from reporting.advanced import generate_full_report

__version__ = "2.0.0"
