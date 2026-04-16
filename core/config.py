"""
Configuration system for QuantSim.

Design: environment-variable-first, file-based fallback, sensible defaults.
A single Config object is imported everywhere; never hardcode a constant
that a user might want to change.

Usage:
    from core.config import config
    
    print(config.initial_capital)
    print(config.db_path)
    config.slippage_model = "fixed"

Override via environment variables:
    QUANTSIM_DB=/path/to/db.db
    QUANTSIM_CAPITAL=500000
    QUANTSIM_SLIPPAGE=volume
    ALPACA_API_KEY=...
    ALPACA_SECRET_KEY=...
    FRED_API_KEY=...

Override via config file (quantsim.toml or quantsim.json):
    [backtesting]
    initial_capital = 250000
    slippage_model = "volume"
"""

from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

CONFIG_FILE_PATHS = [
    Path("quantsim.json"),
    Path("quantsim.toml"),
    Path.home() / ".quantsim" / "config.json",
]


@dataclass
class DatabaseConfig:
    path: str = field(
        default_factory=lambda: os.environ.get(
            "QUANTSIM_DB",
            str(Path.home() / ".quantsim" / "quantsim.db")
        )
    )
    journal_mode: str = "WAL"
    cache_size_mb: int = 64


@dataclass
class BacktestConfig:
    initial_capital: float = field(
        default_factory=lambda: float(os.environ.get("QUANTSIM_CAPITAL", "100000"))
    )
    warmup_bars: int = 252
    slippage_model: str = field(
        default_factory=lambda: os.environ.get("QUANTSIM_SLIPPAGE", "volume")
    )   # "none" | "fixed" | "volume"
    commission_model: str = field(
        default_factory=lambda: os.environ.get("QUANTSIM_COMMISSION", "zero")
    )   # "zero" | "per_share" | "per_contract"
    fill_model: str = "next_bar_open"   # "immediate" | "next_bar_open" | "vwap"
    max_drawdown_halt: float = -0.15
    max_drawdown_close: float = -0.20
    max_position_pct: float = 0.10
    slippage_k: float = 0.10            # Almgren-Chriss impact constant
    fixed_spread_bps: float = 5.0


@dataclass
class LiveTradingConfig:
    alpaca_api_key: str = field(
        default_factory=lambda: os.environ.get("ALPACA_API_KEY", "")
    )
    alpaca_secret_key: str = field(
        default_factory=lambda: os.environ.get("ALPACA_SECRET_KEY", "")
    )
    alpaca_paper: bool = field(
        default_factory=lambda: os.environ.get("ALPACA_PAPER", "true").lower() == "true"
    )
    poll_interval_seconds: float = 60.0  # for daily-bar strategies
    fill_poll_seconds: float = 5.0
    market_open_hour_et: int = 9
    market_open_minute_et: int = 30
    market_close_hour_et: int = 16
    max_positions: int = 50


@dataclass
class MLConfig:
    enabled: bool = field(
        default_factory=lambda: os.environ.get("QUANTSIM_ML", "true").lower() == "true"
    )
    train_years: int = 3
    validation_months: int = 6
    retrain_months: int = 1
    min_ic: float = 0.02
    lgbm_n_estimators: int = 200
    lgbm_learning_rate: float = 0.05
    lgbm_max_depth: int = 6


@dataclass
class PortfolioOptConfig:
    default_optimizer: str = "hrp"  # "hrp" | "risk_parity" | "mean_variance" | "black_litterman"
    rebalance_freq: str = "monthly"
    lookback_days: int = 252
    cov_method: str = "ledoit_wolf"
    max_weight: float = 0.30
    min_weight: float = 0.0


@dataclass
class DataConfig:
    fred_api_key: str = field(
        default_factory=lambda: os.environ.get("FRED_API_KEY", "")
    )
    yfinance_sleep_between_requests: float = 0.5
    bootstrap_start: str = "2010-01-01"
    max_universe_size: int = 500
    options_liquid_universe: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "GLD", "TLT",
        "AAPL", "MSFT", "AMZN", "NVDA", "META",
        "GOOGL", "TSLA", "JPM", "GS",
    ])


@dataclass
class DashboardConfig:
    host: str = "localhost"
    port: int = 8501
    refresh_interval_seconds: int = 2
    equity_curve_lookback_days: int = 730
    max_trades_displayed: int = 100


@dataclass
class QuantSimConfig:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    live: LiveTradingConfig = field(default_factory=LiveTradingConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    portfolio_opt: PortfolioOptConfig = field(default_factory=PortfolioOptConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    # Convenience shortcuts
    @property
    def db_path(self) -> str:
        return self.database.path

    @property
    def initial_capital(self) -> float:
        return self.backtest.initial_capital

    @property
    def alpaca_configured(self) -> bool:
        return bool(self.live.alpaca_api_key and self.live.alpaca_secret_key)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Optional[str] = None) -> None:
        """Save config to JSON file."""
        target = Path(path) if path else Path.home() / ".quantsim" / "config.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Config saved to {target}")

    @classmethod
    def load(cls, path: Optional[str] = None) -> "QuantSimConfig":
        """Load config from file, falling back to defaults + env vars."""
        cfg = cls()

        search_paths = [Path(path)] if path else CONFIG_FILE_PATHS
        for config_path in search_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        data = json.load(f)
                    cfg._apply_dict(data)
                    logger.info(f"Config loaded from {config_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load config from {config_path}: {e}")

        return cfg

    def _apply_dict(self, data: Dict) -> None:
        """Apply a nested dict to the config dataclasses."""
        if "database" in data:
            for k, v in data["database"].items():
                if hasattr(self.database, k):
                    setattr(self.database, k, v)
        if "backtest" in data:
            for k, v in data["backtest"].items():
                if hasattr(self.backtest, k):
                    setattr(self.backtest, k, v)
        if "live" in data:
            for k, v in data["live"].items():
                if hasattr(self.live, k):
                    setattr(self.live, k, v)
        if "ml" in data:
            for k, v in data["ml"].items():
                if hasattr(self.ml, k):
                    setattr(self.ml, k, v)
        if "portfolio_opt" in data:
            for k, v in data["portfolio_opt"].items():
                if hasattr(self.portfolio_opt, k):
                    setattr(self.portfolio_opt, k, v)
        if "data" in data:
            for k, v in data["data"].items():
                if hasattr(self.data, k):
                    setattr(self.data, k, v)

    def validate(self) -> List[str]:
        """Return list of validation warnings."""
        warnings = []
        if not self.alpaca_configured:
            warnings.append(
                "ALPACA_API_KEY/ALPACA_SECRET_KEY not set: "
                "paper trading will simulate fills locally"
            )
        if not self.data.fred_api_key:
            warnings.append(
                "FRED_API_KEY not set: macro data download may be rate-limited"
            )
        if self.backtest.initial_capital < 1000:
            warnings.append("initial_capital < $1000: position sizing will be impractical")
        if self.backtest.max_drawdown_halt > 0:
            warnings.append("max_drawdown_halt should be negative (e.g., -0.15)")
        return warnings


# Singleton config instance — import this everywhere
config = QuantSimConfig.load()
