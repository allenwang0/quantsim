"""
Database schema additions for QuantSim v2.

New tables:
  - ml_model_runs: tracks ML model training runs, IC history
  - wfo_results: stores walk-forward optimization results per window
  - garch_state: persists GARCH model parameters for warm restart
  - strategy_performance: per-strategy daily P&L attribution
  - options_greeks_log: time series of portfolio-level Greeks
  - config_snapshots: immutable audit of config at each run start

Migration: safe to run multiple times (IF NOT EXISTS everywhere).
"""

import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

V2_SCHEMA = """
-- ML model training run audit
CREATE TABLE IF NOT EXISTS ml_model_runs (
    run_id          TEXT    PRIMARY KEY,
    strategy_id     TEXT    NOT NULL,
    trained_at      INTEGER NOT NULL,
    train_start     INTEGER NOT NULL,
    train_end       INTEGER NOT NULL,
    n_samples       INTEGER NOT NULL,
    feature_count   INTEGER,
    train_ic        REAL,
    val_ic          REAL,
    model_params    TEXT,           -- JSON
    feature_importances TEXT        -- JSON
);
CREATE INDEX IF NOT EXISTS idx_ml_runs_strategy ON ml_model_runs(strategy_id, trained_at DESC);

-- Rolling IC (Information Coefficient) tracking for ML models
CREATE TABLE IF NOT EXISTS ml_ic_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,
    eval_date       INTEGER NOT NULL,
    horizon_days    INTEGER NOT NULL,
    ic              REAL    NOT NULL,
    rank_ic         REAL,
    t_stat          REAL
);
CREATE INDEX IF NOT EXISTS idx_ic_run ON ml_ic_history(run_id, eval_date DESC);

-- Walk-forward optimization results
CREATE TABLE IF NOT EXISTS wfo_results (
    result_id       TEXT    PRIMARY KEY,
    strategy_id     TEXT    NOT NULL,
    run_timestamp   INTEGER NOT NULL,
    window_id       INTEGER NOT NULL,
    train_start     INTEGER NOT NULL,
    train_end       INTEGER NOT NULL,
    test_start      INTEGER NOT NULL,
    test_end        INTEGER NOT NULL,
    best_params     TEXT    NOT NULL,   -- JSON
    is_sharpe       REAL,
    oos_sharpe      REAL,
    oos_return      REAL,
    oos_max_dd      REAL,
    oos_calmar      REAL
);
CREATE INDEX IF NOT EXISTS idx_wfo_strategy ON wfo_results(strategy_id, run_timestamp DESC);

-- GARCH model state (for warm restarts)
CREATE TABLE IF NOT EXISTS garch_state (
    asset_id        TEXT    PRIMARY KEY,
    updated_at      INTEGER NOT NULL,
    omega           REAL    NOT NULL,
    alpha           REAL    NOT NULL,
    beta            REAL    NOT NULL,
    current_forecast REAL   NOT NULL,
    long_run_vol    REAL    NOT NULL
);

-- Per-strategy daily P&L attribution
CREATE TABLE IF NOT EXISTS strategy_performance (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       INTEGER NOT NULL,
    strategy_id     TEXT    NOT NULL,
    daily_pnl       REAL    NOT NULL,
    realized_pnl    REAL    NOT NULL,
    unrealized_pnl  REAL    NOT NULL,
    n_positions     INTEGER NOT NULL,
    allocated_capital REAL
);
CREATE INDEX IF NOT EXISTS idx_strat_perf ON strategy_performance(strategy_id, timestamp DESC);

-- Portfolio-level options Greeks time series
CREATE TABLE IF NOT EXISTS options_greeks_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       INTEGER NOT NULL,
    portfolio_delta REAL    NOT NULL,
    portfolio_gamma REAL    NOT NULL,
    portfolio_theta REAL    NOT NULL,
    portfolio_vega  REAL    NOT NULL,
    portfolio_rho   REAL,
    n_options_positions INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_greeks_time ON options_greeks_log(timestamp DESC);

-- GARCH volatility forecasts log
CREATE TABLE IF NOT EXISTS garch_forecasts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id        TEXT    NOT NULL,
    timestamp       INTEGER NOT NULL,
    forecast_vol    REAL    NOT NULL,
    realized_vol    REAL,
    leverage_target REAL
);
CREATE INDEX IF NOT EXISTS idx_garch_asset ON garch_forecasts(asset_id, timestamp DESC);

-- Immutable config snapshot at each run start
CREATE TABLE IF NOT EXISTS config_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,
    timestamp       INTEGER NOT NULL,
    run_type        TEXT    NOT NULL,   -- 'backtest' | 'paper_trading' | 'vectorized'
    config_json     TEXT    NOT NULL,
    strategy_ids    TEXT    NOT NULL,   -- JSON array
    universe        TEXT    NOT NULL    -- JSON array
);

-- Parameter sweep results (from vectorized backtester)
CREATE TABLE IF NOT EXISTS param_sweep_results (
    sweep_id        TEXT    NOT NULL,
    run_timestamp   INTEGER NOT NULL,
    signal_fn       TEXT    NOT NULL,
    params          TEXT    NOT NULL,   -- JSON
    sharpe_ratio    REAL,
    cagr            REAL,
    max_drawdown    REAL,
    calmar_ratio    REAL,
    deflated_sharpe REAL,
    trade_count     INTEGER,
    PRIMARY KEY (sweep_id, params)
);
CREATE INDEX IF NOT EXISTS idx_sweep ON param_sweep_results(sweep_id, deflated_sharpe DESC);
"""


def migrate_v2(db_path: str) -> None:
    """
    Apply v2 schema additions to an existing database.
    Safe to run on a fresh database or an existing one.
    """
    from core.database import get_connection
    conn = get_connection(db_path)
    try:
        conn.executescript(V2_SCHEMA)
        conn.commit()
        logger.info(f"[DB] v2 migration complete: {db_path}")
    except Exception as e:
        logger.error(f"[DB] v2 migration failed: {e}")
        raise
    finally:
        conn.close()


def init_full_db(db_path: str) -> None:
    """Initialize both v1 and v2 schemas."""
    from core.database import init_db
    init_db(db_path)
    migrate_v2(db_path)


# ── Database Helpers for New Tables ───────────────────────────────────────────

def log_ml_run(
    db_path: str,
    run_id: str,
    strategy_id: str,
    train_start: int,
    train_end: int,
    n_samples: int,
    train_ic: float,
    val_ic: float,
    model_params: dict,
    feature_importances: dict = None,
) -> None:
    import json, time
    from core.database import db_conn
    try:
        with db_conn(db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO ml_model_runs
                   (run_id, strategy_id, trained_at, train_start, train_end,
                    n_samples, train_ic, val_ic, model_params, feature_importances)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, strategy_id, int(time.time()), train_start, train_end,
                 n_samples, train_ic, val_ic,
                 json.dumps(model_params),
                 json.dumps(feature_importances or {})),
            )
    except Exception as e:
        logger.error(f"log_ml_run failed: {e}")


def log_garch_state(
    db_path: str,
    asset_id: str,
    omega: float,
    alpha: float,
    beta: float,
    current_forecast: float,
) -> None:
    import time
    from core.database import db_conn
    long_run_vol = (omega / (1 - alpha - beta)) ** 0.5 * (252 ** 0.5) if alpha + beta < 1 else 0.15
    try:
        with db_conn(db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO garch_state
                   (asset_id, updated_at, omega, alpha, beta, current_forecast, long_run_vol)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (asset_id, int(time.time()), omega, alpha, beta, current_forecast, long_run_vol),
            )
    except Exception as e:
        logger.error(f"log_garch_state failed: {e}")


def log_options_greeks(
    db_path: str,
    timestamp: int,
    portfolio,
) -> None:
    from core.database import db_conn
    try:
        with db_conn(db_path) as conn:
            conn.execute(
                """INSERT INTO options_greeks_log
                   (timestamp, portfolio_delta, portfolio_gamma,
                    portfolio_theta, portfolio_vega, n_options_positions)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (timestamp,
                 portfolio.portfolio_delta,
                 portfolio.portfolio_gamma,
                 portfolio.portfolio_theta,
                 portfolio.portfolio_vega,
                 sum(1 for p in portfolio.positions.values()
                     if p.asset_type.value == "OPTION")),
            )
    except Exception as e:
        logger.error(f"log_options_greeks failed: {e}")


def log_wfo_result(
    db_path: str,
    result_id: str,
    strategy_id: str,
    window: dict,
    best_params: dict,
    metrics: dict,
) -> None:
    import json, time
    from core.database import db_conn
    try:
        with db_conn(db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO wfo_results
                   (result_id, strategy_id, run_timestamp, window_id,
                    train_start, train_end, test_start, test_end,
                    best_params, is_sharpe, oos_sharpe, oos_return,
                    oos_max_dd, oos_calmar)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (result_id, strategy_id, int(time.time()),
                 window.get("window_id", 0),
                 int(window.get("train_start", pd.Timestamp("2010-01-01")).timestamp()),
                 int(window.get("train_end", pd.Timestamp("2013-01-01")).timestamp()),
                 int(window.get("test_start", pd.Timestamp("2013-01-01")).timestamp()),
                 int(window.get("test_end", pd.Timestamp("2014-01-01")).timestamp()),
                 json.dumps(best_params),
                 metrics.get("train_best_sharpe", 0),
                 metrics.get("oos_sharpe", 0),
                 metrics.get("oos_return", 0),
                 metrics.get("oos_max_dd", 0),
                 metrics.get("oos_calmar", 0)),
            )
    except Exception as e:
        logger.error(f"log_wfo_result failed: {e}")


import pandas as pd
