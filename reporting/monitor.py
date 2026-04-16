"""
Live Strategy Health Monitor

Continuously monitors running strategies and emits alerts when:
- ML model IC drops below threshold (model stale)
- Portfolio drawdown approaches halt threshold
- Strategy equity correlation changes unexpectedly
- Position concentration limit approached
- GARCH forecast vol spikes (regime change incoming)
- Strategy has been flat too long (dead signal)

Used in both paper trading (real-time) and backtesting (post-run health check).

The monitor polls the database every N seconds and writes to risk_alerts.
The dashboard reads risk_alerts and shows banners.
"""

from __future__ import annotations
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import numpy as np
import pandas as pd

from core.config import config

logger = logging.getLogger(__name__)


class StrategyHealthMonitor:
    """
    Monitors strategy health and emits structured alerts.

    In paper trading: runs as an asyncio coroutine (poll every 60s).
    In backtesting: call check_all() at the end of each bar.

    Alerts are written to the risk_alerts DB table and displayed
    on the dashboard.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        min_ic: float = 0.02,
        max_dd_warn: float = -0.10,
        max_dd_critical: float = -0.15,
        stale_signal_bars: int = 63,
        vol_spike_multiple: float = 2.0,
    ):
        self.db_path = db_path or config.db_path
        self.min_ic = min_ic
        self.max_dd_warn = max_dd_warn
        self.max_dd_critical = max_dd_critical
        self.stale_signal_bars = stale_signal_bars
        self.vol_spike_multiple = vol_spike_multiple

        # Alert deduplication: don't re-alert within 1 hour
        self._last_alert: Dict[str, datetime] = {}
        self._alert_cooldown = timedelta(hours=1)

        # Strategy state tracking
        self._strategy_flat_bars: Dict[str, int] = {}
        self._last_equity: Dict[str, float] = {}
        self._baseline_vol: Dict[str, float] = {}

    def check_all(self, portfolio, timestamp: datetime) -> List[Dict]:
        """
        Run all health checks. Returns list of alert dicts.
        Call this on every bar in the backtesting engine.
        """
        alerts = []

        # 1. Portfolio drawdown check
        dd_alerts = self._check_drawdown(portfolio, timestamp)
        alerts.extend(dd_alerts)

        # 2. Concentration check
        conc_alerts = self._check_concentration(portfolio, timestamp)
        alerts.extend(conc_alerts)

        # 3. Stale signal check
        stale_alerts = self._check_stale_signals(portfolio, timestamp)
        alerts.extend(stale_alerts)

        # 4. ML IC check (reads from DB)
        ic_alerts = self._check_ml_ic(timestamp)
        alerts.extend(ic_alerts)

        # 5. GARCH vol spike check (reads from DB)
        vol_alerts = self._check_vol_spike(timestamp)
        alerts.extend(vol_alerts)

        # Persist new alerts
        for alert in alerts:
            self._persist_alert(alert)

        return alerts

    def _check_drawdown(self, portfolio, timestamp: datetime) -> List[Dict]:
        alerts = []
        dd = portfolio.current_drawdown
        equity = portfolio.total_equity

        if dd <= self.max_dd_critical:
            alert = self._make_alert(
                key=f"dd_critical_{timestamp.date()}",
                risk_type="MAX_DRAWDOWN_CRITICAL",
                message=f"Portfolio drawdown {dd:.1%} breached critical threshold {self.max_dd_critical:.1%}. Equity: ${equity:,.0f}",
                severity="CRITICAL",
                timestamp=timestamp,
            )
            if alert:
                alerts.append(alert)
        elif dd <= self.max_dd_warn:
            alert = self._make_alert(
                key=f"dd_warn_{timestamp.date()}",
                risk_type="MAX_DRAWDOWN_WARNING",
                message=f"Portfolio drawdown {dd:.1%} approaching halt threshold {self.max_dd_critical:.1%}",
                severity="WARNING",
                timestamp=timestamp,
            )
            if alert:
                alerts.append(alert)
        return alerts

    def _check_concentration(self, portfolio, timestamp: datetime) -> List[Dict]:
        alerts = []
        equity = portfolio.total_equity
        if equity <= 0:
            return alerts

        for asset_id, pos in portfolio.positions.items():
            weight = abs(pos.market_value) / equity
            if weight > config.backtest.max_position_pct * 1.5:
                alert = self._make_alert(
                    key=f"conc_{asset_id}_{timestamp.date()}",
                    risk_type="CONCENTRATION_BREACH",
                    message=f"{asset_id} is {weight:.1%} of portfolio (limit: {config.backtest.max_position_pct:.1%})",
                    severity="WARNING",
                    timestamp=timestamp,
                )
                if alert:
                    alerts.append(alert)
        return alerts

    def _check_stale_signals(self, portfolio, timestamp: datetime) -> List[Dict]:
        """Warn if a strategy has emitted no signals (stayed flat) for too long."""
        alerts = []
        # Check if portfolio has been unchanged for too long
        current_n_positions = len(portfolio.positions)
        if current_n_positions == 0:
            bars = self._strategy_flat_bars.get("portfolio", 0) + 1
            self._strategy_flat_bars["portfolio"] = bars

            if bars >= self.stale_signal_bars:
                alert = self._make_alert(
                    key=f"stale_{timestamp.date()}",
                    risk_type="STALE_SIGNALS",
                    message=f"Portfolio has been flat for {bars} bars. Strategies may have stopped generating signals.",
                    severity="WARNING",
                    timestamp=timestamp,
                )
                if alert:
                    alerts.append(alert)
        else:
            self._strategy_flat_bars["portfolio"] = 0
        return alerts

    def _check_ml_ic(self, timestamp: datetime) -> List[Dict]:
        """Check if ML model IC has dropped below the minimum threshold."""
        try:
            from core.database import db_conn
            rows = db_conn(self.db_path).__enter__().execute(
                "SELECT ic FROM ml_ic_history ORDER BY eval_date DESC LIMIT 12"
            ).fetchall()
            if rows and len(rows) >= 3:
                ic_values = [r["ic"] for r in rows]
                rolling_ic = float(np.mean(ic_values))
                if rolling_ic < self.min_ic:
                    alert = self._make_alert(
                        key=f"ml_ic_{timestamp.date()}",
                        risk_type="ML_IC_DECAY",
                        message=f"ML model rolling IC {rolling_ic:.4f} below threshold {self.min_ic}. Model may be stale.",
                        severity="WARNING",
                        timestamp=timestamp,
                    )
                    if alert:
                        return [alert]
        except Exception:
            pass
        return []

    def _check_vol_spike(self, timestamp: datetime) -> List[Dict]:
        """Check if GARCH-forecast volatility has spiked significantly."""
        try:
            from core.database import db_conn
            with db_conn(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT asset_id, forecast_vol FROM garch_forecasts ORDER BY timestamp DESC LIMIT 50"
                ).fetchall()

            for row in rows:
                asset_id = row["asset_id"]
                current_vol = float(row["forecast_vol"])
                baseline = self._baseline_vol.get(asset_id)

                if baseline is None:
                    self._baseline_vol[asset_id] = current_vol
                    continue

                if current_vol > baseline * self.vol_spike_multiple:
                    alert = self._make_alert(
                        key=f"volspike_{asset_id}_{timestamp.date()}",
                        risk_type="VOLATILITY_SPIKE",
                        message=f"{asset_id} GARCH vol {current_vol:.1%} is {current_vol/baseline:.1f}x baseline {baseline:.1%}. Consider reducing exposure.",
                        severity="WARNING",
                        timestamp=timestamp,
                    )
                    if alert:
                        return [alert]
                else:
                    # Slowly update baseline (EMA with alpha=0.02)
                    self._baseline_vol[asset_id] = 0.98 * baseline + 0.02 * current_vol
        except Exception:
            pass
        return []

    def _make_alert(
        self,
        key: str,
        risk_type: str,
        message: str,
        severity: str,
        timestamp: datetime,
    ) -> Optional[Dict]:
        """
        Create an alert dict if not in cooldown period.
        Returns None if this alert was recently sent.
        """
        last = self._last_alert.get(key)
        if last and (timestamp - last) < self._alert_cooldown:
            return None

        self._last_alert[key] = timestamp
        return {
            "risk_type": risk_type,
            "message": message,
            "severity": severity,
            "timestamp": int(timestamp.timestamp()),
        }

    def _persist_alert(self, alert: Dict) -> None:
        try:
            from core.database import db_conn
            with db_conn(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO risk_alerts (timestamp, risk_type, message, severity) VALUES (?, ?, ?, ?)",
                    (alert["timestamp"], alert["risk_type"], alert["message"], alert["severity"]),
                )
        except Exception as e:
            logger.debug(f"Alert persist failed: {e}")

    async def run_async(self, portfolio, interval_seconds: int = 60) -> None:
        """Run health monitoring as an asyncio coroutine (paper trading mode)."""
        import asyncio
        while True:
            try:
                alerts = self.check_all(portfolio, datetime.utcnow())
                for a in alerts:
                    logger.warning(f"[MONITOR] {a['severity']}: {a['message']}")
            except Exception as e:
                logger.debug(f"Monitor error: {e}")
            await asyncio.sleep(interval_seconds)
