"""
Performance analytics: every metric documented in the spec.
Deflated Sharpe Ratio, trade statistics, benchmark comparison, rolling metrics.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
EULER_MASCHERONI = 0.5772156649015329


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    n_trials: int = 1,
    trial_sharpe_variance: Optional[float] = None,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Deflated Sharpe ratio (Bailey and Lopez de Prado, 2014), a probability.

    sharpe: per-period (not annualized) Sharpe ratio of the selected strategy.
    n_obs: number of return observations behind it.
    n_trials: number of configurations tried before selecting it.
    trial_sharpe_variance: variance of the per-period Sharpe ratios across
        those trials. When omitted, the sampling variance of a Sharpe
        estimate under the null, 1 / (n_obs - 1), is used.
    kurtosis: raw kurtosis (3 for normal returns), not excess kurtosis.

    Returns the probability that the true Sharpe ratio exceeds the expected
    maximum Sharpe of n_trials unskilled strategies. Above 0.95 is the
    usual significance cut-off.
    """
    if n_obs < 2:
        return 0.0
    if trial_sharpe_variance is None:
        trial_sharpe_variance = 1.0 / (n_obs - 1)

    if n_trials > 1:
        z_first = stats.norm.ppf(1 - 1.0 / n_trials)
        z_second = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        expected_max = np.sqrt(trial_sharpe_variance) * (
            (1 - EULER_MASCHERONI) * z_first + EULER_MASCHERONI * z_second
        )
    else:
        expected_max = 0.0

    sharpe_variance_factor = 1 - skewness * sharpe + (kurtosis - 1) / 4 * sharpe**2
    if sharpe_variance_factor <= 0:
        return 0.0
    z = (sharpe - expected_max) * np.sqrt(n_obs - 1) / np.sqrt(sharpe_variance_factor)
    return float(stats.norm.cdf(z))


class PerformanceAnalytics:
    """
    Computes the full performance analytics suite from a portfolio equity curve
    and trade log.

    All metrics documented:
    - Return: total return, CAGR, monthly distribution
    - Risk: Sharpe, Sortino, Calmar, max drawdown, drawdown duration
    - Trade stats: win rate, profit factor, expectancy, holding period
    - Advanced: deflated Sharpe ratio, information ratio, alpha/beta
    - Rolling: 12m rolling Sharpe, 12m rolling Calmar
    """

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate

    def compute_all(
        self,
        equity_curve: pd.Series,  # index=datetime, values=portfolio equity
        trades: Optional[pd.DataFrame] = None,
        benchmark: Optional[pd.Series] = None,
        n_strategies_tested: int = 1,  # for deflated Sharpe calculation
        trial_sharpe_variance: Optional[float] = None,  # per-period, across trials
    ) -> Dict:
        """
        Master analytics computation. Returns full dict of all metrics.
        equity_curve: DatetimeIndex, float values (dollar equity)
        trades: DataFrame with columns [strategy_id, direction, entry_price,
                exit_price, quantity, realized_pnl, holding_bars]
        benchmark: DatetimeIndex, float values (benchmark equity)
        """
        if equity_curve.empty:
            return {}

        results = {}

        # Daily returns
        daily_returns = equity_curve.pct_change().dropna()
        if daily_returns.empty:
            return {}

        # ── Return Metrics ─────────────────────────────────────────────────────
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0
        n_years = len(daily_returns) / 252.0
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1.0 / max(n_years, 0.01)) - 1.0

        results["total_return"] = total_return
        results["cagr"] = cagr
        results["n_years"] = n_years
        results["final_equity"] = float(equity_curve.iloc[-1])
        results["initial_equity"] = float(equity_curve.iloc[0])

        # Monthly returns
        monthly_equity = equity_curve.resample("ME").last()
        monthly_returns = monthly_equity.pct_change(fill_method=None).dropna()
        results["monthly_returns"] = monthly_returns.to_dict()
        results["best_month"] = float(monthly_returns.max()) if not monthly_returns.empty else 0
        results["worst_month"] = float(monthly_returns.min()) if not monthly_returns.empty else 0

        # ── Risk Metrics ────────────────────────────────────────────────────────
        annual_vol = float(daily_returns.std() * np.sqrt(252))
        results["annual_volatility"] = annual_vol

        # Sharpe ratio (annualized properly, not sqrt(12) * monthly)
        daily_rf = self.risk_free_rate / 252
        excess_returns = daily_returns - daily_rf
        sharpe = (
            float(excess_returns.mean() / excess_returns.std() * np.sqrt(252))
            if excess_returns.std() > 0
            else 0
        )
        results["sharpe_ratio"] = sharpe

        # Sortino: downside deviation below the risk-free target, over all periods
        shortfall = np.minimum(excess_returns.to_numpy(), 0.0)
        downside_dev = float(np.sqrt(np.mean(shortfall**2)))
        sortino = (
            float(excess_returns.mean() / downside_dev * np.sqrt(TRADING_DAYS_PER_YEAR))
            if downside_dev > 0
            else 0.0
        )
        results["sortino_ratio"] = sortino

        # Drawdown metrics
        dd_series, max_dd, avg_dd, max_dd_duration = self._compute_drawdowns(equity_curve)
        results["max_drawdown"] = max_dd
        results["avg_drawdown"] = avg_dd
        results["max_drawdown_duration_days"] = max_dd_duration
        results["drawdown_series"] = dd_series.to_dict()

        # Calmar
        calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-8 else 0
        results["calmar_ratio"] = calmar

        # ── Advanced Analytics ─────────────────────────────────────────────────
        # Deflated Sharpe Ratio (Bailey and Lopez de Prado 2014), on per-period Sharpe
        skewness = float(daily_returns.skew()) if len(daily_returns) > 2 else 0.0
        excess_kurtosis = float(daily_returns.kurt()) if len(daily_returns) > 3 else 0.0
        dsr = deflated_sharpe_ratio(
            sharpe=sharpe / np.sqrt(TRADING_DAYS_PER_YEAR),
            n_obs=len(daily_returns),
            n_trials=n_strategies_tested,
            trial_sharpe_variance=trial_sharpe_variance,
            skewness=0.0 if np.isnan(skewness) else skewness,
            kurtosis=3.0 + (0.0 if np.isnan(excess_kurtosis) else excess_kurtosis),
        )
        results["deflated_sharpe_ratio"] = dsr
        results["deflated_sharpe_significant"] = dsr > 0.95

        # Rolling metrics
        rolling_sharpe = self._rolling_sharpe(daily_returns, window=252)
        results["rolling_sharpe_12m"] = rolling_sharpe.to_dict()

        rolling_calmar = self._rolling_calmar(equity_curve, window=252)
        results["rolling_calmar_12m"] = rolling_calmar.to_dict()

        # Benchmark comparison
        if benchmark is not None and not benchmark.empty:
            bench_stats = self._benchmark_comparison(daily_returns, benchmark)
            results.update(bench_stats)

        # ── Trade Statistics ───────────────────────────────────────────────────
        if trades is not None and not trades.empty:
            trade_stats = self._compute_trade_stats(trades)
            results.update(trade_stats)

        # ── Annualized Turnover ────────────────────────────────────────────────
        # Computed from trades if available
        if trades is not None and not trades.empty and "quantity" in trades.columns:
            avg_equity = float(equity_curve.mean())
            if avg_equity > 0:
                total_traded = float(
                    (trades.get("quantity", pd.Series()) * trades.get("entry_price", pd.Series()))
                    .abs()
                    .sum()
                )
                turnover = total_traded / avg_equity / max(n_years, 1)
                results["annual_turnover"] = turnover

        return results

    def _compute_drawdowns(self, equity: pd.Series) -> Tuple[pd.Series, float, float, int]:
        """
        Returns (drawdown_series, max_drawdown, avg_drawdown, max_duration_days).
        Drawdown series: negative fractions from rolling peak.
        """
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max

        max_dd = float(drawdown.min())

        # Compute individual drawdown periods for avg
        drawdown_magnitudes = []
        max_duration = 0
        current_start = None

        for i, (ts, val) in enumerate(drawdown.items()):
            if val < -1e-6 and current_start is None:
                current_start = ts
            elif val >= -1e-6 and current_start is not None:
                if hasattr(ts, "to_pydatetime") and hasattr(current_start, "to_pydatetime"):
                    duration = (ts - current_start).days
                else:
                    duration = i
                max_duration = max(max_duration, duration)
                drawdown_magnitudes.append(float(drawdown[current_start:ts].min()))
                current_start = None

        avg_dd = float(np.mean(drawdown_magnitudes)) if drawdown_magnitudes else 0.0
        return drawdown, max_dd, avg_dd, max_duration

    def _rolling_sharpe(self, daily_returns: pd.Series, window: int = 252) -> pd.Series:
        """Rolling annualized Sharpe ratio."""
        daily_rf = self.risk_free_rate / 252
        excess = daily_returns - daily_rf
        rolling_mean = excess.rolling(window).mean()
        rolling_std = excess.rolling(window).std()
        return (rolling_mean / rolling_std * np.sqrt(252)).fillna(0)

    def _rolling_calmar(self, equity: pd.Series, window: int = 252) -> pd.Series:
        """Rolling Calmar ratio."""
        rolling_returns = equity.pct_change(window)

        def rolling_max_dd(sub):
            if len(sub) < 2:
                return 0.0
            roll_max = sub.expanding().max()
            dd = (sub - roll_max) / roll_max
            return float(dd.min())

        rolling_cagr = (1 + rolling_returns) ** (252 / window) - 1
        rolling_dd = equity.rolling(window).apply(rolling_max_dd, raw=False)
        calmar = rolling_cagr / rolling_dd.abs().replace(0, np.nan)
        return calmar.fillna(0)

    def _benchmark_comparison(self, strategy_returns: pd.Series, benchmark: pd.Series) -> Dict:
        """Alpha, beta, information ratio vs benchmark."""
        bench_returns = benchmark.pct_change().dropna()

        # Align on common dates
        aligned = pd.concat([strategy_returns, bench_returns], axis=1, join="inner")
        aligned.columns = ["strategy", "benchmark"]
        aligned = aligned.dropna()

        if len(aligned) < 30:
            return {}

        # OLS: strategy_return = alpha + beta * benchmark_return
        beta, alpha_daily, r_val, p_val, se = stats.linregress(
            aligned["benchmark"], aligned["strategy"]
        )
        alpha_annual = alpha_daily * 252

        # Information ratio
        active_returns = aligned["strategy"] - aligned["benchmark"]
        ir = (
            float(active_returns.mean() / active_returns.std() * np.sqrt(252))
            if active_returns.std() > 0
            else 0
        )

        bench_total = (1 + aligned["benchmark"]).prod() - 1
        strat_total = (1 + aligned["strategy"]).prod() - 1

        return {
            "alpha_annual": float(alpha_annual),
            "beta": float(beta),
            "r_squared": float(r_val**2),
            "information_ratio": ir,
            "benchmark_total_return": float(bench_total),
            "strategy_vs_benchmark": float(strat_total - bench_total),
        }

    def _compute_trade_stats(self, trades: pd.DataFrame) -> Dict:
        """Full trade statistics from the trade log."""
        closed = trades[trades["realized_pnl"].notna()].copy()
        if closed.empty:
            return {}

        pnls = closed["realized_pnl"]
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0
        avg_win = float(wins.mean()) if len(wins) > 0 else 0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0
        profit_factor = (
            float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else float("inf")
        )
        expectancy = win_rate * avg_win - (1 - win_rate) * abs(avg_loss)

        avg_holding = (
            float(closed["holding_bars"].mean()) if "holding_bars" in closed.columns else 0
        )

        # Minimum sample size warning
        n_trades = len(closed)
        insufficient_sample = n_trades < 30

        return {
            "n_trades": n_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_win_loss_ratio": abs(avg_win / avg_loss) if avg_loss != 0 else float("inf"),
            "profit_factor": profit_factor,
            "expectancy": float(expectancy),
            "avg_holding_bars": avg_holding,
            "gross_profit": float(wins.sum()),
            "gross_loss": float(losses.sum()),
            "total_pnl": float(pnls.sum()),
            "insufficient_sample_warning": insufficient_sample,
            "sample_warning_message": (
                f"Only {n_trades} trades: Sharpe unreliable below 30 independent trades"
                if insufficient_sample
                else ""
            ),
        }

    def print_summary(self, results: Dict) -> None:
        """Print a formatted performance summary to stdout."""
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"  Total Return:          {results.get('total_return', 0):.2%}")
        print(f"  CAGR:                  {results.get('cagr', 0):.2%}")
        print(f"  Annual Volatility:     {results.get('annual_volatility', 0):.2%}")
        print(f"  Sharpe Ratio:          {results.get('sharpe_ratio', 0):.3f}")
        print(f"  Sortino Ratio:         {results.get('sortino_ratio', 0):.3f}")
        print(f"  Calmar Ratio:          {results.get('calmar_ratio', 0):.3f}")
        print(f"  Max Drawdown:          {results.get('max_drawdown', 0):.2%}")
        print(f"  Avg Drawdown:          {results.get('avg_drawdown', 0):.2%}")
        print(f"  Max DD Duration:       {results.get('max_drawdown_duration_days', 0)} days")
        print(f"  Deflated SR:           {results.get('deflated_sharpe_ratio', 0):.3f}")
        print(f"  SR Significant:        {results.get('deflated_sharpe_significant', False)}")
        if "alpha_annual" in results:
            print(f"  Alpha (annual):        {results.get('alpha_annual', 0):.2%}")
            print(f"  Beta:                  {results.get('beta', 0):.3f}")
            print(f"  Information Ratio:     {results.get('information_ratio', 0):.3f}")
        if "n_trades" in results:
            print(f"  Trade Count:           {results.get('n_trades', 0)}")
            print(f"  Win Rate:              {results.get('win_rate', 0):.2%}")
            print(f"  Profit Factor:         {results.get('profit_factor', 0):.2f}")
            print(f"  Expectancy:            ${results.get('expectancy', 0):.2f}")
            print(f"  Avg Holding:           {results.get('avg_holding_bars', 0):.1f} bars")
            if results.get("insufficient_sample_warning"):
                print(f"  ⚠ WARNING: {results.get('sample_warning_message', '')}")
        print("=" * 60 + "\n")


def load_equity_curve_from_db(db_path: str) -> pd.Series:
    """Load portfolio equity curve from database snapshots."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT timestamp, total_equity FROM portfolio_snapshots ORDER BY timestamp ASC"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.Series(dtype=float)

    timestamps = [datetime.utcfromtimestamp(r[0]) for r in rows]
    equities = [r[1] for r in rows]
    return pd.Series(equities, index=pd.DatetimeIndex(timestamps))


def load_trades_from_db(db_path: str) -> pd.DataFrame:
    """Load trade log from database."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """SELECT trade_id, strategy_id, asset_id, direction,
                      entry_timestamp, exit_timestamp, entry_price, exit_price,
                      quantity, realized_pnl, commission, holding_bars
               FROM trades ORDER BY entry_timestamp ASC"""
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        rows,
        columns=[
            "trade_id",
            "strategy_id",
            "asset_id",
            "direction",
            "entry_timestamp",
            "exit_timestamp",
            "entry_price",
            "exit_price",
            "quantity",
            "realized_pnl",
            "commission",
            "holding_bars",
        ],
    )
