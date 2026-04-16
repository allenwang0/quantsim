"""
Advanced Reporting and Analytics

What institutional quant research adds beyond basic Sharpe/drawdown:
- Information Coefficient (IC) time series for ML models
- Factor decay curves (how quickly does an alpha decay?)
- Turnover analysis and transaction cost impact
- Capacity analysis (would this work at larger capital?)
- Regime breakdown (bull/bear, high/low vol)
- CAGR vs. max drawdown scatter across parameter space
- Monte Carlo simulation for confidence intervals on strategy statistics
"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class AdvancedAnalytics:
    """
    Institutional-grade analytics beyond basic performance metrics.
    """

    @staticmethod
    def monte_carlo_sharpe(
        daily_returns: pd.Series,
        n_simulations: int = 10_000,
        confidence_level: float = 0.95,
    ) -> Dict:
        """
        Bootstrap confidence interval for Sharpe ratio.
        
        The standard error of Sharpe is large for short samples.
        A strategy with Sharpe 0.8 and 252 bars has a 95% CI of roughly (0.3, 1.3).
        This is why minimum sample size matters so much.
        
        Uses stationary bootstrap (Politis & Romano 1994) which
        preserves autocorrelation structure of returns.
        """
        n = len(daily_returns)
        if n < 30:
            return {"error": "Insufficient sample size (<30 obs)"}

        returns_arr = daily_returns.values
        rf_daily = 0.05 / 252

        sharpe_samples = []
        block_size = max(1, int(np.sqrt(n)))  # typical block size for bootstrap

        for _ in range(n_simulations):
            # Stationary bootstrap: random block lengths
            bootstrap_sample = []
            while len(bootstrap_sample) < n:
                start = np.random.randint(0, n)
                length = np.random.geometric(1.0 / block_size)
                block = returns_arr[start:min(start + length, n)]
                bootstrap_sample.extend(block)

            sample = np.array(bootstrap_sample[:n])
            excess = sample - rf_daily
            sharpe = float(excess.mean() / (excess.std() + 1e-10) * np.sqrt(252))
            sharpe_samples.append(sharpe)

        sharpe_samples = np.array(sharpe_samples)
        alpha = 1 - confidence_level
        ci_low = np.percentile(sharpe_samples, alpha/2 * 100)
        ci_high = np.percentile(sharpe_samples, (1 - alpha/2) * 100)

        actual_sharpe = float(
            (daily_returns.mean() - rf_daily) / (daily_returns.std() + 1e-10) * np.sqrt(252)
        )

        return {
            "sharpe_point_estimate": actual_sharpe,
            "sharpe_ci_low": ci_low,
            "sharpe_ci_high": ci_high,
            "confidence_level": confidence_level,
            "n_simulations": n_simulations,
            "p_positive_sharpe": float((sharpe_samples > 0).mean()),
            "warning": (
                "Wide confidence interval: need more data"
                if (ci_high - ci_low) > 1.0
                else ""
            ),
        }

    @staticmethod
    def factor_decay_analysis(
        signal: pd.DataFrame,
        forward_returns: pd.DataFrame,
        horizons: List[int] = [1, 5, 10, 21, 42, 63],
    ) -> pd.DataFrame:
        """
        Compute IC (Information Coefficient) at each forward horizon.
        Shows how quickly the signal's predictive power decays.
        
        A momentum signal might have IC=0.03 at 1 day, 0.05 at 21 days,
        decaying to 0 at 252 days. The peak IC horizon is when to hold.
        
        Returns DataFrame: horizons x columns=[IC, rank_IC, std_IC, t_stat]
        """
        results = []

        for h in horizons:
            fwd_ret = forward_returns.pct_change(h).shift(-h)

            ics = []
            for date in signal.index:
                if date not in fwd_ret.index:
                    continue
                sig_row = signal.loc[date].dropna()
                ret_row = fwd_ret.loc[date].dropna()
                common = sig_row.index.intersection(ret_row.index)

                if len(common) < 5:
                    continue

                ic = float(sig_row[common].corr(ret_row[common], method="spearman"))
                ics.append(ic)

            if ics:
                ics_arr = np.array(ics)
                t_stat = float(
                    np.mean(ics_arr) / (np.std(ics_arr) / np.sqrt(len(ics_arr)) + 1e-10)
                )
                results.append({
                    "horizon": h,
                    "mean_ic": float(np.mean(ics_arr)),
                    "std_ic": float(np.std(ics_arr)),
                    "ir": float(np.mean(ics_arr) / (np.std(ics_arr) + 1e-10)),  # information ratio
                    "t_stat": t_stat,
                    "pct_positive": float((ics_arr > 0).mean()),
                    "n_obs": len(ics_arr),
                })

        return pd.DataFrame(results).set_index("horizon")

    @staticmethod
    def turnover_analysis(
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        commission_rate: float = 0.001,
    ) -> Dict:
        """
        Analyze portfolio turnover and its cost impact.
        
        High-turnover strategies that look profitable on paper often fail
        once realistic transaction costs are applied. This is the most
        common reason retail backtests overstate real performance.
        
        Returns turnover rate, estimated annual transaction cost drag,
        and break-even commission rate.
        """
        # Compute daily position changes
        position_changes = signals.diff().abs().fillna(0)
        daily_turnover = position_changes.sum(axis=1) / max(len(signals.columns), 1)
        annual_turnover = float(daily_turnover.mean() * 252)

        # Transaction cost drag
        avg_price = prices.mean().mean()
        daily_cost = daily_turnover * avg_price * commission_rate
        annual_cost_pct = float(daily_cost.mean() * 252)

        # Gross return estimate (before costs)
        raw_returns = prices.pct_change()
        if not signals.empty and not raw_returns.empty:
            weighted_returns = (signals.shift(1) * raw_returns).sum(axis=1)
            gross_annual_return = float((1 + weighted_returns).prod() ** (252 / len(weighted_returns)) - 1)
        else:
            gross_annual_return = 0.0

        net_annual_return = gross_annual_return - annual_cost_pct

        return {
            "annual_turnover_rate": annual_turnover,
            "daily_avg_turnover": float(daily_turnover.mean()),
            "estimated_annual_cost_pct": annual_cost_pct,
            "gross_annual_return": gross_annual_return,
            "net_annual_return": net_annual_return,
            "cost_as_pct_of_gross": float(annual_cost_pct / abs(gross_annual_return + 1e-10)),
            "break_even_commission_bps": float(
                gross_annual_return / (annual_turnover + 1e-10) * 10_000
            ),
        }

    @staticmethod
    def capacity_analysis(
        trades_df: pd.DataFrame,
        prices: pd.DataFrame,
        current_capital: float,
        target_capital: float,
        adv_fraction: float = 0.01,
    ) -> Dict:
        """
        Estimate how a strategy scales with capital.
        
        Market impact grows as sqrt(order_size / ADV).
        A strategy that works at $100K may not work at $10M
        if it needs to trade illiquid names.
        
        Returns the estimated performance degradation at target capital.
        """
        if trades_df.empty:
            return {"error": "No trade data"}

        avg_trade_size_current = float(trades_df.get("quantity", pd.Series()).mean())
        scale_factor = target_capital / current_capital

        # Estimate ADV (use prices as proxy)
        avg_daily_volume_usd = float(prices.mean().mean() * 1_000_000)  # assumed 1M shares

        # Current market impact cost (Almgren-Chriss)
        k = 0.1  # impact constant
        current_participation = avg_trade_size_current / (avg_daily_volume_usd / prices.mean().mean() + 1)
        current_slippage = float(k * np.sqrt(current_participation))

        # Slippage at target capital
        scaled_participation = current_participation * scale_factor
        target_slippage = float(k * np.sqrt(scaled_participation))

        slippage_increase = target_slippage - current_slippage

        return {
            "current_capital": current_capital,
            "target_capital": target_capital,
            "scale_factor": scale_factor,
            "current_avg_slippage_pct": current_slippage * 100,
            "target_avg_slippage_pct": target_slippage * 100,
            "slippage_increase_pct": slippage_increase * 100,
            "annual_slippage_drag_bps": slippage_increase * 10_000 * 2,  # round trip
            "max_viable_capital_estimate": float(
                current_capital * (adv_fraction ** 2) / (current_participation + 1e-10)
            ),
            "recommendation": (
                "Strategy is capacity-constrained at target capital"
                if slippage_increase > 0.005
                else "Strategy should scale reasonably to target capital"
            ),
        }

    @staticmethod
    def return_attribution(
        equity_curve: pd.Series,
        benchmark: pd.Series,
        factor_returns: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Brinson-Hood-Beebower style return attribution.
        Decomposes strategy return into:
        - Market beta contribution
        - Factor exposures (if provided)
        - Idiosyncratic (strategy-specific) alpha
        """
        # Align series
        returns = equity_curve.pct_change().dropna()
        bench_returns = benchmark.pct_change().dropna()
        aligned = pd.concat([returns, bench_returns], axis=1, join="inner").dropna()

        if len(aligned) < 30:
            return {"error": "Insufficient data for attribution"}

        strat_ret, bench_ret = aligned.iloc[:, 0], aligned.iloc[:, 1]

        # Market regression
        beta, alpha_daily, r_sq, p_val, se = stats.linregress(bench_ret, strat_ret)
        alpha_annual = alpha_daily * 252

        market_contribution = float(beta * bench_ret.mean() * 252)
        alpha_contribution = float(alpha_annual)
        total_return = float(strat_ret.mean() * 252)

        attribution = {
            "total_annual_return": total_return,
            "market_contribution": market_contribution,
            "alpha_contribution": alpha_contribution,
            "beta": float(beta),
            "r_squared": float(r_sq),
            "alpha_tstat": float(alpha_daily / (se + 1e-10) * np.sqrt(252)),
            "alpha_significant": float(abs(alpha_daily / (se + 1e-10) * np.sqrt(252))) > 2.0,
        }

        # Factor attribution if factor returns provided
        if factor_returns is not None:
            factor_aligned = factor_returns.reindex(aligned.index).dropna()
            if not factor_aligned.empty:
                X = factor_aligned.values
                y = strat_ret.values
                try:
                    from sklearn.linear_model import Ridge
                    model = Ridge(alpha=0.01)
                    model.fit(X, y)
                    factor_betas = dict(zip(factor_returns.columns, model.coef_))
                    attribution["factor_betas"] = factor_betas
                    residual_vol = float(np.std(y - model.predict(X)) * np.sqrt(252))
                    attribution["residual_volatility"] = residual_vol
                except Exception:
                    pass

        return attribution


def generate_full_report(
    equity_curve: pd.Series,
    trades_df: Optional[pd.DataFrame] = None,
    signals: Optional[pd.DataFrame] = None,
    prices: Optional[pd.DataFrame] = None,
    benchmark: Optional[pd.Series] = None,
    n_strategies_tested: int = 1,
    initial_capital: float = 100_000,
) -> Dict:
    """
    Generate a comprehensive report combining all analytics modules.
    """
    from reporting.analytics import PerformanceAnalytics
    analytics = PerformanceAnalytics()
    advanced = AdvancedAnalytics()

    report = {}

    # Core metrics
    core = analytics.compute_all(
        equity_curve=equity_curve,
        trades=trades_df,
        benchmark=benchmark,
        n_strategies_tested=n_strategies_tested,
    )
    report["core_metrics"] = core

    # Monte Carlo confidence intervals
    daily_returns = equity_curve.pct_change().dropna()
    if len(daily_returns) >= 30:
        report["monte_carlo"] = advanced.monte_carlo_sharpe(daily_returns)

    # Turnover analysis
    if signals is not None and prices is not None:
        report["turnover"] = advanced.turnover_analysis(signals, prices)

    # Return attribution
    if benchmark is not None:
        report["attribution"] = advanced.return_attribution(equity_curve, benchmark)

    # Summary verdict
    sharpe = core.get("sharpe_ratio", 0)
    dsr = core.get("deflated_sharpe_ratio", -1)
    max_dd = core.get("max_drawdown", -1)
    mc = report.get("monte_carlo", {})
    ci_low = mc.get("sharpe_ci_low", -99)

    verdict_lines = []
    if dsr > 0:
        verdict_lines.append(f"✓ Statistically significant (DSR={dsr:.2f})")
    else:
        verdict_lines.append(f"✗ Not statistically significant (DSR={dsr:.2f})")

    if sharpe > 0.5:
        verdict_lines.append(f"✓ Good Sharpe ({sharpe:.2f})")
    elif sharpe > 0:
        verdict_lines.append(f"△ Weak Sharpe ({sharpe:.2f})")
    else:
        verdict_lines.append(f"✗ Negative Sharpe ({sharpe:.2f})")

    if ci_low > 0:
        verdict_lines.append(f"✓ 95% CI lower bound positive ({ci_low:.2f})")
    elif ci_low > -99:
        verdict_lines.append(f"✗ 95% CI includes zero (CI low={ci_low:.2f})")

    if abs(max_dd) < 0.20:
        verdict_lines.append(f"✓ Max drawdown acceptable ({max_dd:.1%})")
    else:
        verdict_lines.append(f"△ Large max drawdown ({max_dd:.1%})")

    report["verdict"] = {
        "lines": verdict_lines,
        "proceed": dsr > 0 and sharpe > 0.3,
        "summary": " | ".join(verdict_lines),
    }

    return report
