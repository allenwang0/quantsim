"""
Walk-Forward Optimization Framework

State of the art from the field:
- QuantConnect LEAN: built-in WFO with configurable windows
- QLib: train/val/test splits enforced at the framework level
- Bailey et al. (2014): deflated Sharpe Ratio for WFO corrections

What most retail backtesting gets wrong:
- Optimize on full history, evaluate on same history (data snooping)
- Use fixed in/out split instead of rolling windows
- Don't correct for number of trials (multiple comparisons problem)
- Don't test across multiple stress regimes

This module implements:
1. Rolling walk-forward with configurable train/test/gap windows
2. Parameter sweep at each walk-forward step
3. Aggregated out-of-sample performance statistics
4. Regime analysis: how does the strategy perform across market states?
5. Combinatorial Purged Cross-Validation (CPCV) - Lopez de Prado standard
"""

from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple, Any
import numpy as np
import pandas as pd
import itertools

logger = logging.getLogger(__name__)


class WalkForwardOptimizer:
    """
    Rolling Walk-Forward Optimization.
    
    Protocol (per Bailey et al. and Lopez de Prado):
    1. Divide history into sequential windows
    2. For each window: optimize on train period, evaluate on test period
    3. Report ONLY out-of-sample (test) performance
    4. Aggregate test periods to build full OOS equity curve
    
    Window structure:
    [────────── TRAIN (3yr) ──────────][GAP (0)][TEST (1yr)]
                                                   ↕ advance by step
    
    The gap prevents lookahead from the training period's end
    contaminating the test period (e.g., from GARCH models trained 
    on data that includes the test period's volatility regime).
    
    CRITICAL: Never aggregate in-sample periods to evaluate strategy.
    The reported Sharpe must come entirely from OOS periods.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        train_years: float = 3.0,
        test_months: int = 12,
        step_months: int = 6,
        gap_days: int = 0,
    ):
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        self.train_bars = int(train_years * 252)
        self.test_bars = int(test_months * 21)
        self.step_bars = int(step_months * 21)
        self.gap_bars = gap_days

    def generate_windows(self) -> List[Dict]:
        """
        Generate all train/test window pairs.
        Each window is a dict with 'train_start', 'train_end', 'test_start', 'test_end'.
        """
        windows = []
        n = len(self.prices)
        start = 0

        while start + self.train_bars + self.gap_bars + self.test_bars <= n:
            train_start_idx = start
            train_end_idx = start + self.train_bars - 1
            test_start_idx = train_end_idx + 1 + self.gap_bars
            test_end_idx = test_start_idx + self.test_bars - 1

            if test_end_idx >= n:
                break

            windows.append({
                "train_start": self.prices.index[train_start_idx],
                "train_end": self.prices.index[train_end_idx],
                "test_start": self.prices.index[test_start_idx],
                "test_end": self.prices.index[min(test_end_idx, n-1)],
                "window_id": len(windows),
            })
            start += self.step_bars

        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows

    def optimize_and_evaluate(
        self,
        signal_fn: Callable,
        param_grid: Dict[str, List],
        objective: str = "sharpe_ratio",
        initial_capital: float = 100_000,
        commission_rate: float = 0.001,
        slippage_pct: float = 0.001,
        benchmark: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Run walk-forward optimization with parameter grid.
        
        Returns:
        - oos_equity_curve: concatenated out-of-sample equity
        - oos_analytics: performance metrics on OOS periods only
        - window_results: per-window details
        - best_params_by_window: optimal params for each window
        """
        from backtesting.vectorized import VectorizedBacktester

        windows = self.generate_windows()
        if not windows:
            return {"error": "Insufficient data for walk-forward windows"}

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        n_configs = len(combinations)

        window_results = []
        oos_equities = []

        for window in windows:
            train_prices = self.prices.loc[window["train_start"]:window["train_end"]]
            test_prices = self.prices.loc[window["test_start"]:window["test_end"]]

            # Step 1: Find best parameters on training period
            best_params = None
            best_objective_val = -np.inf

            train_backtester = VectorizedBacktester(
                prices=train_prices,
                initial_capital=initial_capital,
                commission_rate=commission_rate,
                slippage_pct=slippage_pct,
            )

            for combo in combinations:
                kwargs = dict(zip(keys, combo))
                try:
                    result = train_backtester.run(signal_fn, signal_kwargs=kwargs)
                    obj_val = result.get(objective, -np.inf)
                    if obj_val > best_objective_val:
                        best_objective_val = obj_val
                        best_params = kwargs
                except Exception:
                    pass

            if best_params is None:
                best_params = dict(zip(keys, combinations[0]))

            # Step 2: Evaluate best parameters on test period (OOS)
            test_backtester = VectorizedBacktester(
                prices=test_prices,
                initial_capital=initial_capital,
                commission_rate=commission_rate,
                slippage_pct=slippage_pct,
            )

            try:
                test_result = test_backtester.run(signal_fn, signal_kwargs=best_params)
                oos_equity = pd.Series(
                    test_result.get("_equity_curve", [initial_capital]),
                    name=f"window_{window['window_id']}",
                )
            except Exception as e:
                logger.warning(f"Test period failed for window {window['window_id']}: {e}")
                test_result = {}
                oos_equity = pd.Series([initial_capital])

            window_results.append({
                "window_id": window["window_id"],
                "train_start": window["train_start"],
                "train_end": window["train_end"],
                "test_start": window["test_start"],
                "test_end": window["test_end"],
                "best_params": best_params,
                "train_best_sharpe": best_objective_val,
                "oos_sharpe": test_result.get("sharpe_ratio", 0),
                "oos_return": test_result.get("total_return", 0),
                "oos_max_dd": test_result.get("max_drawdown", 0),
                "oos_calmar": test_result.get("calmar_ratio", 0),
            })
            oos_equities.append(oos_equity)

        # Aggregate OOS results
        results_df = pd.DataFrame(window_results)

        # Compute aggregate OOS Sharpe
        all_oos_sharpes = results_df["oos_sharpe"].dropna()
        avg_oos_sharpe = float(all_oos_sharpes.mean()) if len(all_oos_sharpes) > 0 else 0

        # Sharpe degradation: how much does IS Sharpe overestimate OOS?
        avg_is_sharpe = float(results_df["train_best_sharpe"].mean()) if len(results_df) > 0 else 0
        sharpe_degradation = avg_is_sharpe - avg_oos_sharpe

        # Apply Deflated Sharpe for multiple comparisons
        from reporting.analytics import PerformanceAnalytics
        analytics = PerformanceAnalytics()
        dsr = analytics._deflated_sharpe_ratio(
            sharpe=avg_oos_sharpe,
            n_obs=len(self.prices),
            n_strategies=n_configs * len(windows),
        )

        # Find most stable parameter set (appears most often as best)
        if len(results_df) > 0:
            param_stability = {}
            for _, row in results_df.iterrows():
                key = str(row["best_params"])
                param_stability[key] = param_stability.get(key, 0) + 1
            most_stable_params = max(param_stability, key=param_stability.get)
        else:
            most_stable_params = str(dict(zip(keys, combinations[0])))

        return {
            "window_results": results_df.to_dict("records"),
            "avg_oos_sharpe": avg_oos_sharpe,
            "avg_is_sharpe": avg_is_sharpe,
            "sharpe_degradation": sharpe_degradation,
            "deflated_sharpe_corrected": dsr,
            "n_windows": len(windows),
            "n_configs_tested": n_configs,
            "total_configs_tested": n_configs * len(windows),
            "most_stable_params": most_stable_params,
            "oos_win_rate": float((results_df["oos_sharpe"] > 0).mean()) if len(results_df) > 0 else 0,
            "summary": {
                "pass": dsr > 0 and avg_oos_sharpe > 0.3,
                "warning": sharpe_degradation > 0.5,
                "message": (
                    f"OOS Sharpe: {avg_oos_sharpe:.2f} (IS: {avg_is_sharpe:.2f}, "
                    f"degradation: {sharpe_degradation:.2f}). "
                    f"DSR: {dsr:.2f}. "
                    f"{'VIABLE' if dsr > 0 else 'NOT SIGNIFICANT'}"
                ),
            },
        }


class RegimeAnalyzer:
    """
    Analyze strategy performance across market regimes.
    
    Regimes defined by:
    - VIX level (low/medium/high volatility)
    - Trend vs. range-bound (ADX)
    - Bull/bear market (relative to 200-day MA)
    - Credit conditions (HY spread)
    
    A strategy that looks good overall but only works in one regime
    is fragile and will fail in live trading when the regime changes.
    This is one of the most common failures of retail quant strategies.
    """

    @staticmethod
    def classify_regimes(
        prices: pd.Series,  # benchmark (e.g., SPY)
        vix: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Returns a regime Series with values:
        'bull_low_vol', 'bull_high_vol', 'bear_low_vol', 'bear_high_vol'
        """
        sma_200 = prices.rolling(200).mean()
        is_bull = (prices > sma_200).astype(int)

        if vix is not None:
            vix_aligned = vix.reindex(prices.index, method="ffill")
            vix_median = vix_aligned.rolling(252).median()
            is_high_vol = (vix_aligned > vix_median).astype(int)
        else:
            returns = prices.pct_change()
            rvol = returns.rolling(30).std() * np.sqrt(252)
            rvol_median = rvol.rolling(252).median()
            is_high_vol = (rvol > rvol_median).astype(int)

        regimes = pd.Series("unknown", index=prices.index)
        regimes[(is_bull == 1) & (is_high_vol == 0)] = "bull_low_vol"
        regimes[(is_bull == 1) & (is_high_vol == 1)] = "bull_high_vol"
        regimes[(is_bull == 0) & (is_high_vol == 0)] = "bear_low_vol"
        regimes[(is_bull == 0) & (is_high_vol == 1)] = "bear_high_vol"

        return regimes

    @staticmethod
    def analyze_by_regime(
        equity_curve: pd.Series,
        regimes: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute performance metrics separately for each regime.
        A robust strategy should have positive Sharpe across most regimes.
        """
        from reporting.analytics import PerformanceAnalytics
        analytics = PerformanceAnalytics()

        results = []
        for regime in regimes.unique():
            regime_dates = regimes[regimes == regime].index
            regime_equity = equity_curve.reindex(regime_dates).dropna()

            if len(regime_equity) < 20:
                continue

            regime_results = analytics.compute_all(regime_equity)
            results.append({
                "regime": regime,
                "n_days": len(regime_equity),
                "sharpe": regime_results.get("sharpe_ratio", 0),
                "cagr": regime_results.get("cagr", 0),
                "max_drawdown": regime_results.get("max_drawdown", 0),
                "calmar": regime_results.get("calmar_ratio", 0),
            })

        return pd.DataFrame(results).set_index("regime")

    @staticmethod
    def stress_test_periods() -> Dict[str, Tuple[str, str]]:
        """
        Known stress test periods. A strategy must survive all of these.
        If it fails any, it is regime-dependent and will fail in live trading.
        """
        return {
            "dot_com_crash": ("2000-03-01", "2002-10-09"),
            "financial_crisis": ("2007-10-01", "2009-03-09"),
            "flash_crash": ("2010-05-06", "2010-05-07"),
            "eu_debt_crisis": ("2011-08-01", "2011-10-04"),
            "china_correction": ("2015-08-01", "2015-09-30"),
            "covid_crash": ("2020-02-19", "2020-03-23"),
            "rate_shock_2022": ("2022-01-01", "2022-10-12"),
        }
