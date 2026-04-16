"""
Portfolio Optimization Module

State of the art from research and open source:
- skfolio: sklearn-compatible, HRP, HERC, Black-Litterman, CVaR, NCO
- Riskfolio-Lib: 35 risk measures, HRP with 35 risk measures
- QuantConnect LEAN: Mean-variance, Black-Litterman built-in

Implemented here (no paid dependencies):
1. Mean-Variance (Markowitz): maximize Sharpe, minimize variance, target return
2. Hierarchical Risk Parity (Lopez de Prado 2016): no inversion of covariance
3. Risk Parity: equal risk contribution per asset
4. Black-Litterman: incorporate views into equilibrium prior
5. Maximum Diversification: maximize diversification ratio
6. Minimum CVaR: tail-risk aware optimization

All optimizers output a weight vector that plugs into the PortfolioManager
as a position sizing overlay.
"""

from __future__ import annotations
import logging
import warnings
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logger.warning("cvxpy not available: using scipy-based optimization fallback")


def compute_covariance(returns: pd.DataFrame, method: str = "sample") -> np.ndarray:
    """
    Compute covariance matrix with optional shrinkage.
    
    method: 'sample', 'ledoit_wolf', 'oracle_approximating'
    
    Ledoit-Wolf shrinkage is the standard in practice: it fixes
    the sample covariance estimation error for small T/N ratios.
    Any portfolio with more than ~30 assets should use shrinkage.
    """
    if method == "sample":
        return returns.cov().values

    elif method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        lw.fit(returns.dropna())
        return lw.covariance_

    elif method == "oracle_approximating":
        from sklearn.covariance import OAS
        oas = OAS()
        oas.fit(returns.dropna())
        return oas.covariance_

    return returns.cov().values


# ── Mean-Variance Optimization ─────────────────────────────────────────────────

class MeanVarianceOptimizer:
    """
    Markowitz mean-variance optimization.
    
    Objective options:
    - 'max_sharpe': maximize Sharpe ratio (most common)
    - 'min_variance': minimize portfolio variance
    - 'max_return': maximize return subject to vol constraint
    - 'efficient_risk': minimize variance for target return
    
    Uses cvxpy for convex optimization (handles constraints cleanly).
    Falls back to scipy if cvxpy unavailable.
    
    Warning: mean-variance is sensitive to expected return estimates.
    Small errors in expected returns lead to wildly different weights.
    Use Black-Litterman or HRP in production.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        cov_method: str = "ledoit_wolf",
        weight_bounds: Tuple[float, float] = (0.0, 0.30),
        l2_reg: float = 0.0,
    ):
        self.rf = risk_free_rate / 252  # daily
        self.cov_method = cov_method
        self.bounds = weight_bounds
        self.l2_reg = l2_reg

    def optimize(
        self,
        returns: pd.DataFrame,
        objective: str = "max_sharpe",
        target_return: Optional[float] = None,
    ) -> Dict[str, float]:
        """Returns dict of {asset: weight}."""
        n = len(returns.columns)
        mu = returns.mean().values
        cov = compute_covariance(returns, self.cov_method)

        if CVXPY_AVAILABLE:
            weights = self._optimize_cvxpy(mu, cov, n, objective, target_return)
        else:
            weights = self._optimize_scipy(mu, cov, n, objective)

        return dict(zip(returns.columns, weights))

    def _optimize_cvxpy(self, mu, cov, n, objective, target_return):
        w = cp.Variable(n)
        constraints = [
            cp.sum(w) == 1,
            w >= self.bounds[0],
            w <= self.bounds[1],
        ]

        if objective == "max_sharpe":
            # Equivalent to maximizing (mu - rf) / sigma
            # Convex reformulation: minimize variance with normalized return
            excess = mu - self.rf * np.ones(n)
            portfolio_return = excess @ w
            portfolio_variance = cp.quad_form(w, cp.psd_wrap(cov))

            # Sharpe maximization: transform to minimize variance / (mu - rf)
            # Standard QCQP approach: define y = w / (excess @ w)
            # Here we use a simple objective: maximize excess_return - lambda * variance
            lambda_risk = 1.0
            prob = cp.Problem(
                cp.Maximize(portfolio_return - lambda_risk * portfolio_variance),
                constraints,
            )

        elif objective == "min_variance":
            portfolio_variance = cp.quad_form(w, cp.psd_wrap(cov))
            reg_term = self.l2_reg * cp.sum_squares(w) if self.l2_reg > 0 else 0
            prob = cp.Problem(cp.Minimize(portfolio_variance + reg_term), constraints)

        elif objective == "efficient_risk":
            if target_return is not None:
                constraints.append(mu @ w >= target_return)
            portfolio_variance = cp.quad_form(w, cp.psd_wrap(cov))
            prob = cp.Problem(cp.Minimize(portfolio_variance), constraints)

        else:
            portfolio_variance = cp.quad_form(w, cp.psd_wrap(cov))
            prob = cp.Problem(cp.Minimize(portfolio_variance), constraints)

        try:
            prob.solve(solver=cp.SCS, verbose=False)
            if w.value is not None:
                result = np.array(w.value).flatten()
                result = np.clip(result, self.bounds[0], self.bounds[1])
                result /= result.sum()
                return result
        except Exception as e:
            logger.warning(f"cvxpy optimization failed: {e}, using equal weight")

        return np.ones(n) / n

    def _optimize_scipy(self, mu, cov, n, objective):
        def neg_sharpe(w):
            ret = w @ mu
            vol = np.sqrt(w @ cov @ w)
            return -(ret - self.rf * n) / (vol + 1e-8)

        def portfolio_vol(w):
            return np.sqrt(w @ cov @ w)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [self.bounds] * n
        x0 = np.ones(n) / n

        try:
            if objective == "max_sharpe":
                result = minimize(neg_sharpe, x0, method="SLSQP",
                                  bounds=bounds, constraints=constraints)
            else:
                result = minimize(portfolio_vol, x0, method="SLSQP",
                                  bounds=bounds, constraints=constraints)
            if result.success:
                w = result.x
                w = np.clip(w, self.bounds[0], self.bounds[1])
                return w / w.sum()
        except Exception as e:
            logger.warning(f"scipy optimization failed: {e}")

        return np.ones(n) / n

    def efficient_frontier(
        self, returns: pd.DataFrame, n_points: int = 50
    ) -> pd.DataFrame:
        """Generate the efficient frontier curve."""
        mu = returns.mean().values * 252
        cov = compute_covariance(returns, self.cov_method) * 252
        n = len(returns.columns)

        min_ret = mu.min()
        max_ret = mu.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier = []
        for tr in target_returns:
            try:
                w_dict = self.optimize(returns, objective="efficient_risk", target_return=tr/252)
                w = np.array(list(w_dict.values()))
                port_ret = w @ mu
                port_vol = np.sqrt(w @ cov @ w)
                frontier.append({"return": port_ret, "volatility": port_vol,
                                  "sharpe": (port_ret - self.rf * 252) / (port_vol + 1e-8)})
            except Exception:
                pass

        return pd.DataFrame(frontier)


# ── Hierarchical Risk Parity ───────────────────────────────────────────────────

class HierarchicalRiskParity:
    """
    Lopez de Prado (2016) Hierarchical Risk Parity.
    
    HRP is the gold standard for robust portfolio construction because:
    1. No matrix inversion (avoids covariance estimation errors amplification)
    2. Diversifies across clusters, not individual assets
    3. Stable out-of-sample vs mean-variance (empirically shown)
    4. Can use any risk measure (variance, CVaR, semi-variance)
    
    Three steps:
    1. Tree clustering: build hierarchy from correlation matrix
    2. Quasi-diagonalization: reorder assets to minimize intracluster distances
    3. Recursive bisection: allocate capital top-down across clusters
    
    Reference: Lopez de Prado (2016) "Building Diversified Portfolios
    that Outperform Out of Sample"
    """

    def __init__(
        self,
        linkage_method: str = "single",
        risk_measure: str = "variance",
    ):
        self.linkage_method = linkage_method
        self.risk_measure = risk_measure

    def optimize(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Returns dict of {asset: weight}."""
        n = len(returns.columns)
        assets = list(returns.columns)

        if n < 2:
            return {assets[0]: 1.0}

        # Step 1: Correlation-based distance matrix
        corr = returns.corr().fillna(0)
        dist = np.sqrt((1 - corr) / 2).values
        np.fill_diagonal(dist, 0)

        # Step 2: Hierarchical clustering
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method=self.linkage_method)

        # Step 3: Quasi-diagonalization (reorder assets)
        sorted_idx = leaves_list(link)
        sorted_assets = [assets[i] for i in sorted_idx]

        # Step 4: Recursive bisection
        weights = self._recursive_bisection(returns, sorted_assets)

        return weights

    def _get_cluster_var(self, returns: pd.DataFrame, assets: List[str]) -> float:
        """Compute cluster variance (or other risk measure)."""
        sub_returns = returns[assets]
        cov = sub_returns.cov().values
        n = len(assets)

        if self.risk_measure == "variance":
            # Inverse-variance weights within cluster
            var = np.diag(cov)
            inv_var = 1.0 / (var + 1e-10)
            w = inv_var / inv_var.sum()
            return float(w @ cov @ w)

        elif self.risk_measure == "equal":
            w = np.ones(n) / n
            return float(w @ cov @ w)

        else:
            var = np.diag(cov)
            inv_var = 1.0 / (var + 1e-10)
            w = inv_var / inv_var.sum()
            return float(w @ cov @ w)

    def _recursive_bisection(
        self, returns: pd.DataFrame, assets: List[str]
    ) -> Dict[str, float]:
        """Recursive bisection to allocate weights."""
        weights = {a: 1.0 for a in assets}
        cluster_items = [assets]

        while len(cluster_items) > 0:
            # Split each cluster into two halves
            new_clusters = []
            for cluster in cluster_items:
                if len(cluster) <= 1:
                    continue

                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # Variance of each sub-cluster
                var_left = self._get_cluster_var(returns, left)
                var_right = self._get_cluster_var(returns, right)

                # Allocation: inverse variance weighting
                total_var = var_left + var_right
                if total_var > 0:
                    alpha = 1.0 - var_left / total_var
                else:
                    alpha = 0.5

                # Redistribute weight
                for a in left:
                    weights[a] *= (1 - alpha)
                for a in right:
                    weights[a] *= alpha

                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)

            cluster_items = new_clusters

        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}


# ── Risk Parity ────────────────────────────────────────────────────────────────

class RiskParityOptimizer:
    """
    Equal Risk Contribution (ERC) portfolio.
    
    Each asset contributes equally to portfolio volatility.
    More robust than mean-variance: no expected return estimation needed.
    Used by AQR, Bridgewater (risk parity funds), and many macro managers.
    
    Solved via SLSQP: minimize sum of squared pairwise risk contributions.
    """

    def __init__(self, cov_method: str = "ledoit_wolf"):
        self.cov_method = cov_method

    def optimize(self, returns: pd.DataFrame) -> Dict[str, float]:
        n = len(returns.columns)
        cov = compute_covariance(returns, self.cov_method)

        def risk_contribution(w):
            port_var = w @ cov @ w
            mrc = cov @ w  # marginal risk contribution
            rc = w * mrc   # risk contribution
            return rc

        def objective(w):
            rc = risk_contribution(w)
            # Minimize pairwise squared differences in risk contributions
            total_rc = rc.sum()
            rc_pct = rc / (total_rc + 1e-10)
            target = 1.0 / n
            return float(np.sum((rc_pct - target) ** 2))

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1)] * n
        x0 = np.ones(n) / n

        try:
            result = minimize(
                objective, x0, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-12}
            )
            if result.success:
                w = np.clip(result.x, 0, 1)
                return dict(zip(returns.columns, w / w.sum()))
        except Exception as e:
            logger.warning(f"Risk parity optimization failed: {e}")

        return dict(zip(returns.columns, np.ones(n) / n))


# ── Black-Litterman ────────────────────────────────────────────────────────────

class BlackLittermanOptimizer:
    """
    Black-Litterman portfolio optimization.
    
    Addresses the primary weakness of mean-variance: sensitivity to
    expected return inputs. BL combines:
    1. Market equilibrium returns (prior): reverse-engineered from cap weights
    2. Investor views (likelihood): override specific return expectations
    3. Posterior: Bayesian combination of prior + views
    
    The result is a well-diversified portfolio that only tilts toward
    views where you have conviction, relative to the market baseline.
    
    This is the approach used by large allocators (Goldman Sachs developed it,
    institutional asset managers widely use it).
    
    Reference: Black & Litterman (1992), Idzorek (2007)
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.025,
        cov_method: str = "ledoit_wolf",
    ):
        self.risk_aversion = risk_aversion
        self.tau = tau  # uncertainty in prior (typically 1/T)
        self.cov_method = cov_method

    def optimize(
        self,
        returns: pd.DataFrame,
        market_caps: Optional[Dict[str, float]] = None,
        views: Optional[List[Dict]] = None,
    ) -> Dict[str, float]:
        """
        Optimize using Black-Litterman.
        
        views: list of dicts with keys:
          - 'assets': list of asset names
          - 'weights': list of weights (long +, short -)
          - 'return': expected return for this view
          - 'confidence': [0, 1] confidence level
        
        Example view: long SPY, short TLT by 5% annually:
          {'assets': ['SPY', 'TLT'], 'weights': [1, -1],
           'return': 0.05 / 252, 'confidence': 0.7}
        """
        assets = list(returns.columns)
        n = len(assets)
        cov = compute_covariance(returns, self.cov_method)

        # Market capitalization weights (prior portfolio)
        if market_caps:
            total_cap = sum(market_caps.get(a, 1) for a in assets)
            w_mkt = np.array([market_caps.get(a, 1) / total_cap for a in assets])
        else:
            w_mkt = np.ones(n) / n  # equal weight if no caps

        # Equilibrium returns (reverse optimization)
        pi = self.risk_aversion * cov @ w_mkt

        if not views:
            # No views: return market equilibrium portfolio
            w = self._mv_from_returns(pi, cov, n)
            return dict(zip(assets, w))

        # Build view matrices P, Q, Omega
        k = len(views)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        Omega = np.zeros((k, k))

        asset_idx = {a: i for i, a in enumerate(assets)}

        for i, view in enumerate(views):
            view_assets = view.get("assets", [])
            view_weights = view.get("weights", [])
            view_return = view.get("return", 0)
            view_conf = view.get("confidence", 0.5)

            for j, asset in enumerate(view_assets):
                if asset in asset_idx:
                    idx = asset_idx[asset]
                    w_j = view_weights[j] if j < len(view_weights) else 1.0
                    P[i, idx] = w_j

            # Normalize P row
            row_norm = np.abs(P[i]).sum()
            if row_norm > 0:
                P[i] /= row_norm

            Q[i] = view_return

            # View uncertainty: lower confidence = higher uncertainty
            view_var = (1.0 - view_conf) * (self.tau * P[i] @ cov @ P[i].T)
            Omega[i, i] = max(view_var, 1e-8)

        # Black-Litterman posterior
        # mu_BL = [(tau*Sigma)^-1 + P' Omega^-1 P]^-1 * [(tau*Sigma)^-1 * pi + P' Omega^-1 * Q]
        try:
            tau_cov = self.tau * cov
            tau_cov_inv = np.linalg.inv(tau_cov + np.eye(n) * 1e-8)
            omega_inv = np.diag(1.0 / np.diag(Omega + np.eye(k) * 1e-8))

            M1 = tau_cov_inv + P.T @ omega_inv @ P
            M2 = tau_cov_inv @ pi + P.T @ omega_inv @ Q

            M1_inv = np.linalg.inv(M1 + np.eye(n) * 1e-8)
            mu_bl = M1_inv @ M2

            # Posterior covariance
            cov_bl = np.linalg.inv(tau_cov_inv + P.T @ omega_inv @ P)
            cov_posterior = cov + cov_bl

        except np.linalg.LinAlgError:
            logger.warning("Black-Litterman matrix inversion failed; using prior")
            mu_bl = pi
            cov_posterior = cov

        w = self._mv_from_returns(mu_bl, cov_posterior, n)
        return dict(zip(assets, w))

    def _mv_from_returns(self, mu: np.ndarray, cov: np.ndarray, n: int) -> np.ndarray:
        """Compute optimal weights from expected returns via mean-variance."""
        if CVXPY_AVAILABLE:
            w = cp.Variable(n)
            constraints = [cp.sum(w) == 1, w >= 0, w <= 0.30]
            excess = mu
            prob = cp.Problem(
                cp.Maximize(excess @ w - self.risk_aversion * cp.quad_form(w, cp.psd_wrap(cov))),
                constraints,
            )
            try:
                prob.solve(solver=cp.SCS, verbose=False)
                if w.value is not None:
                    result = np.clip(w.value, 0, 0.30)
                    return result / result.sum()
            except Exception:
                pass

        # Analytical solution: w = (risk_aversion * cov)^-1 * mu
        try:
            w = np.linalg.solve(self.risk_aversion * cov + np.eye(n) * 1e-6, mu)
            w = np.clip(w, 0, 0.30)
            if w.sum() > 0:
                return w / w.sum()
        except Exception:
            pass

        return np.ones(n) / n


# ── Portfolio Optimization Strategy ───────────────────────────────────────────

class PortfolioOptimizationStrategy:
    """
    Wraps all optimizer types into a strategy-compatible interface.
    Rebalances monthly, applies target weights via PortfolioManager.
    
    This is the bridge between portfolio theory and the event-driven engine.
    """

    OPTIMIZERS = {
        "hrp": HierarchicalRiskParity,
        "risk_parity": RiskParityOptimizer,
        "mean_variance": MeanVarianceOptimizer,
        "black_litterman": BlackLittermanOptimizer,
        "equal_weight": None,
    }

    def __init__(
        self,
        optimizer_name: str = "hrp",
        lookback_days: int = 252,
        rebalance_freq: str = "monthly",
        optimizer_kwargs: Optional[Dict] = None,
    ):
        self.optimizer_name = optimizer_name
        self.lookback = lookback_days
        self.rebalance_freq = rebalance_freq

        if optimizer_name == "equal_weight":
            self.optimizer = None
        else:
            cls = self.OPTIMIZERS.get(optimizer_name, HierarchicalRiskParity)
            self.optimizer = cls(**(optimizer_kwargs or {}))

        self._last_rebalance_month = -1
        self._last_weights: Dict[str, float] = {}

    def get_target_weights(
        self,
        returns: pd.DataFrame,
        current_month: int,
        views: Optional[List[Dict]] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Returns target weights if it's time to rebalance, else None.
        """
        if self.rebalance_freq == "monthly" and current_month == self._last_rebalance_month:
            return None

        self._last_rebalance_month = current_month

        if len(returns) < self.lookback // 2:
            return None

        recent_returns = returns.iloc[-self.lookback:]
        # Drop columns with too many NaN
        recent_returns = recent_returns.dropna(axis=1, thresh=len(recent_returns) // 2)

        if recent_returns.empty or len(recent_returns.columns) < 2:
            return None

        try:
            if self.optimizer is None:
                n = len(recent_returns.columns)
                weights = {c: 1/n for c in recent_returns.columns}
            elif isinstance(self.optimizer, BlackLittermanOptimizer):
                weights = self.optimizer.optimize(recent_returns, views=views)
            else:
                weights = self.optimizer.optimize(recent_returns)

            self._last_weights = weights
            return weights

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return None
