"""
Options chain reconstruction for historical backtesting.
Since no free source provides historical chains, we reconstruct them from:
  1. Historical underlying prices (yfinance)
  2. IV estimate = realized_vol_30d * scaling_factor
  3. BSM (py_vollib) for all Greeks and prices

This introduces IV reconstruction error of ~20-30%. Results are indicative,
not precise. Strategies with wide margin of safety (PMCC, deep OTM spreads)
are more robust to this error.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)

# Try to import py_vollib; fall back to pure numpy BSM if unavailable
try:
    from py_vollib.black_scholes import black_scholes as bsm_price
    from py_vollib.black_scholes.greeks.analytical import (
        delta as bsm_delta,
        gamma as bsm_gamma,
        theta as bsm_theta,
        vega as bsm_vega,
        rho as bsm_rho,
    )
    from py_vollib.black_scholes.implied_volatility import implied_volatility
    PY_VOLLIB_AVAILABLE = True
except ImportError:
    PY_VOLLIB_AVAILABLE = False
    logger.warning("py_vollib not available; using numpy BSM fallback")


def norm_cdf(x: float) -> float:
    from scipy.stats import norm
    return norm.cdf(x)


def bsm_price_numpy(S: float, K: float, T: float, r: float, sigma: float, flag: str) -> float:
    """Pure numpy BSM price. flag: 'c' or 'p'."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if flag == 'c' else max(K - S, 0)
        return intrinsic

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if flag == 'c':
        price = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    return max(price, 0.0)


def bsm_greeks_numpy(S: float, K: float, T: float, r: float, sigma: float, flag: str) -> Dict:
    """Compute all BSM Greeks with pure numpy."""
    from scipy.stats import norm

    if T <= 0:
        intrinsic = max(S - K, 0) if flag == 'c' else max(K - S, 0)
        return {
            "delta": 1.0 if (flag == 'c' and S > K) else (-1.0 if (flag == 'p' and S < K) else 0.0),
            "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0,
            "price": intrinsic,
        }

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = bsm_price_numpy(S, K, T, r, sigma, flag)
    delta = norm.cdf(d1) if flag == 'c' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta_annual = (
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * (norm.cdf(d2) if flag == 'c' else norm.cdf(-d2))
        + (0 if flag == 'c' else r * K * np.exp(-r * T))
    )
    theta_daily = theta_annual / 365.0
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% IV change
    rho_val = (
        K * T * np.exp(-r * T) * norm.cdf(d2) / 100 if flag == 'c'
        else -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    )

    return {
        "delta": delta, "gamma": gamma, "theta": theta_daily,
        "vega": vega, "rho": rho_val, "price": price,
    }


def compute_greeks(
    S: float, K: float, T_years: float, r: float, sigma: float, flag: str
) -> Dict:
    """
    Compute BSM option price and Greeks.
    Uses py_vollib if available; falls back to numpy implementation.
    flag: 'c' (call) or 'p' (put)
    """
    flag = flag.lower()
    try:
        if PY_VOLLIB_AVAILABLE and T_years > 0 and sigma > 0:
            price = bsm_price(flag, S, K, T_years, r, sigma)
            d = bsm_delta(flag, S, K, T_years, r, sigma)
            g = bsm_gamma(flag, S, K, T_years, r, sigma)
            t = bsm_theta(flag, S, K, T_years, r, sigma)
            v = bsm_vega(flag, S, K, T_years, r, sigma) / 100
            rh = bsm_rho(flag, S, K, T_years, r, sigma) / 100
            return {
                "delta": d, "gamma": g, "theta": t,
                "vega": v, "rho": rh, "price": price,
            }
    except Exception:
        pass
    return bsm_greeks_numpy(S, K, T_years, r, sigma, flag)


def estimate_iv_from_history(
    prices: pd.Series,
    vix_series: Optional[pd.Series] = None,
    window: int = 30,
    premium_mult: float = 1.15,
) -> pd.Series:
    """
    Estimate historical IV at each date using realized volatility + scaling.
    
    Method:
    - Compute 30-day trailing realized vol on log returns
    - Scale by (VIX / SPY_realized_vol) ratio if VIX data available
    - For non-SPY names, apply a flat premium_mult = 1.15
    
    IV reconstruction error: ~20-30%. Results are indicative.
    """
    log_returns = np.log(prices / prices.shift(1))
    realized_vol = log_returns.rolling(window).std() * np.sqrt(252)

    if vix_series is not None:
        # Align VIX to our price series
        aligned_vix = vix_series.reindex(prices.index, method="ffill")
        vix_annual = aligned_vix / 100.0  # VIX is in percentage points

        # SPY realized vol for the ratio denominator
        spy_rv = log_returns.rolling(20).std() * np.sqrt(252)
        spy_rv = spy_rv.clip(lower=0.05)  # avoid division by zero

        iv_ratio = (vix_annual / spy_rv).clip(0.5, 3.0)
        iv_estimate = realized_vol * iv_ratio
    else:
        iv_estimate = realized_vol * premium_mult

    return iv_estimate.clip(lower=0.05, upper=3.0)


def reconstruct_chain(
    underlying: str,
    as_of_date: date,
    prices_df: pd.DataFrame,
    risk_free_rate: float = 0.05,
    iv_series: Optional[pd.Series] = None,
    expirations_dte: List[int] = [21, 30, 45, 60, 90],
    strike_range_pct: float = 0.20,
    n_strikes_each_side: int = 10,
) -> List[Dict]:
    """
    Reconstruct a full options chain for `underlying` as of `as_of_date`.
    
    Returns a list of option quote dicts (one per strike/expiry/right).
    All quotes have is_reconstructed=True.
    """
    as_of_ts = pd.Timestamp(as_of_date)

    if as_of_ts not in prices_df.index and as_of_ts not in prices_df.index:
        # Find most recent bar
        prior = prices_df.index[prices_df.index <= as_of_ts]
        if len(prior) == 0:
            return []
        as_of_ts = prior[-1]

    S = float(prices_df.loc[as_of_ts, "adj_close"] if "adj_close" in prices_df.columns
              else prices_df.loc[as_of_ts, "close"])

    if iv_series is not None and as_of_ts in iv_series.index:
        iv = float(iv_series.loc[as_of_ts])
    else:
        # Fallback: compute from trailing 30 bars
        idx = prices_df.index.get_loc(as_of_ts)
        if idx < 30:
            iv = 0.20
        else:
            slice_ = prices_df.iloc[max(0, idx-30):idx+1]
            closes = slice_["adj_close"] if "adj_close" in slice_.columns else slice_["close"]
            returns = np.log(closes / closes.shift(1)).dropna()
            iv = float(returns.std() * np.sqrt(252) * 1.15)
            iv = max(0.05, min(iv, 3.0))

    quotes = []

    for dte in expirations_dte:
        exp_date = as_of_date + timedelta(days=dte)
        T = dte / 365.0

        # Generate strikes around current price
        strike_step = S * strike_range_pct / n_strikes_each_side
        strikes = [
            round(S * (1 + (i - n_strikes_each_side) * strike_range_pct / n_strikes_each_side), 2)
            for i in range(2 * n_strikes_each_side + 1)
        ]

        for K in strikes:
            for flag, right in [("c", "C"), ("p", "P")]:
                greeks = compute_greeks(S, K, T, risk_free_rate, iv, flag)
                price = greeks["price"]

                if price < 0.01:
                    continue

                # Estimate bid-ask spread: wider for further OTM, wider for lower OI
                moneyness = abs(np.log(S / K))
                spread_pct = max(0.05, min(iv * 0.10 + moneyness * 0.5, 0.50))
                spread = price * spread_pct
                bid = max(0.01, price - spread / 2)
                ask = price + spread / 2

                # Estimate synthetic OI and volume from ATM proxy
                delta_abs = abs(greeks["delta"])
                synthetic_oi = max(10, int(1000 * delta_abs * (1 - moneyness)))
                synthetic_vol = max(1, int(100 * delta_abs))

                quotes.append({
                    "underlying_id": underlying,
                    "timestamp": as_of_date,
                    "expiration": exp_date,
                    "strike": K,
                    "right": right,
                    "bid": round(bid, 2),
                    "ask": round(ask, 2),
                    "last": round(price, 2),
                    "volume": synthetic_vol,
                    "open_interest": synthetic_oi,
                    "iv": iv,
                    "delta": greeks["delta"],
                    "gamma": greeks["gamma"],
                    "theta": greeks["theta"],
                    "vega": greeks["vega"],
                    "rho": greeks["rho"],
                    "is_reconstructed": True,
                })

    return quotes


def find_option_by_delta(
    chain: List[Dict],
    target_delta: float,
    right: str,
    expiration: Optional[date] = None,
) -> Optional[Dict]:
    """
    Find the option in the chain closest to target_delta.
    For puts, target_delta should be negative (e.g., -0.30).
    """
    candidates = [q for q in chain if q["right"] == right]
    if expiration:
        candidates = [q for q in candidates if q["expiration"] == expiration]

    if not candidates:
        return None

    return min(candidates, key=lambda q: abs(q["delta"] - target_delta))


def get_expected_move(atm_call: Dict, atm_put: Dict) -> float:
    """
    Expected move = ATM call mid + ATM put mid.
    Represents the market's implied 1 std deviation move to expiration.
    """
    call_mid = (atm_call["bid"] + atm_call["ask"]) / 2
    put_mid = (atm_put["bid"] + atm_put["ask"]) / 2
    return call_mid + put_mid


def compute_iv_rank(
    iv_history: pd.Series,
    current_iv: float,
    window: int = 252,
) -> Tuple[float, float]:
    """
    Returns (IV_Rank, IV_Percentile) for current_iv given iv_history.
    
    IV Rank  = (current - 52wk_low) / (52wk_high - 52wk_low)
    IV Pctile = fraction of days in window where IV was below current
    
    The distinction matters: a stock at its 52wk low IV has IVR=0 but
    may still have high IV Percentile if most days were even lower.
    """
    recent = iv_history.iloc[-window:] if len(iv_history) >= window else iv_history

    low_52 = recent.min()
    high_52 = recent.max()

    if high_52 == low_52:
        return 0.5, 0.5

    ivr = (current_iv - low_52) / (high_52 - low_52)
    iv_pct = float((recent < current_iv).mean())

    return float(np.clip(ivr, 0, 1)), float(iv_pct)
