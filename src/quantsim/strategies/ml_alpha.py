"""
Alpha features and the turbulence index.

compute_alpha_features builds price and volume factors per asset;
compute_turbulence_index and turbulence_regime_filter flag market stress
(Mahalanobis distance of returns, as in FinRL). The LightGBM ranking model
that used to live here was removed in 3.0.0; see CHANGELOG.md.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Alpha Factor Library (QLib Alpha158-inspired) ─────────────────────────────


def compute_alpha_features(
    prices: pd.DataFrame,
    volumes: Optional[pd.DataFrame] = None,
    vix: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute ~80 alpha factors for a universe of assets.

    Factor families (inspired by QLib Alpha158/Alpha360):
    - Price momentum: multi-period returns, normalized
    - Price reversion: short-term reversion signals
    - Volatility: realized vol at multiple horizons
    - Volume/turnover: volume patterns and anomalies
    - Technical: RSI, Bollinger, MACD, ADX
    - Macro: VIX regime, yield curve

    All factors are cross-sectionally ranked (rank normalization)
    to remove distributional differences across assets.

    Returns: MultiIndex DataFrame (date x asset) x factors
    """
    features = {}
    close = prices

    # ── Momentum factors (multi-period) ──────────────────────────────────────
    for period in [5, 10, 20, 60, 120, 252]:
        # Return over period
        ret = close.pct_change(period)
        features[f"ret_{period}d"] = ret

        # Normalized by volatility (risk-adjusted momentum)
        vol = close.pct_change().rolling(period).std() * np.sqrt(252)
        features[f"mom_adj_{period}d"] = ret / (vol + 1e-8)

    # 12-1 month momentum (Jegadeesh-Titman)
    features["mom_12_1"] = close.shift(21) / close.shift(252) - 1

    # ── Reversion factors ────────────────────────────────────────────────────
    for period in [1, 3, 5]:
        features[f"rev_{period}d"] = -close.pct_change(period)  # short-term reversion

    # Mean reversion z-score
    for window in [10, 20]:
        roll_mean = close.rolling(window).mean()
        roll_std = close.rolling(window).std()
        features[f"zscore_{window}d"] = (close - roll_mean) / (roll_std + 1e-8)

    # ── Volatility factors ───────────────────────────────────────────────────
    log_ret = np.log(close / close.shift(1))
    for period in [5, 10, 20, 60]:
        features[f"vol_{period}d"] = log_ret.rolling(period).std() * np.sqrt(252)

    # Volatility ratio (short vs long term)
    features["vol_ratio_5_20"] = features["vol_5d"] / (features["vol_20d"] + 1e-8)
    features["vol_ratio_20_60"] = features["vol_20d"] / (features["vol_60d"] + 1e-8)

    # ── Volume factors (if available) ────────────────────────────────────────
    if volumes is not None:
        vol_roll_20 = volumes.rolling(20).mean()
        features["vol_ratio"] = volumes / (vol_roll_20 + 1)
        features["amihud_illiq"] = (log_ret.abs() / (volumes * close + 1)).rolling(20).mean()

        # Dollar volume
        dollar_vol = volumes * close
        features["dollar_vol_20"] = dollar_vol.rolling(20).mean()

    # ── Technical indicators ─────────────────────────────────────────────────
    # RSI (14-day)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    features["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    # Bollinger Band position
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    features["bb_pos"] = (close - sma_20) / (2 * std_20 + 1e-8)

    # MACD signal
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    features["macd_hist"] = (macd - signal) / (close + 1e-8)

    # Price relative to 52-week high/low
    high_52w = close.rolling(252).max()
    low_52w = close.rolling(252).min()
    features["pct_from_high"] = (close - high_52w) / (high_52w + 1e-8)
    features["pct_from_low"] = (close - low_52w) / (low_52w + 1e-8)

    # ── Macro regime features ────────────────────────────────────────────────
    if vix is not None:
        vix_aligned = vix.reindex(close.index, method="ffill")
        for col in close.columns:
            features["vix_level"] = pd.DataFrame({col: vix_aligned for col in close.columns})
        features["vix_20ma"] = features["vix_level"].rolling(20).mean()

    # Combine into MultiIndex
    feature_df = pd.concat(features, axis=1)

    # Cross-sectional rank normalization: rank each factor each day
    # This removes the need for factor-specific scaling
    ranked = feature_df.rank(axis=1, pct=True) - 0.5

    return ranked.astype(np.float32)


# ── Turbulence Index (FinRL-inspired risk indicator) ─────────────────────────


def compute_turbulence_index(returns: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """
    Mahalanobis distance-based turbulence index.
    Used in FinRL as a market stress indicator.

    High turbulence = extreme, unusual market behavior = reduce exposure.

    turbulence_t = (y_t - mu)' * Sigma^-1 * (y_t - mu)

    where y_t = cross-sectional return vector on day t,
    mu = historical mean, Sigma = historical covariance.
    """
    turbulence = pd.Series(index=returns.index, dtype=float)

    for i in range(lookback, len(returns)):
        current = returns.iloc[i].values
        hist = returns.iloc[i - lookback : i]
        mu = hist.mean().values
        cov = hist.cov().values

        try:
            cov_inv = np.linalg.pinv(cov)
            diff = current - mu
            turb = float(diff @ cov_inv @ diff.T)
        except Exception:
            turb = 0.0

        turbulence.iloc[i] = turb

    return turbulence.fillna(0)


def turbulence_regime_filter(
    turbulence: pd.Series,
    threshold_pct: float = 0.90,
) -> pd.Series:
    """
    Returns a binary regime series: 1 = normal, 0 = high turbulence.
    Threshold at 90th percentile of historical turbulence.
    """
    threshold = turbulence.quantile(threshold_pct)
    return (turbulence <= threshold).astype(int)
