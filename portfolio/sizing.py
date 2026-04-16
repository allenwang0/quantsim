"""
Position sizing models. Each model returns the target dollar allocation
given signal confidence and portfolio state.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from portfolio.portfolio import Portfolio


class PositionSizer:
    """Base class for all sizing models."""

    def size(
        self,
        portfolio: Portfolio,
        asset_id: str,
        confidence: float = 1.0,
        price: float = 0.0,
        vol_estimate: Optional[float] = None,
    ) -> int:
        """Returns number of shares/contracts to trade."""
        raise NotImplementedError


class FixedFractionalSizer(PositionSizer):
    """
    Risk a fixed fraction of equity on each trade.
    target_position_value = equity * fraction * confidence
    """

    def __init__(self, fraction: float = 0.02, max_position_fraction: float = 0.10):
        self.fraction = fraction
        self.max_fraction = max_position_fraction

    def size(self, portfolio: Portfolio, asset_id: str,
             confidence: float = 1.0, price: float = 0.0,
             vol_estimate: Optional[float] = None) -> int:
        if price <= 0:
            return 0
        equity = portfolio.total_equity
        target_value = equity * self.fraction * confidence
        target_value = min(target_value, equity * self.max_fraction)
        return max(1, int(target_value / price))


class VolatilityTargetSizer(PositionSizer):
    """
    Position sized so that each position contributes equal risk (target_vol_contribution).
    position_size = (target_vol_contribution * equity) / (asset_vol * price)
    
    This is the AQR volatility targeting approach.
    """

    def __init__(
        self,
        target_vol_per_position: float = 0.01,  # 1% annualized vol contribution per position
        max_position_fraction: float = 0.15,
    ):
        self.target_vol = target_vol_per_position
        self.max_fraction = max_position_fraction

    def size(self, portfolio: Portfolio, asset_id: str,
             confidence: float = 1.0, price: float = 0.0,
             vol_estimate: Optional[float] = None) -> int:
        if price <= 0 or not vol_estimate or vol_estimate <= 0:
            return 0

        equity = portfolio.total_equity
        # Position in dollar terms: target_vol / asset_vol * equity
        dollar_risk = (self.target_vol / vol_estimate) * equity * confidence
        dollar_risk = min(dollar_risk, equity * self.max_fraction)
        return max(1, int(dollar_risk / price))


class KellySizer(PositionSizer):
    """
    Kelly criterion: f* = (bp - q) / b where b = win/loss ratio, p = win rate.
    Uses fractional Kelly (half-Kelly) for practical risk management.
    Requires win_rate and avg_win_loss_ratio parameters.
    """

    def __init__(
        self,
        win_rate: float = 0.55,
        win_loss_ratio: float = 1.5,
        kelly_fraction: float = 0.5,  # half-Kelly
        max_position_fraction: float = 0.20,
    ):
        self.win_rate = win_rate
        self.win_loss_ratio = win_loss_ratio
        self.kelly_fraction = kelly_fraction
        self.max_fraction = max_position_fraction

        # Precompute Kelly fraction
        p = win_rate
        q = 1 - p
        b = win_loss_ratio
        full_kelly = (b * p - q) / b
        self._f = max(0, full_kelly * kelly_fraction)

    def size(self, portfolio: Portfolio, asset_id: str,
             confidence: float = 1.0, price: float = 0.0,
             vol_estimate: Optional[float] = None) -> int:
        if price <= 0:
            return 0
        equity = portfolio.total_equity
        target_value = equity * self._f * confidence
        target_value = min(target_value, equity * self.max_fraction)
        return max(1, int(target_value / price))


class EqualWeightSizer(PositionSizer):
    """
    Equal weight across N expected positions.
    Simple and robust; correct starting point for most strategies.
    """

    def __init__(self, n_positions: int = 20, max_position_fraction: float = 0.10):
        self.n_positions = n_positions
        self.max_fraction = max_position_fraction

    def size(self, portfolio: Portfolio, asset_id: str,
             confidence: float = 1.0, price: float = 0.0,
             vol_estimate: Optional[float] = None) -> int:
        if price <= 0:
            return 0
        equity = portfolio.total_equity
        target_value = equity / self.n_positions * confidence
        target_value = min(target_value, equity * self.max_fraction)
        return max(1, int(target_value / price))


def compute_realized_vol(prices: pd.Series, window: int = 30) -> float:
    """
    Annualized realized volatility from a price series.
    Used by VolatilityTargetSizer and regime detection.
    """
    if len(prices) < window + 1:
        return 0.20  # default fallback

    log_returns = np.log(prices / prices.shift(1)).dropna()
    rv = log_returns.iloc[-window:].std() * np.sqrt(252)
    return float(rv)


class GARCHVolatilityTargetSizer(PositionSizer):
    """
    VolatilityTargetSizer powered by GARCH(1,1) instead of rolling realized vol.

    GARCH forecasts are more responsive to volatility regime changes than
    simple rolling standard deviation, making position sizing faster to adapt.

    Integrated here so the PortfolioManager can pass it in as a drop-in
    replacement for VolatilityTargetSizer.
    """

    def __init__(
        self,
        target_annual_vol: float = 0.10,
        max_position_fraction: float = 0.15,
        fit_window: int = 252,
    ):
        self.target_vol = target_annual_vol
        self.max_fraction = max_position_fraction
        try:
            from strategies.garch_vol import GARCHVolatilityAdapter
            self._adapter = GARCHVolatilityAdapter(
                fit_window=fit_window,
                vol_target=target_annual_vol,
            )
            self._garch_available = True
        except Exception:
            self._garch_available = False
            self._adapter = None

    def update(self, asset_id: str, return_value: float) -> None:
        """Call on each bar to feed the GARCH model."""
        if self._adapter:
            self._adapter.add_bar(asset_id, return_value)

    def size(
        self,
        portfolio,
        asset_id: str,
        confidence: float = 1.0,
        price: float = 0.0,
        vol_estimate: float = None,
    ) -> int:
        if price <= 0:
            return 0

        equity = portfolio.total_equity

        # Use GARCH forecast if available, else fall back to passed vol_estimate
        if self._garch_available and self._adapter:
            garch_vol = self._adapter.get_vol_forecast(asset_id)
            vol = garch_vol if garch_vol > 0 else (vol_estimate or 0.20)
        else:
            vol = vol_estimate or 0.20

        # Vol-target sizing: target_vol / asset_vol * equity
        dollar_risk = (self.target_vol / max(vol, 0.01)) * equity * confidence
        dollar_risk = min(dollar_risk, equity * self.max_fraction)
        return max(1, int(dollar_risk / price))
