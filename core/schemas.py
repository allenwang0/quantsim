"""
Canonical data schemas. RawBar is always stored unadjusted.
AdjustedBar is computed at query time. Never store adjusted prices.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Literal


@dataclass
class RawBar:
    asset_id: str
    timestamp: datetime       # UTC midnight for daily bars
    open: float               # unadjusted
    high: float               # unadjusted
    low: float                # unadjusted
    close: float              # unadjusted
    volume: int
    exchange: str = ""
    source: str = ""          # 'yfinance', 'stooq', etc.

    def validate(self) -> bool:
        """Detect common data quality failures."""
        if self.high < self.low:
            return False
        if self.close < self.low or self.close > self.high:
            return False
        if self.open <= 0 or self.close <= 0:
            return False
        if self.volume < 0:
            return False
        return True


@dataclass
class AdjustedBar:
    """Computed view. Never stored in the database."""
    asset_id: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float              # split-adjusted + dividend-adjusted (total return)
    volume: int
    raw_close: float          # original unadjusted close
    split_factor: float = 1.0
    div_adjustment: float = 0.0


@dataclass
class AdjustmentFactor:
    asset_id: str
    effective_date: date
    cumulative_split_factor: float    # multiply raw price by this → split-adjusted
    cumulative_div_adjustment: float  # add to split-adjusted → total return price


@dataclass
class OptionsQuote:
    underlying_id: str
    timestamp: datetime
    expiration: date
    strike: float
    right: Literal["C", "P"]
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    iv: float          # implied volatility as decimal (0.25 = 25%)
    delta: float
    gamma: float
    theta: float       # per calendar day
    vega: float
    rho: float
    is_reconstructed: bool = True  # True if computed from BSM reconstruction

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_pct(self) -> float:
        if self.mid == 0:
            return 1.0
        return (self.ask - self.bid) / self.mid

    @property
    def is_liquid(self) -> bool:
        return self.open_interest > 1000 and self.volume > 100 and self.spread_pct < 0.15

    @property
    def option_symbol(self) -> str:
        """Standard OCC option symbol format."""
        exp_str = self.expiration.strftime("%y%m%d")
        strike_int = int(self.strike * 1000)
        return f"{self.underlying_id}{exp_str}{self.right}{strike_int:08d}"


@dataclass
class MacroSeries:
    series_id: str
    release_timestamp: date   # date the value was PUBLISHED (use for point-in-time)
    reference_period: date    # period the value refers to
    value: float
    is_revised: bool = False


@dataclass
class FundamentalData:
    asset_id: str
    as_of_date: date          # SEC EDGAR filing date (point-in-time anchor)
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    roe: Optional[float] = None
    gross_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    earnings_per_share: Optional[float] = None
