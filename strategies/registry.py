"""
Strategy Registry and Ensemble Coordinator

Problems this solves:
1. Strategies are scattered across modules with no central catalog
2. Multi-strategy ensembles have no formal signal aggregation
3. No way to check if two strategies are trading the same asset
   and coordinate sizing accordingly
4. No mechanism to discover and instantiate strategies by name

This module provides:
- StrategyRegistry: catalog of all available strategies
- SignalAggregator: combines signals from multiple strategies
- StrategyCorrelationMonitor: detects when strategies are generating
  correlated signals (so they don't double-size a position)
- EnsembleEngine: runs multiple strategies and aggregates their signals

Signal aggregation methods (per spec):
- majority_vote: 3 of 5 LONG -> LONG
- confidence_weighted: net_signal = sum(conf_i * dir_i) / sum(conf_i)
- orthogonality_check: if agreeing strategies are same type, count as one vote
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Type, Callable, Tuple, Any
import numpy as np
import pandas as pd

from core.events import SignalEvent, Direction, BarEvent
from core.event_queue import EventQueue

logger = logging.getLogger(__name__)


# ── Strategy Registry ─────────────────────────────────────────────────────────

class StrategyRegistry:
    """
    Central catalog of all available strategies.
    
    Allows discovery by name, by type, and by asset class.
    Strategies register themselves or are registered by the factory.
    """

    _registry: Dict[str, Dict] = {}

    @classmethod
    def register(
        cls,
        name: str,
        strategy_class,
        description: str = "",
        strategy_type: str = "",
        asset_classes: List[str] = None,
        default_params: Dict = None,
    ) -> None:
        cls._registry[name] = {
            "class": strategy_class,
            "description": description,
            "type": strategy_type,
            "asset_classes": asset_classes or ["equity"],
            "default_params": default_params or {},
        }

    @classmethod
    def get(cls, name: str) -> Optional[Dict]:
        return cls._registry.get(name)

    @classmethod
    def list_all(cls) -> pd.DataFrame:
        """Return a DataFrame summary of all registered strategies."""
        rows = []
        for name, info in cls._registry.items():
            rows.append({
                "name": name,
                "type": info["type"],
                "asset_classes": ", ".join(info["asset_classes"]),
                "description": info["description"],
            })
        return pd.DataFrame(rows)

    @classmethod
    def build(
        cls,
        name: str,
        asset_ids: List[str],
        event_queue: EventQueue,
        **kwargs,
    ):
        """Instantiate a registered strategy by name."""
        entry = cls._registry.get(name)
        if not entry:
            raise ValueError(
                f"Strategy '{name}' not found. "
                f"Available: {list(cls._registry.keys())}"
            )
        params = {**entry["default_params"], **kwargs}
        # Special handling for strategies that don't use asset_ids
        if name == 'pairs':
            asset_a = params.pop('asset_a', asset_ids[0] if len(asset_ids) > 0 else 'SPY')
            asset_b = params.pop('asset_b', asset_ids[1] if len(asset_ids) > 1 else 'QQQ')
            return entry["class"](
                asset_a=asset_a,
                asset_b=asset_b,
                event_queue=event_queue,
                **params,
            )
        return entry["class"](
            asset_ids=asset_ids,
            event_queue=event_queue,
            **params,
        )


def _register_all_strategies():
    """Register all built-in strategies."""
    from strategies.trend import (
        SMAcrossover, EMACrossover, MACDStrategy,
        DonchianBreakout, ADXFilteredTrend, TimeSeriesMomentum,
    )
    from strategies.mean_reversion import (
        BollingerBandMeanReversion, RSIMeanReversion, PairsTradingStrategy,
    )
    from strategies.momentum_factor import (
        CrossSectionalMomentum, DualMomentum, LowVolatilityFactor, BuyAndHold,
    )
    from strategies.options_strategies import (
        CoveredCallStrategy, IronCondorStrategy, LongStraddleStrategy,
    )

    regs = [
        ("buy_and_hold", BuyAndHold, "Buy and hold baseline (walking skeleton validation)", "passive", ["equity", "etf"]),
        ("sma", SMAcrossover, "SMA crossover (50/200 default)", "trend", ["equity", "etf"], {"fast": 50, "slow": 200}),
        ("ema", EMACrossover, "EMA crossover (12/26 default)", "trend", ["equity", "etf"], {"fast_span": 12, "slow_span": 26}),
        ("macd", MACDStrategy, "MACD signal line crossover", "trend", ["equity", "etf"]),
        ("donchian", DonchianBreakout, "Donchian channel breakout (Turtle)", "trend", ["equity", "etf", "commodity_etf"], {"entry_period": 20}),
        ("adx", ADXFilteredTrend, "ADX-filtered trend (go flat when ranging)", "trend", ["equity", "etf"]),
        ("tsmom", TimeSeriesMomentum, "Time-series momentum (Moskowitz et al.)", "trend", ["equity", "etf"], {"lookback_months": 12}),
        ("bollinger", BollingerBandMeanReversion, "Bollinger Band mean reversion", "mean_reversion", ["equity"]),
        ("rsi", RSIMeanReversion, "RSI oversold/overbought with trend filter", "mean_reversion", ["equity"]),
        ("xs_momentum", CrossSectionalMomentum, "Cross-sectional momentum (Jegadeesh-Titman)", "momentum", ["equity"]),
        # DualMomentum has a non-standard constructor; registered but built manually
        # ("dual_momentum", DualMomentum, "Dual momentum (Antonacci)", "momentum", ["equity", "etf"]),
    ]

    # Register DualMomentum separately with a factory wrapper
    class _DualMomWrapper:
        """Wraps DualMomentum to accept standard (asset_ids, event_queue) signature."""
        def __new__(cls, asset_ids, event_queue, **kwargs):
            return DualMomentum(
                event_queue=event_queue,
                equity_assets=[a for a in asset_ids if a not in ("AGG", "BIL")],
                **{k: v for k, v in kwargs.items() if k in ("bond_asset", "tbill_asset")},
            )

    StrategyRegistry.register(
        name="dual_momentum",
        strategy_class=_DualMomWrapper,
        description="Dual momentum (Antonacci)",
        strategy_type="momentum",
        asset_classes=["equity", "etf"],
        default_params={},
    )

    regs += [
        ('pairs', PairsTradingStrategy, 'Cointegration pairs trading (Kalman)', 'mean_reversion', ['equity']),
        ("low_vol", LowVolatilityFactor, "Low volatility factor (Baker et al.)", "factor", ["equity"]),
        ("covered_call", CoveredCallStrategy, "Covered call income strategy", "options", ["equity", "etf"], {}),
        ("iron_condor", IronCondorStrategy, "Iron condor theta decay", "options", ["equity", "etf"], {}),
        ("long_straddle", LongStraddleStrategy, "Long straddle for high vol events", "options", ["equity", "etf"], {}),
    ]

    for entry in regs:
        name = entry[0]
        cls_ = entry[1]
        desc = entry[2]
        stype = entry[3]
        asset_classes = entry[4]
        default_params = entry[5] if len(entry) > 5 else {}
        StrategyRegistry.register(
            name=name,
            strategy_class=cls_,
            description=desc,
            strategy_type=stype,
            asset_classes=asset_classes,
            default_params=default_params,
        )


def _register_ml_strategies():
    """Register ML strategies (optional - only if lightgbm available)."""
    try:
        from strategies.ml_strategy import MLCrossSectionalStrategy, TurbulenceFilteredStrategy

        class _MLWrapper:
            def __new__(cls, asset_ids, event_queue, **kwargs):
                return MLCrossSectionalStrategy(
                    asset_ids=asset_ids, event_queue=event_queue,
                    **{k: v for k, v in kwargs.items()
                       if k in ('train_years','retrain_months','top_n','bottom_n','min_ic','long_only')}
                )

        StrategyRegistry.register(
            name='ml_xsectional',
            strategy_class=_MLWrapper,
            description='ML cross-sectional alpha (LightGBM, QLib-inspired)',
            strategy_type='ml',
            asset_classes=['equity'],
            default_params={'train_years': 3, 'top_n': 3, 'long_only': True},
        )
    except ImportError:
        pass  # lightgbm not installed

# Auto-register on import
try:
    _register_all_strategies()
    _register_ml_strategies()
except Exception as e:
    logger.warning(f"Strategy auto-registration partial failure: {e}")


# ── Signal Aggregator ──────────────────────────────────────────────────────────

@dataclass
class AggregatedSignal:
    """Result of combining signals from multiple strategies."""
    asset_id: str
    direction: Direction
    confidence: float
    contributing_strategies: List[str]
    vote_breakdown: Dict[str, Direction]
    aggregation_method: str
    timestamp: datetime
    net_score: float = 0.0


class SignalAggregator:
    """
    Combines SignalEvents from multiple strategies into a single
    actionable direction with confidence.
    
    Three methods:
    
    1. majority_vote: plurality wins; confidence = vote margin / total votes
    2. confidence_weighted: weighted average of numeric directions
    3. orthogonality_checked: same as confidence_weighted, but strategies
       of the same type on the same asset count as a single vote
    
    The orthogonality check is critical: if you have 3 trend-following
    strategies all going long SPY, that's one directional bet counted three
    times, not three independent confirmations.
    """

    def __init__(
        self,
        method: str = "confidence_weighted",
        direction_threshold: float = 0.25,
    ):
        if method not in ("majority_vote", "confidence_weighted", "orthogonality_checked"):
            raise ValueError(f"Unknown aggregation method: {method}")
        self.method = method
        self.threshold = direction_threshold

        # Buffer of pending signals per asset
        self._signal_buffer: Dict[str, List[SignalEvent]] = {}

    def add_signal(self, signal: SignalEvent) -> None:
        """Add a signal to the buffer for its asset."""
        asset_id = signal.asset_id
        if asset_id not in self._signal_buffer:
            self._signal_buffer[asset_id] = []
        # Replace any existing signal from the same strategy
        self._signal_buffer[asset_id] = [
            s for s in self._signal_buffer[asset_id]
            if s.strategy_id != signal.strategy_id
        ]
        self._signal_buffer[asset_id].append(signal)

    def aggregate(self, asset_id: str) -> Optional[AggregatedSignal]:
        """Aggregate all buffered signals for an asset."""
        signals = self._signal_buffer.get(asset_id, [])
        if not signals:
            return None

        if self.method == "majority_vote":
            return self._majority_vote(asset_id, signals)
        elif self.method == "confidence_weighted":
            return self._confidence_weighted(asset_id, signals)
        elif self.method == "orthogonality_checked":
            return self._orthogonality_checked(asset_id, signals)

        return None

    def aggregate_all(self) -> List[AggregatedSignal]:
        """Aggregate signals for all assets in buffer."""
        results = []
        for asset_id in list(self._signal_buffer.keys()):
            agg = self.aggregate(asset_id)
            if agg is not None:
                results.append(agg)
        return results

    def clear(self, asset_id: Optional[str] = None) -> None:
        """Clear the signal buffer."""
        if asset_id:
            self._signal_buffer.pop(asset_id, None)
        else:
            self._signal_buffer.clear()

    def _numeric_direction(self, d: Direction) -> float:
        return {Direction.LONG: 1.0, Direction.SHORT: -1.0, Direction.FLAT: 0.0}.get(d, 0.0)

    def _direction_from_score(self, score: float) -> Direction:
        if score > self.threshold:
            return Direction.LONG
        elif score < -self.threshold:
            return Direction.SHORT
        return Direction.FLAT

    def _majority_vote(self, asset_id: str, signals: List[SignalEvent]) -> AggregatedSignal:
        vote_counts = {Direction.LONG: 0, Direction.SHORT: 0, Direction.FLAT: 0}
        for s in signals:
            vote_counts[s.direction] += 1

        total = len(signals)
        winner = max(vote_counts, key=vote_counts.get)
        confidence = vote_counts[winner] / total

        return AggregatedSignal(
            asset_id=asset_id,
            direction=winner,
            confidence=confidence,
            contributing_strategies=[s.strategy_id for s in signals],
            vote_breakdown={s.strategy_id: s.direction for s in signals},
            aggregation_method="majority_vote",
            timestamp=max(s.timestamp for s in signals),
            net_score=vote_counts[Direction.LONG] / total - vote_counts[Direction.SHORT] / total,
        )

    def _confidence_weighted(self, asset_id: str, signals: List[SignalEvent]) -> AggregatedSignal:
        total_conf = sum(s.confidence for s in signals)
        if total_conf == 0:
            total_conf = len(signals)

        net_score = sum(
            s.confidence * self._numeric_direction(s.direction) for s in signals
        ) / total_conf

        direction = self._direction_from_score(net_score)
        confidence = min(1.0, abs(net_score))

        return AggregatedSignal(
            asset_id=asset_id,
            direction=direction,
            confidence=confidence,
            contributing_strategies=[s.strategy_id for s in signals],
            vote_breakdown={s.strategy_id: s.direction for s in signals},
            aggregation_method="confidence_weighted",
            timestamp=max(s.timestamp for s in signals),
            net_score=net_score,
        )

    def _orthogonality_checked(self, asset_id: str, signals: List[SignalEvent]) -> AggregatedSignal:
        """
        Confidence-weighted aggregation, but strategies of the same signal_type
        pointing in the same direction are collapsed to a single vote.
        This prevents over-counting correlated signals.
        """
        # Group by (signal_type, direction)
        type_direction_groups: Dict[Tuple, List[SignalEvent]] = {}
        for s in signals:
            key = (s.signal_type, s.direction)
            if key not in type_direction_groups:
                type_direction_groups[key] = []
            type_direction_groups[key].append(s)

        # Collapse each group to its mean confidence
        collapsed_signals = []
        for (stype, direction), group in type_direction_groups.items():
            mean_conf = sum(sg.confidence for sg in group) / len(group)
            # Use the first signal as representative, override confidence
            rep = SignalEvent(
                timestamp=max(sg.timestamp for sg in group),
                strategy_id=f"collapsed_{stype}",
                asset_id=asset_id,
                direction=direction,
                confidence=mean_conf,
                signal_type=stype,
            )
            collapsed_signals.append(rep)

        # Now run confidence-weighted on the collapsed signals
        result = self._confidence_weighted(asset_id, collapsed_signals)
        result.aggregation_method = "orthogonality_checked"
        result.contributing_strategies = [s.strategy_id for s in signals]
        result.vote_breakdown = {s.strategy_id: s.direction for s in signals}
        return result


# ── Strategy Correlation Monitor ──────────────────────────────────────────────

class StrategyCorrelationMonitor:
    """
    Monitors pairwise correlation of strategy equity curves.
    
    If two strategies have correlation > threshold, they are treated as
    one strategy for position sizing. This prevents inadvertent doubling
    of exposure when two strategies respond to the same signal.
    
    Used by the PortfolioManager's CapitalRequest arbitration.
    """

    def __init__(self, correlation_threshold: float = 0.70, window: int = 60):
        self.threshold = correlation_threshold
        self.window = window
        self._equity_histories: Dict[str, List[float]] = {}

    def update(self, strategy_id: str, equity: float) -> None:
        if strategy_id not in self._equity_histories:
            self._equity_histories[strategy_id] = []
        self._equity_histories[strategy_id].append(equity)
        # Keep only rolling window
        if len(self._equity_histories[strategy_id]) > self.window * 2:
            self._equity_histories[strategy_id] = \
                self._equity_histories[strategy_id][-self.window * 2:]

    def get_correlated_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Returns list of (strategy_a, strategy_b, correlation)
        for pairs exceeding the threshold.
        """
        strategies = list(self._equity_histories.keys())
        correlated = []

        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                a, b = strategies[i], strategies[j]
                hist_a = self._equity_histories[a][-self.window:]
                hist_b = self._equity_histories[b][-self.window:]

                if len(hist_a) < 10 or len(hist_b) < 10:
                    continue

                n = min(len(hist_a), len(hist_b))
                arr_a = np.array(hist_a[-n:])
                arr_b = np.array(hist_b[-n:])

                # Correlation on returns, not levels
                ret_a = np.diff(arr_a) / (arr_a[:-1] + 1e-10)
                ret_b = np.diff(arr_b) / (arr_b[:-1] + 1e-10)

                if len(ret_a) < 5:
                    continue

                corr = float(np.corrcoef(ret_a, ret_b)[0, 1])
                if abs(corr) >= self.threshold:
                    correlated.append((a, b, corr))

        return correlated

    def are_correlated(self, strategy_a: str, strategy_b: str) -> bool:
        pairs = self.get_correlated_pairs()
        for a, b, _ in pairs:
            if (a == strategy_a and b == strategy_b) or \
               (a == strategy_b and b == strategy_a):
                return True
        return False


# ── Ensemble Engine ───────────────────────────────────────────────────────────

class EnsembleEngine:
    """
    Runs multiple strategies in parallel and routes aggregated signals
    to the portfolio manager.
    
    This is the top-level coordinator for multi-strategy systems.
    It replaces the simple "all strategies process the bar" loop
    in BacktestEngine with a proper signal aggregation pipeline.
    
    Pipeline:
    1. All strategies process BarEvent -> emit SignalEvents to a local buffer
    2. SignalAggregator combines signals per asset
    3. Aggregated signals are emitted to the main EventQueue
    4. PortfolioManager processes them as normal
    
    The EnsembleEngine is transparent to the rest of the system:
    the PortfolioManager and ExecutionHandler don't know they're
    receiving aggregated signals.
    """

    def __init__(
        self,
        strategies,
        main_queue: EventQueue,
        aggregation_method: str = "confidence_weighted",
        min_votes: int = 1,
    ):
        self.strategies = strategies
        self.main_queue = main_queue
        self.min_votes = min_votes

        # Internal queue captures raw strategy signals
        self._internal_queue = EventQueue()

        # Wire strategies to internal queue
        for strategy in strategies:
            strategy._queue = self._internal_queue

        self.aggregator = SignalAggregator(method=aggregation_method)
        self.corr_monitor = StrategyCorrelationMonitor()

        # Map strategy_id -> strategy for metadata lookup
        self._strategy_map = {s.strategy_id: s for s in strategies}

    def on_bar(self, event: BarEvent, data_handler) -> None:
        """Process a bar through all strategies, then aggregate signals."""
        # All strategies process the bar into internal queue
        for strategy in self.strategies:
            if event.asset_id in strategy.asset_ids:
                try:
                    strategy.on_bar(event, data_handler)
                except Exception as e:
                    logger.error(
                        f"Strategy {strategy.strategy_id} error "
                        f"on {event.asset_id}: {e}"
                    )

        # Drain internal queue and add signals to aggregator
        pending_assets = set()
        while not self._internal_queue.empty():
            raw_event = self._internal_queue.get()
            from core.events import EventType
            if raw_event.event_type == EventType.SIGNAL:
                self.aggregator.add_signal(raw_event)
                pending_assets.add(raw_event.asset_id)

        # Aggregate and forward to main queue
        for asset_id in pending_assets:
            agg = self.aggregator.aggregate(asset_id)
            if agg is None:
                continue

            # Check minimum vote requirement
            n_signals = len(agg.contributing_strategies)
            if n_signals < self.min_votes:
                continue

            # Emit aggregated signal to main queue
            aggregated_event = SignalEvent(
                timestamp=agg.timestamp,
                strategy_id=f"ensemble_{agg.aggregation_method}",
                asset_id=agg.asset_id,
                direction=agg.direction,
                confidence=agg.confidence,
                signal_type="ensemble",
                metadata={
                    "contributors": agg.contributing_strategies,
                    "net_score": agg.net_score,
                    "votes": {k: v.value for k, v in agg.vote_breakdown.items()},
                },
            )
            self.main_queue.put(aggregated_event)

        # Update correlation monitor
        # (would use portfolio equity per strategy in production)
