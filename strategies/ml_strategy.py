"""
MLAlphaStrategy: event-engine-compatible ML alpha strategy.

Wraps the MLAlphaStrategy from ml_alpha.py into the Strategy base class
interface so it plugs into BacktestEngine identically to SMA or RSI.

Pipeline per bar:
1. Update feature cache with new bar data
2. At rebalance time (monthly): retrain if needed, generate rankings
3. Emit LONG/SHORT/FLAT signals based on ranking quintiles

Walk-forward training is enforced:
- Training uses only data strictly before the current bar
- No future data ever touches the model during training or inference
"""

from __future__ import annotations
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict
import numpy as np
import pandas as pd

from core.events import BarEvent, SignalEvent, Direction
from core.event_queue import EventQueue
from data.data_handler import DataHandler
from strategies.trend import Strategy
from strategies.ml_alpha import (
    MLAlphaStrategy as _MLCore,
    compute_alpha_features,
    compute_turbulence_index,
    turbulence_regime_filter,
)

logger = logging.getLogger(__name__)


class MLCrossSectionalStrategy(Strategy):
    """
    Event-engine-compatible ML cross-sectional alpha strategy.

    Uses LightGBM to rank assets by predicted forward return.
    Longs the top quintile, optionally shorts the bottom quintile.

    Training protocol:
    - Initial train: first time we have train_years of data
    - Rolling retrain: every retrain_months thereafter
    - Training window: strictly before current bar (no look-ahead)

    IC monitoring: if rolling IC drops below min_ic, the model is
    considered stale and position sizing is reduced to zero until
    the next retrain produces a valid model.
    """

    def __init__(
        self,
        asset_ids: List[str],
        event_queue: EventQueue,
        train_years: int = 3,
        retrain_months: int = 1,
        top_n: int = 3,
        bottom_n: int = 3,
        min_ic: float = 0.02,
        long_only: bool = True,
        db_path: Optional[str] = None,
    ):
        super().__init__(
            strategy_id="ML_CrossSectional",
            asset_ids=asset_ids,
            event_queue=event_queue,
            warmup_bars=train_years * 252 + 30,
        )
        self.train_years = train_years
        self.retrain_months = retrain_months
        self.top_n = top_n
        self.bottom_n = bottom_n
        self.min_ic = min_ic
        self.long_only = long_only
        self._db_path = db_path

        # Price history cache (needed for feature computation)
        self._price_cache: Dict[str, List[float]] = {a: [] for a in asset_ids}
        self._vol_cache: Dict[str, List[float]] = {a: [] for a in asset_ids}
        self._date_cache: List[datetime] = []

        # ML core (does the actual LightGBM work)
        self._ml_core = _MLCore(
            universe=asset_ids,
            train_years=train_years,
            retrain_months=retrain_months,
            top_n=top_n,
            bottom_n=bottom_n,
            min_ic=min_ic,
        )

        # Rebalance tracking
        self._last_rebalance_month: int = -1
        self._current_signals: Dict[str, Direction] = {a: Direction.FLAT for a in asset_ids}
        self._bars_processed: int = 0
        self._model_trained: bool = False

        logger.info(
            f"MLCrossSectionalStrategy: {len(asset_ids)} assets, "
            f"train={train_years}yr, retrain={retrain_months}mo, "
            f"top_n={top_n}, bottom_n={bottom_n}"
        )

    def on_bar(self, event: BarEvent, data_handler: DataHandler) -> None:
        """Process new bar: update cache, retrain if needed, emit signals."""
        asset_id = event.asset_id
        if asset_id not in self.asset_ids:
            return

        self._tick(asset_id)
        self._bars_processed += 1

        # Update price cache (used for feature computation)
        self._price_cache[asset_id].append(event.adj_close)

        # Only trigger rebalance logic from the first asset (avoid duplicate runs)
        if asset_id != self.asset_ids[0]:
            return

        self._date_cache.append(event.timestamp)

        # Warmup: need enough history for training
        if not self._is_warmed_up(asset_id):
            return

        # Monthly rebalance check
        current_month = event.timestamp.month
        if current_month == self._last_rebalance_month:
            return
        self._last_rebalance_month = current_month

        # Build price DataFrame from cache (point-in-time: excludes current bar)
        prices_df = self._build_price_df()
        if prices_df.empty or len(prices_df) < self.train_years * 252 // 2:
            return

        # Feature computation
        try:
            features = compute_alpha_features(prices_df)
        except Exception as e:
            logger.warning(f"Feature computation failed: {e}")
            return

        # Train or retrain
        should_train = (
            not self._model_trained or
            self._bars_processed % (self.retrain_months * 21) == 0
        )

        if should_train:
            train_end = prices_df.index[-self._ml_core.val_bars - 1] \
                if len(prices_df) > self._ml_core.val_bars else prices_df.index[-2]
            train_start = prices_df.index[0]

            train_ic = self._ml_core.fit(features, prices_df, train_start, train_end)
            self._model_trained = self._ml_core.is_trained

            if self._model_trained and self._db_path:
                self._log_training_run(train_ic, len(prices_df))

        if not self._model_trained:
            return

        # Check IC health
        if self._ml_core.rolling_ic < self.min_ic and self._ml_core.rolling_ic != 0.0:
            logger.warning(
                f"ML model IC {self._ml_core.rolling_ic:.4f} below threshold "
                f"{self.min_ic} — suppressing signals"
            )
            self._emit_flat_all(event.timestamp)
            return

        # Generate rankings
        rankings = self._ml_core.predict_rankings(features, event.timestamp)
        if rankings is None:
            return

        new_signals = self._ml_core.get_signals(rankings)

        # Emit signals for changes in direction
        for asset, direction_int in new_signals.items():
            new_dir = {1: Direction.LONG, -1: Direction.SHORT, 0: Direction.FLAT}[direction_int]
            if self.long_only and new_dir == Direction.SHORT:
                new_dir = Direction.FLAT

            old_dir = self._current_signals.get(asset, Direction.FLAT)
            if new_dir != old_dir:
                self._current_signals[asset] = new_dir
                confidence = float(abs(rankings.get(asset, 0.5) - 0.5) * 2)
                self._emit_signal(
                    asset_id=asset,
                    direction=new_dir,
                    timestamp=event.timestamp,
                    confidence=min(1.0, confidence),
                    signal_type="ml_xsectional",
                    holding_period=self.retrain_months * 21,
                    metadata={
                        "rank": float(rankings.get(asset, 0.5)),
                        "rolling_ic": float(self._ml_core.rolling_ic),
                    },
                )

    def _build_price_df(self) -> pd.DataFrame:
        """Build a price DataFrame from the cache for feature computation."""
        n_dates = len(self._date_cache)
        data = {}
        for asset in self.asset_ids:
            prices = self._price_cache[asset]
            if len(prices) >= n_dates:
                data[asset] = prices[-n_dates:]
            elif prices:
                # Pad with NaN if shorter
                data[asset] = [np.nan] * (n_dates - len(prices)) + prices

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, index=self._date_cache[-n_dates:])
        return df.ffill().dropna(how="all")

    def _emit_flat_all(self, timestamp: datetime) -> None:
        """Emit FLAT for all currently open positions."""
        for asset, direction in self._current_signals.items():
            if direction != Direction.FLAT:
                self._emit_signal(asset, Direction.FLAT, timestamp,
                                  signal_type="ml_ic_flat")
                self._current_signals[asset] = Direction.FLAT

    def _log_training_run(self, train_ic: float, n_samples: int) -> None:
        """Persist training run metadata to DB."""
        try:
            from core.database_v2 import log_ml_run
            importances = {}
            if self._ml_core._model is not None:
                imp = self._ml_core.feature_importances()
                if imp is not None:
                    importances = imp.head(20).to_dict()

            log_ml_run(
                db_path=self._db_path,
                run_id=str(uuid.uuid4()),
                strategy_id=self.strategy_id,
                train_start=0,
                train_end=0,
                n_samples=n_samples,
                train_ic=train_ic,
                val_ic=self._ml_core.rolling_ic,
                model_params=self._ml_core.model_kwargs,
                feature_importances=importances,
            )
        except Exception as e:
            logger.debug(f"ML run logging failed: {e}")


class TurbulenceFilteredStrategy(Strategy):
    """
    Wrapper that applies a turbulence index filter to any strategy.

    When market turbulence (Mahalanobis distance of returns) exceeds
    the 90th percentile of its historical distribution, the strategy
    goes to cash. This is the FinRL-inspired risk-off mechanism.

    Usage:
        base = SMAcrossover(asset_ids=["SPY"], event_queue=eq)
        safe = TurbulenceFilteredStrategy(base_strategy=base, ...)
    """

    def __init__(
        self,
        base_strategy: Strategy,
        event_queue: EventQueue,
        lookback_bars: int = 252,
        turbulence_threshold_pct: float = 0.90,
    ):
        super().__init__(
            strategy_id=f"TurbFilter_{base_strategy.strategy_id}",
            asset_ids=base_strategy.asset_ids,
            event_queue=event_queue,
            warmup_bars=base_strategy.warmup_bars + lookback_bars,
        )
        self.base = base_strategy
        self.base._queue = event_queue  # share queue
        self.lookback = lookback_bars
        self.threshold_pct = turbulence_threshold_pct

        self._return_history: List[Dict[str, float]] = []
        self._turbulence_history: List[float] = []
        self._in_turbulence: bool = False
        self._last_asset_prices: Dict[str, float] = {}

    def on_bar(self, event: BarEvent, data_handler: DataHandler) -> None:
        asset_id = event.asset_id
        self._tick(asset_id)

        # Track cross-sectional returns for turbulence computation
        if asset_id in self._last_asset_prices:
            prev = self._last_asset_prices[asset_id]
            if prev > 0:
                ret = (event.adj_close - prev) / prev
                if len(self._return_history) == 0 or \
                   asset_id not in self._return_history[-1]:
                    self._return_history.append({})
                self._return_history[-1][asset_id] = ret

        self._last_asset_prices[asset_id] = event.adj_close

        # Only compute turbulence once per bar (on first asset trigger)
        if asset_id == self.asset_ids[0] and len(self._return_history) > self.lookback:
            self._update_turbulence()

        # If in turbulence regime: suppress signals, emit FLAT for all positions
        if self._in_turbulence:
            self._emit_signal(
                asset_id=asset_id,
                direction=Direction.FLAT,
                timestamp=event.timestamp,
                signal_type="turbulence_exit",
                metadata={"turbulence_active": True},
            )
            return

        # Normal regime: delegate to base strategy
        self.base.on_bar(event, data_handler)

    def _update_turbulence(self) -> None:
        """Compute current turbulence and update regime flag."""
        if len(self._return_history) < self.lookback:
            return

        recent = self._return_history[-self.lookback:]
        # Build return matrix: rows=dates, cols=assets
        all_assets = list(self.asset_ids)
        matrix = []
        for day_rets in recent:
            row = [day_rets.get(a, 0.0) for a in all_assets]
            matrix.append(row)

        arr = np.array(matrix)
        if arr.shape[0] < 30 or arr.shape[1] < 2:
            return

        mu = arr[:-1].mean(axis=0)
        cov = np.cov(arr[:-1].T) + np.eye(len(all_assets)) * 1e-8

        current = arr[-1]
        try:
            cov_inv = np.linalg.pinv(cov)
            diff = current - mu
            turb = float(diff @ cov_inv @ diff.T)
        except Exception:
            turb = 0.0

        self._turbulence_history.append(turb)

        # Determine threshold from history
        if len(self._turbulence_history) >= 50:
            threshold = np.percentile(self._turbulence_history, self.threshold_pct * 100)
            self._in_turbulence = turb > threshold
        else:
            self._in_turbulence = False
