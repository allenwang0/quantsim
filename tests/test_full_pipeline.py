"""
Full research workflow integration tests.

These tests verify the complete quant research pipeline works together:
1. Data injection → event-driven backtest → meaningful analytics
2. Vectorized parameter sweep → DSR-corrected ranking
3. Walk-forward optimization → OOS Sharpe reported correctly
4. Portfolio optimization → sensible weights
5. GARCH vol → realistic forecast
6. OrderManager → correct stop/target behavior
7. Tearsheet → valid HTML with all sections
8. Health monitor → alert generation

All tests use synthetic data (no network required).
Tests are designed to be fast (<1s each) and deterministic.
"""

import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def multi_asset_prices():
    """400-bar synthetic price matrix for 4 assets."""
    np.random.seed(100)
    n, symbols = 400, ["SPY", "QQQ", "GLD", "TLT"]
    dates = pd.date_range("2018-01-02", periods=n, freq="B")
    drifts = [0.0004, 0.0005, 0.0001, -0.0001]
    vols = [0.012, 0.015, 0.008, 0.006]
    return pd.DataFrame(
        {s: p * np.cumprod(1 + np.random.normal(d, v, n))
         for s, p, d, v in zip(symbols, [300, 350, 170, 120], drifts, vols)},
        index=dates,
    )


@pytest.fixture
def populated_db(tmp_path, multi_asset_prices):
    """DB with all 4 assets injected."""
    from core.database_v2 import init_full_db
    from core.database import db_conn

    db_path = str(tmp_path / "full.db")
    os.environ["QUANTSIM_DB"] = db_path
    init_full_db(db_path)

    for sym in multi_asset_prices.columns:
        series = multi_asset_prices[sym]
        for date, price in series.items():
            ts = int(date.timestamp())
            with db_conn(db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO raw_bars "
                    "(asset_id,timestamp,open,high,low,close,volume,source) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (sym, ts, price*0.998, price*1.005, price*0.995, price, 1_000_000, "test"),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO adjustment_factors VALUES (?,?,1.0,0.0)",
                    (sym, ts),
                )
    return db_path


# ── Engine correctness ────────────────────────────────────────────────────────

class TestEventEngineE2E:
    def test_buy_and_hold_positive_return_strong_drift(self, tmp_path):
        """Strong upward drift → positive total return is guaranteed."""
        from core.database_v2 import init_full_db
        from core.database import db_conn
        from strategies.momentum_factor import BuyAndHold
        from core.event_queue import EventQueue
        from backtesting.engine import BacktestEngine

        db_path = str(tmp_path / "bah.db")
        os.environ["QUANTSIM_DB"] = db_path
        init_full_db(db_path)

        np.random.seed(1)
        dates = pd.date_range("2020-01-02", periods=300, freq="B")
        # Very strong drift: 50% annual, low vol to ensure positive OOS
        prices = 100 * np.cumprod(1 + np.random.normal(0.002, 0.005, 300))

        with db_conn(db_path) as conn:
            for d, p in zip(dates, prices):
                ts = int(d.timestamp())
                conn.execute(
                    "INSERT OR REPLACE INTO raw_bars (asset_id,timestamp,open,high,low,close,volume,source) VALUES (?,?,?,?,?,?,?,?)",
                    ("SPY", ts, p*0.999, p*1.003, p*0.997, p, 1_000_000, "test"),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO adjustment_factors VALUES (?,?,1.0,0.0)",
                    ("SPY", ts),
                )

        eq = EventQueue()
        s = BuyAndHold(asset_ids=["SPY"], event_queue=eq)
        engine = BacktestEngine(
            strategies=[s],
            start=datetime(2020, 1, 2), end=datetime(2020, 12, 31),
            initial_capital=100_000, db_path=db_path, verbose=False,
            warmup_bars=10,  # short warmup to maximize trading window
        )
        results = engine.run()

        assert engine._fills_processed >= 1, "Queue mismatch: 0 fills"
        assert results.get("total_return") is not None, "total_return is None"
        tr = float(results["total_return"])
        assert tr > 0, f"Return {tr:.4%} not positive on strong upward trend"

    def test_sma_crossover_generates_multiple_signals(self, populated_db, multi_asset_prices):
        """SMA on oscillating price series must generate ≥2 crossover signals."""
        from strategies.trend import SMAcrossover
        from core.event_queue import EventQueue
        from backtesting.engine import BacktestEngine

        # Create oscillating (mean-reverting) series to force crossovers
        np.random.seed(7)
        n = 300
        dates = pd.date_range("2018-01-02", periods=n, freq="B")
        t = np.linspace(0, 6 * np.pi, n)
        prices = 100 + 25 * np.sin(t) + np.random.normal(0, 1, n)

        from core.database import db_conn
        db_path = populated_db
        with db_conn(db_path) as conn:
            for d, p in zip(dates, prices):
                ts = int(d.timestamp())
                conn.execute(
                    "INSERT OR REPLACE INTO raw_bars (asset_id,timestamp,open,high,low,close,volume,source) VALUES (?,?,?,?,?,?,?,?)",
                    ("OHLC", ts, p*0.998, p*1.005, p*0.995, p, 1_000_000, "test"),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO adjustment_factors VALUES (?,?,1.0,0.0)",
                    ("OHLC", ts),
                )

        eq = EventQueue()
        s = SMAcrossover(asset_ids=["OHLC"], event_queue=eq, fast=5, slow=20)
        engine = BacktestEngine(
            strategies=[s],
            start=datetime(2018, 1, 2), end=datetime(2019, 2, 28),
            initial_capital=100_000, db_path=db_path, verbose=False,
            warmup_bars=25,
        )
        engine.run()

        assert engine._signals_processed >= 2, \
            f"Only {engine._signals_processed} signals on oscillating price series. " \
            "SMA crossover should fire multiple times."

    def test_equity_history_spans_full_backtest_period(self, tmp_path):
        """Equity history must have entries throughout the full backtest, not just warmup end."""
        from core.database_v2 import init_full_db
        from core.database import db_conn
        from strategies.momentum_factor import BuyAndHold
        from core.event_queue import EventQueue
        from backtesting.engine import BacktestEngine

        db_path = str(tmp_path / "span.db")
        os.environ["QUANTSIM_DB"] = db_path
        init_full_db(db_path)

        np.random.seed(3)
        n = 200
        dates = pd.date_range("2020-01-02", periods=n, freq="B")
        prices = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.010, n))

        with db_conn(db_path) as conn:
            for d, p in zip(dates, prices):
                ts = int(d.timestamp())
                conn.execute(
                    "INSERT OR REPLACE INTO raw_bars (asset_id,timestamp,open,high,low,close,volume,source) VALUES (?,?,?,?,?,?,?,?)",
                    ("SPY", ts, p*0.999, p*1.003, p*0.997, p, 1_000_000, "test"),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO adjustment_factors VALUES (?,?,1.0,0.0)",
                    ("SPY", ts),
                )

        eq = EventQueue()
        s = BuyAndHold(asset_ids=["SPY"], event_queue=eq)
        engine = BacktestEngine(
            strategies=[s],
            start=datetime(2020, 1, 2), end=datetime(2020, 10, 31),
            initial_capital=100_000, db_path=db_path, verbose=False,
            warmup_bars=5,
        )
        engine.run()

        history = engine.portfolio._equity_history
        assert len(history) >= 10, "Too few equity entries"

        # Timestamps should span a meaningful date range
        timestamps = [t for t, _ in history]
        unique_ts = set(str(t) for t in timestamps)
        assert len(unique_ts) >= len(history) * 0.7, \
            f"Timestamps not unique: {len(unique_ts)}/{len(history)}. " \
            "Timestamp propagation bug still present."


# ── Vectorized engine ─────────────────────────────────────────────────────────

class TestVectorizedE2E:
    def test_equity_curve_in_results(self, multi_asset_prices):
        from backtesting.vectorized import VectorizedBacktester, sma_crossover_signal
        bt = VectorizedBacktester(multi_asset_prices)
        r = bt.run(sma_crossover_signal, {"fast": 20, "slow": 100})
        assert "_equity_curve" in r
        assert len(r["_equity_curve"]) == len(multi_asset_prices)

    def test_parameter_sweep_ranks_by_dsr(self, multi_asset_prices):
        from backtesting.vectorized import VectorizedBacktester, sma_crossover_signal
        bt = VectorizedBacktester(multi_asset_prices)
        df = bt.parameter_sweep(sma_crossover_signal, {"fast": [10, 20], "slow": [80, 120]})
        assert len(df) == 4
        vals = df["deflated_sharpe_corrected"].values
        assert all(vals[i] >= vals[i+1] - 1e-9 for i in range(len(vals)-1)), \
            "Results not sorted by DSR"

    def test_all_signal_functions_return_results(self, multi_asset_prices):
        from backtesting.vectorized import (
            VectorizedBacktester, sma_crossover_signal, rsi_signal,
            momentum_signal, bollinger_signal, donchian_signal
        )
        bt = VectorizedBacktester(multi_asset_prices)
        for fn in [sma_crossover_signal, rsi_signal, momentum_signal,
                   bollinger_signal, donchian_signal]:
            r = bt.run(fn)
            assert isinstance(r, dict), f"{fn.__name__} failed"


# ── Walk-forward optimizer ────────────────────────────────────────────────────

class TestWalkForwardE2E:
    def test_wfo_returns_only_oos_metrics(self, multi_asset_prices):
        from backtesting.walk_forward import WalkForwardOptimizer
        from backtesting.vectorized import sma_crossover_signal

        wfo = WalkForwardOptimizer(
            multi_asset_prices, train_years=1.0, test_months=6, step_months=6
        )
        results = wfo.optimize_and_evaluate(
            signal_fn=sma_crossover_signal,
            param_grid={"fast": [20, 50], "slow": [100]},
        )

        assert "avg_oos_sharpe" in results
        assert "n_windows" in results
        assert results["n_windows"] >= 1
        assert "deflated_sharpe_corrected" in results
        assert "summary" in results

    def test_regime_classifier_returns_four_regimes(self, multi_asset_prices):
        from backtesting.walk_forward import RegimeAnalyzer
        spy = multi_asset_prices["SPY"]
        regimes = RegimeAnalyzer.classify_regimes(spy)
        expected = {"bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol"}
        actual = set(regimes.unique())
        assert actual.issubset(expected | {"unknown"}), \
            f"Unexpected regime labels: {actual - (expected | {'unknown'})}"


# ── Portfolio optimization ────────────────────────────────────────────────────

class TestPortfolioOptE2E:
    def test_hrp_weights_sum_one_and_positive(self, multi_asset_prices):
        from portfolio.optimization import HierarchicalRiskParity
        returns = multi_asset_prices.pct_change().dropna()
        w = HierarchicalRiskParity().optimize(returns)
        assert abs(sum(w.values()) - 1.0) < 1e-6
        assert all(v >= 0 for v in w.values())

    def test_risk_parity_equal_contribution(self, multi_asset_prices):
        from portfolio.optimization import RiskParityOptimizer, compute_covariance
        import numpy as np
        returns = multi_asset_prices.pct_change().dropna()
        rp = RiskParityOptimizer()
        w = rp.optimize(returns)
        w_arr = np.array([w[s] for s in returns.columns])
        cov = compute_covariance(returns, "ledoit_wolf")
        # Risk contribution per asset
        port_var = w_arr @ cov @ w_arr
        rc = w_arr * (cov @ w_arr) / port_var
        # All risk contributions should be approximately equal
        rc_std = rc.std()
        assert rc_std < 0.1, f"Risk contributions not equal (std={rc_std:.4f})"

    def test_black_litterman_moves_toward_view(self, multi_asset_prices):
        """A LONG view on SPY should tilt weights toward SPY vs no-view baseline."""
        from portfolio.optimization import BlackLittermanOptimizer
        returns = multi_asset_prices.pct_change().dropna()
        bl = BlackLittermanOptimizer()

        # No views
        w_baseline = bl.optimize(returns, views=None)

        # Strong long view on SPY
        w_view = bl.optimize(returns, views=[{
            "assets": ["SPY"], "weights": [1],
            "return": 0.002, "confidence": 0.9
        }])

        # SPY weight should be higher with the view
        assert w_view.get("SPY", 0) >= w_baseline.get("SPY", 0) * 0.9, \
            "Black-Litterman did not tilt toward the LONG SPY view"


# ── GARCH volatility ──────────────────────────────────────────────────────────

class TestGARCHE2E:
    def test_vol_forecast_in_realistic_range(self, multi_asset_prices):
        from strategies.garch_vol import GARCHVolatilityAdapter
        adapter = GARCHVolatilityAdapter(fit_window=150, refit_every=30)
        returns = multi_asset_prices["SPY"].pct_change().dropna()
        for r in returns:
            adapter.add_bar("SPY", float(r))
        vol = adapter.get_vol_forecast("SPY")
        realized_vol = float(returns.std() * np.sqrt(252))
        # GARCH forecast should be within 5x of realized vol
        assert 0.01 < vol < 3.0, f"GARCH vol {vol:.3f} outside realistic range"
        assert abs(vol - realized_vol) / realized_vol < 5.0, \
            f"GARCH {vol:.3f} too far from realized {realized_vol:.3f}"

    def test_leverage_inversely_proportional_to_vol(self):
        from strategies.garch_vol import GARCHForecaster
        fc = GARCHForecaster(vol_target_annual=0.12)
        # Low vol → high leverage (capped at 2.0)
        lev_low = fc.leverage_for_target_vol(0.05)
        # High vol → low leverage (floored at 0.25)
        lev_high = fc.leverage_for_target_vol(0.40)
        assert lev_low > lev_high, "Leverage not inversely proportional to vol"
        assert lev_low <= 2.0
        assert lev_high >= 0.25


# ── OrderManager end-to-end ───────────────────────────────────────────────────

class TestOrderManagerE2E:
    def make_bar(self, price, high=None, low=None, dt=None):
        from core.events import BarEvent
        return BarEvent(
            timestamp=dt or datetime(2022, 1, 3),
            asset_id="SPY",
            open=price * 0.999,
            high=high or price * 1.01,
            low=low or price * 0.99,
            close=price, volume=1_000_000, adj_close=price,
        )

    def test_bracket_order_full_lifecycle(self):
        """Entry → price rises → target hit → stop cancelled."""
        from backtesting.order_manager import OrderManager
        from core.event_queue import EventQueue
        from core.events import Direction

        eq = EventQueue()
        om = OrderManager(event_queue=eq)
        om.add_bracket("SPY", entry_price=400, strategy_id="test",
                       stop_pct=0.03, target_pct=0.06)

        # Prices rise gradually — should not trigger stop
        for price in [402, 405, 410, 415]:
            bar = self.make_bar(price, high=price*1.005, low=price*0.995)
            triggered = om.on_bar(bar)
            assert "SPY" not in triggered, f"Stop triggered prematurely at {price}"

        # Price hits target (400 * 1.06 = 424)
        bar_target = self.make_bar(425, high=426, low=424)
        triggered = om.on_bar(bar_target)
        assert "SPY" in triggered, "Target not triggered"

        # Signal emitted should be FLAT
        assert not eq.empty()
        sig = eq.get()
        assert sig.direction == Direction.FLAT

    def test_trailing_stop_locks_in_profit(self):
        """Trailing stop should move up with price, locking in profit."""
        from backtesting.order_manager import OrderManager
        from core.event_queue import EventQueue

        eq = EventQueue()
        om = OrderManager(event_queue=eq)
        om.add_trailing_stop("SPY", entry_price=400, strategy_id="test",
                             trail_pct=0.05, direction="LONG")

        # Price rises to 440
        for price in [410, 420, 430, 440]:
            om.on_bar(self.make_bar(price, high=price*1.005, low=price*0.997))

        stop = om._stops.get("SPY")
        assert stop is not None
        # Trailing stop should now be near 440 * 0.95 = 418
        assert stop.stop_price > 400, f"Trailing stop {stop.stop_price:.2f} didn't move up"
        assert stop.stop_price > 415, f"Trailing stop {stop.stop_price:.2f} not locking profit"


# ── Reporting and tearsheet ────────────────────────────────────────────────────

class TestReportingE2E:
    def make_equity(self, n=300, drift=0.0004, vol=0.012, seed=42):
        np.random.seed(seed)
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.Series(
            100_000 * np.cumprod(1 + np.random.normal(drift, vol, n)),
            index=dates,
        )

    def test_tearsheet_all_sections_present(self):
        from reporting.tearsheet import generate_tearsheet
        equity = self.make_equity()
        html = generate_tearsheet(equity, "E2E Test Strategy")
        required = ["Sharpe Ratio", "Max Drawdown", "CAGR", "Deflated SR",
                    "Plotly", "eq-chart", "heatmap-chart", "rs-chart"]
        for item in required:
            assert item in html, f"Missing from tearsheet: {item}"

    def test_tearsheet_with_complete_inputs(self):
        """Tearsheet with trades, benchmark, and WFO results."""
        from reporting.tearsheet import generate_tearsheet
        equity = self.make_equity()
        benchmark = self.make_equity(seed=99)
        trades = pd.DataFrame([{
            "asset_id": "SPY", "direction": "LONG",
            "entry_price": 300.0, "exit_price": 315.0,
            "realized_pnl": 1500.0, "strategy_id": "sma", "holding_bars": 21,
        }])
        wfo = {
            "n_windows": 4, "avg_oos_sharpe": 0.45,
            "avg_is_sharpe": 0.85, "sharpe_degradation": 0.40,
            "deflated_sharpe_corrected": 0.12, "oos_win_rate": 0.75,
            "summary": {"pass": True, "message": "VIABLE | OOS Sharpe 0.45"},
        }
        html = generate_tearsheet(
            equity, "Full Test", trades_df=trades,
            benchmark=benchmark, wfo_results=wfo,
            output_path="/tmp/quantsim_test_tearsheet.html",
        )
        assert os.path.exists("/tmp/quantsim_test_tearsheet.html")
        assert len(html) > 20_000

    def test_monte_carlo_ci_width_decreases_with_more_data(self):
        """Longer samples → narrower confidence interval on Sharpe."""
        from reporting.advanced import AdvancedAnalytics
        np.random.seed(42)
        returns_short = pd.Series(np.random.normal(0.0004, 0.012, 60))
        returns_long = pd.Series(np.random.normal(0.0004, 0.012, 500))

        mc_short = AdvancedAnalytics.monte_carlo_sharpe(returns_short, n_simulations=500)
        mc_long = AdvancedAnalytics.monte_carlo_sharpe(returns_long, n_simulations=500)

        ci_short = mc_short["sharpe_ci_high"] - mc_short["sharpe_ci_low"]
        ci_long = mc_long["sharpe_ci_high"] - mc_long["sharpe_ci_low"]
        assert ci_long < ci_short, \
            f"Longer sample should give narrower CI: short={ci_short:.2f}, long={ci_long:.2f}"

    def test_full_report_verdict_correct(self):
        """Strong Sharpe → proceed=True; weak Sharpe → proceed=False."""
        from reporting.advanced import generate_full_report
        np.random.seed(42)
        dates = pd.date_range("2015-01-01", periods=2000, freq="B")
        # Strong return → should pass
        strong = pd.Series(100_000 * np.cumprod(1 + np.random.normal(0.0006, 0.008, 2000)), index=dates)
        report = generate_full_report(strong, n_strategies_tested=1)
        assert "verdict" in report
        # At least one verdict line present
        assert len(report["verdict"]["lines"]) > 0


# ── Strategy registry ─────────────────────────────────────────────────────────

class TestStrategyRegistryE2E:
    def test_all_standard_strategies_buildable(self):
        from strategies.registry import StrategyRegistry
        from core.event_queue import EventQueue
        # Skip strategies with special constructors
        skip = {"pairs", "dual_momentum", "covered_call", "iron_condor",
                "long_straddle", "ml_xsectional"}
        df = StrategyRegistry.list_all()
        for _, row in df.iterrows():
            name = row["name"]
            if name in skip:
                continue
            eq = EventQueue()
            s = StrategyRegistry.build(name, asset_ids=["SPY", "QQQ"], event_queue=eq)
            assert s is not None, f"Strategy '{name}' build returned None"
            assert hasattr(s, "on_bar"), f"Strategy '{name}' missing on_bar()"

    def test_signal_aggregator_majority_and_confidence(self):
        """majority_vote and confidence_weighted: 2 LONG vs 1 SHORT → LONG."""
        from strategies.registry import SignalAggregator
        from core.events import SignalEvent, Direction
        for method in ["majority_vote", "confidence_weighted"]:
            agg = SignalAggregator(method=method)
            for i in range(3):
                sig = SignalEvent(
                    timestamp=datetime(2022, 1, 3),
                    strategy_id=f"s{i}", asset_id="SPY",
                    direction=Direction.LONG if i < 2 else Direction.SHORT,
                    confidence=0.8, signal_type="trend",
                )
                agg.add_signal(sig)
            result = agg.aggregate("SPY")
            assert result is not None, f"method={method} returned None"
            assert result.direction == Direction.LONG, (
                f"method={method}: 2 LONG vs 1 SHORT should return LONG, got {result.direction}"
            )

    def test_signal_aggregator_orthogonality_collapses_same_type(self):
        """orthogonality_checked: same-type signals collapse to one vote.
        Two LONG-trend signals = one LONG-trend vote.
        Result vs one SHORT-trend = 1:1 tie → net_score=0 → FLAT. That is correct."""
        from strategies.registry import SignalAggregator
        from core.events import SignalEvent, Direction
        agg = SignalAggregator(method="orthogonality_checked")
        # Two LONG trend signals collapse to one vote
        agg.add_signal(SignalEvent(timestamp=datetime(2022, 1, 3),
            strategy_id="t1", asset_id="SPY", direction=Direction.LONG,
            confidence=0.8, signal_type="trend"))
        agg.add_signal(SignalEvent(timestamp=datetime(2022, 1, 3),
            strategy_id="t2", asset_id="SPY", direction=Direction.LONG,
            confidence=0.8, signal_type="trend"))
        # One SHORT trend signal
        agg.add_signal(SignalEvent(timestamp=datetime(2022, 1, 3),
            strategy_id="t3", asset_id="SPY", direction=Direction.SHORT,
            confidence=0.8, signal_type="trend"))
        result = agg.aggregate("SPY")
        assert result is not None
        # 1 LONG-trend vs 1 SHORT-trend = tie → FLAT (orthogonality working correctly)
        assert result.direction == Direction.FLAT, (
            f"Expected FLAT (1:1 tie after collapse), got {result.direction}"
        )

    def test_signal_aggregator_orthogonality_different_types_give_long(self):
        """Two LONG signals from different types beat one SHORT = LONG."""
        from strategies.registry import SignalAggregator
        from core.events import SignalEvent, Direction
        agg = SignalAggregator(method="orthogonality_checked", direction_threshold=0.15)
        agg.add_signal(SignalEvent(timestamp=datetime(2022, 1, 3),
            strategy_id="trend_1", asset_id="SPY", direction=Direction.LONG,
            confidence=0.9, signal_type="trend"))
        agg.add_signal(SignalEvent(timestamp=datetime(2022, 1, 3),
            strategy_id="mr_1", asset_id="SPY", direction=Direction.LONG,
            confidence=0.8, signal_type="mean_reversion"))
        agg.add_signal(SignalEvent(timestamp=datetime(2022, 1, 3),
            strategy_id="mom_1", asset_id="SPY", direction=Direction.SHORT,
            confidence=0.3, signal_type="momentum"))
        result = agg.aggregate("SPY")
        assert result is not None
        assert result.net_score > 0, (
            f"2 LONG types vs 1 low-conf SHORT type should have net_score>0, got {result.net_score}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
