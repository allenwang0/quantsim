"""Advanced analytics, the strategy health monitor and the tearsheet."""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from tests.synthetic_bars import make_price_df


class TestAdvancedAnalytics:
    def test_monte_carlo_sharpe(self):
        from quantsim.reporting.advanced import AdvancedAnalytics

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0003, 0.015, 500))
        result = AdvancedAnalytics.monte_carlo_sharpe(returns, n_simulations=1000)
        assert "sharpe_point_estimate" in result
        assert "sharpe_ci_low" in result
        assert "sharpe_ci_high" in result
        assert result["sharpe_ci_low"] < result["sharpe_ci_high"]

    def test_turnover_analysis(self):
        from quantsim.reporting.advanced import AdvancedAnalytics

        prices = make_price_df(n=300)
        signals = pd.DataFrame(
            np.random.choice([-1, 0, 1], size=prices.shape),
            columns=prices.columns,
            index=prices.index,
        )
        result = AdvancedAnalytics.turnover_analysis(signals, prices)
        assert "annual_turnover_rate" in result
        assert "estimated_annual_cost_pct" in result

    def test_insufficient_sample_warning(self):
        from quantsim.reporting.advanced import AdvancedAnalytics

        short_returns = pd.Series(np.random.normal(0, 0.01, 10))
        result = AdvancedAnalytics.monte_carlo_sharpe(short_returns, n_simulations=100)
        assert "error" in result


class TestStrategyHealthMonitor:
    def test_monitor_detects_drawdown_warning(self, tmp_path):
        from quantsim.portfolio.portfolio import Portfolio
        from quantsim.reporting.monitor import StrategyHealthMonitor

        db_path = str(tmp_path / "monitor_test.db")
        from quantsim.core.database_v2 import init_full_db

        init_full_db(db_path)

        monitor = StrategyHealthMonitor(
            db_path=db_path,
            max_dd_warn=-0.05,
            max_dd_critical=-0.10,
        )

        # Create portfolio with heavy drawdown
        p = Portfolio(initial_capital=100_000, cash=100_000)
        p.peak_equity = 100_000
        p._equity_history = [(datetime(2022, 1, i), 100_000 - i * 2000) for i in range(1, 10)]
        # Last equity is 82,000 → drawdown ≈ -18%

        ts = datetime(2022, 1, 10)
        # Manually simulate deep drawdown
        p.cash = 82_000
        alerts = monitor.check_all(p, ts)

        # Should have at least a warning alert
        assert any(a["severity"] in ("WARNING", "CRITICAL") for a in alerts)

    def test_monitor_no_false_alerts_on_healthy_portfolio(self, tmp_path):
        from quantsim.portfolio.portfolio import Portfolio
        from quantsim.reporting.monitor import StrategyHealthMonitor

        db_path = str(tmp_path / "monitor_healthy.db")
        from quantsim.core.database_v2 import init_full_db

        init_full_db(db_path)

        monitor = StrategyHealthMonitor(db_path=db_path)
        p = Portfolio(initial_capital=100_000, cash=100_000)
        p.peak_equity = 100_000
        p.cash = 105_000  # profit

        alerts = monitor.check_all(p, datetime(2022, 6, 1))
        # No drawdown or concentration alerts on a healthy portfolio
        dd_alerts = [a for a in alerts if "DRAWDOWN" in a.get("risk_type", "")]
        assert len(dd_alerts) == 0

    def test_monitor_alert_deduplication(self, tmp_path):
        from quantsim.portfolio.portfolio import Portfolio
        from quantsim.reporting.monitor import StrategyHealthMonitor

        db_path = str(tmp_path / "monitor_dedup.db")
        from quantsim.core.database_v2 import init_full_db

        init_full_db(db_path)

        monitor = StrategyHealthMonitor(db_path=db_path, max_dd_warn=-0.05)
        p = Portfolio(initial_capital=100_000, cash=80_000)
        p.peak_equity = 100_000

        ts = datetime(2022, 1, 1)
        monitor.check_all(p, ts)
        # Same timestamp day: should be deduplicated
        alerts2 = monitor.check_all(p, ts + timedelta(minutes=5))
        # Second call in same cooldown window should produce 0 additional alerts
        assert len(alerts2) == 0


class TestTearsheet:
    def make_equity(self, n=300, seed=42):
        np.random.seed(seed)
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        equity = pd.Series(
            100_000 * np.cumprod(1 + np.random.normal(0.0004, 0.015, n)),
            index=dates,
        )
        return equity

    def test_tearsheet_generates_html(self):
        from quantsim.reporting.tearsheet import generate_tearsheet

        equity = self.make_equity()
        html = generate_tearsheet(equity, strategy_name="Test SMA")
        assert isinstance(html, str)
        assert len(html) > 5000
        assert "<!DOCTYPE html>" in html
        assert "QuantSim" in html
        assert "Plotly" in html

    def test_tearsheet_includes_metrics(self):
        from quantsim.reporting.tearsheet import generate_tearsheet

        equity = self.make_equity()
        html = generate_tearsheet(equity, strategy_name="Test Strategy")
        # Key metric labels should appear
        for label in ["Sharpe Ratio", "Max Drawdown", "CAGR", "Deflated SR"]:
            assert label in html, f"Missing metric: {label}"

    def test_tearsheet_writes_file(self, tmp_path):
        from quantsim.reporting.tearsheet import generate_tearsheet

        equity = self.make_equity()
        output = str(tmp_path / "test_tearsheet.html")
        generate_tearsheet(equity, strategy_name="Test", output_path=output)
        assert os.path.exists(output)
        assert os.path.getsize(output) > 5000

    def test_tearsheet_with_trades_and_benchmark(self):
        from quantsim.reporting.tearsheet import generate_tearsheet

        equity = self.make_equity()
        benchmark = self.make_equity(seed=99)

        trades = pd.DataFrame(
            [
                {
                    "asset_id": "SPY",
                    "direction": "LONG",
                    "entry_price": 400.0,
                    "exit_price": 420.0,
                    "realized_pnl": 2000.0,
                    "strategy_id": "sma",
                    "holding_bars": 21,
                }
            ]
        )

        html = generate_tearsheet(
            equity,
            strategy_name="Full Test",
            trades_df=trades,
            benchmark=benchmark,
        )
        assert "SPY" in html
        assert "Benchmark" in html
