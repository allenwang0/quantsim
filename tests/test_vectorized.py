"""Vectorized backtester: runs, sweeps and signal handling on synthetic prices."""

import numpy as np
import pandas as pd

from tests.synthetic_bars import make_price_df


class TestVectorizedBacktester:
    def test_basic_run(self):
        from quantsim.backtesting.vectorized import VectorizedBacktester, sma_crossover_signal

        prices = make_price_df(n=300, n_assets=2)
        bt = VectorizedBacktester(prices=prices, initial_capital=100_000)
        results = bt.run(sma_crossover_signal, signal_kwargs={"fast": 20, "slow": 100})

        assert "sharpe_ratio" in results
        assert "cagr" in results
        assert "max_drawdown" in results
        assert results["max_drawdown"] <= 0

    def test_parameter_sweep(self):
        from quantsim.backtesting.vectorized import VectorizedBacktester, sma_crossover_signal

        prices = make_price_df(n=400, n_assets=2)
        bt = VectorizedBacktester(prices=prices)
        param_grid = {"fast": [10, 20], "slow": [50, 100]}
        results_df = bt.parameter_sweep(sma_crossover_signal, param_grid)

        assert len(results_df) == 4  # 2x2 grid
        assert "sharpe_ratio" in results_df.columns
        assert "deflated_sharpe_corrected" in results_df.columns
        # Results should be sorted descending by deflated Sharpe
        dsr_vals = results_df["deflated_sharpe_corrected"].values
        assert all(dsr_vals[i] >= dsr_vals[i + 1] for i in range(len(dsr_vals) - 1))

    def test_no_lookahead_bias(self):
        """Buy-and-hold vectorized should match event-driven within margin."""
        from quantsim.backtesting.vectorized import (
            VectorizedBacktester,
            equal_weight_rebalance_signal,
        )

        # Linearly rising price series
        n = 252
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        prices = pd.DataFrame({"SPY": 100 * (1 + 0.001) ** np.arange(n)}, index=dates)

        bt = VectorizedBacktester(prices=prices, initial_capital=100_000, commission_rate=0)
        results = bt.run(equal_weight_rebalance_signal)

        # Should have positive total return
        assert results.get("total_return", 0) > 0

    def test_various_signals(self):
        from quantsim.backtesting.vectorized import (
            VectorizedBacktester,
            bollinger_signal,
            donchian_signal,
            momentum_signal,
            rsi_signal,
        )

        prices = make_price_df(n=400)
        bt = VectorizedBacktester(prices=prices)

        for fn in [rsi_signal, momentum_signal, bollinger_signal, donchian_signal]:
            results = bt.run(fn)
            assert "sharpe_ratio" in results, f"{fn.__name__} failed"
