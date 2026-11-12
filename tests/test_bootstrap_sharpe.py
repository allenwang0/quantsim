"""
Bootstrap interval for the annualized Sharpe ratio.

The interval is what tells a reader how little one Sharpe number says, so it
must be reproducible from a seed, use the caller's risk-free rate, and be as
wide as theory says: the standard error of an annualized Sharpe from one year
of daily data is about 1.0, so a 95% interval is about 3.9 wide.
Returns are seeded synthetic draws.
"""
import numpy as np
import pandas as pd

from quantsim.reporting.advanced import AdvancedAnalytics


def one_year_of_returns(annual_sharpe=0.8, daily_vol=0.01, seed=21):
    rng = np.random.default_rng(seed)
    daily_mean = annual_sharpe / np.sqrt(252) * daily_vol
    return pd.Series(rng.normal(daily_mean, daily_vol, 252))


def test_same_seed_gives_same_interval():
    returns = one_year_of_returns()
    first = AdvancedAnalytics.monte_carlo_sharpe(returns, n_simulations=2000, risk_free_rate=0.0, seed=7)
    second = AdvancedAnalytics.monte_carlo_sharpe(returns, n_simulations=2000, risk_free_rate=0.0, seed=7)
    assert (first["sharpe_ci_low"], first["sharpe_ci_high"]) == (second["sharpe_ci_low"], second["sharpe_ci_high"])


def test_interval_width_matches_theory_for_one_year():
    result = AdvancedAnalytics.monte_carlo_sharpe(
        one_year_of_returns(), n_simulations=5000, risk_free_rate=0.0, seed=1
    )
    width = result["sharpe_ci_high"] - result["sharpe_ci_low"]
    assert 3.0 <= width <= 5.0


def test_risk_free_rate_shifts_the_point_estimate():
    returns = one_year_of_returns()
    zero = AdvancedAnalytics.monte_carlo_sharpe(returns, n_simulations=200, risk_free_rate=0.0, seed=3)
    five = AdvancedAnalytics.monte_carlo_sharpe(returns, n_simulations=200, risk_free_rate=0.05, seed=3)
    expected_shift = 0.05 / 252 / returns.std() * np.sqrt(252)
    assert abs(zero["sharpe_point_estimate"] - five["sharpe_point_estimate"] - expected_shift) < 1e-6
