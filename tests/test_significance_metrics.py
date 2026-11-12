"""
Deflated Sharpe ratio and Sortino ratio against hand-computed values.

The deflated Sharpe ratio is the check that keeps a parameter sweep honest,
so it must be a probability, fall as more trials are run, and match the
Bailey and Lopez de Prado (2014) formula on a case worked out by hand.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quantsim.reporting.analytics import PerformanceAnalytics, deflated_sharpe_ratio


def test_deflated_sharpe_hand_computed_case():
    """
    Per-period SR = 0.1, T = 253 observations, N = 10 trials whose Sharpe
    ratios have variance V = 0.0025, normal returns (skew 0, kurtosis 3).

    Expected maximum Sharpe under the null:
      Z1 = Phi^-1(1 - 1/10)       = 1.2815516
      Z2 = Phi^-1(1 - 1/(10 e))   = 1.7892418
      SR0 = sqrt(V) * ((1 - 0.5772157) * Z1 + 0.5772157 * Z2) = 0.0787299
    Test statistic:
      z = (0.1 - 0.0787299) * sqrt(252) / sqrt(1 - 0 * 0.1 + (3 - 1) / 4 * 0.1^2)
        = 0.3368112
    DSR = Phi(0.3368112) = 0.6318704
    """
    dsr = deflated_sharpe_ratio(
        sharpe=0.1,
        n_obs=253,
        n_trials=10,
        trial_sharpe_variance=0.0025,
        skewness=0.0,
        kurtosis=3.0,
    )
    assert dsr == pytest.approx(0.6318704, abs=1e-6)


def test_deflated_sharpe_falls_as_trials_grow():
    values = [
        deflated_sharpe_ratio(sharpe=0.08, n_obs=1000, n_trials=n, trial_sharpe_variance=0.001)
        for n in (1, 2, 5, 10, 100, 1000)
    ]
    assert all(a > b for a, b in zip(values, values[1:]))


@settings(max_examples=300, deadline=None)
@given(
    sharpe=st.floats(-1, 1),
    n_obs=st.integers(2, 5000),
    n_trials=st.integers(1, 10_000),
    variance=st.floats(0, 0.1),
    skewness=st.floats(-3, 3),
    kurtosis=st.floats(1, 30),
)
def test_deflated_sharpe_is_a_probability(sharpe, n_obs, n_trials, variance, skewness, kurtosis):
    dsr = deflated_sharpe_ratio(sharpe, n_obs, n_trials, variance, skewness, kurtosis)
    assert 0.0 <= dsr <= 1.0


def test_compute_all_reports_dsr_as_probability_with_095_threshold():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.002, 0.01, 756)
    curve = pd.Series(
        100_000 * np.cumprod(1 + returns), index=pd.date_range("2018-01-01", periods=756, freq="B")
    )

    results = PerformanceAnalytics(risk_free_rate=0.0).compute_all(curve)

    assert 0.0 <= results["deflated_sharpe_ratio"] <= 1.0
    assert results["deflated_sharpe_significant"] == (results["deflated_sharpe_ratio"] > 0.95)


def test_sortino_uses_downside_deviation_over_all_periods():
    """
    Returns 2%, -1%, 3%, -2%, 1%, 0% with a 0% target:
      mean = 0.005
      downside deviation = sqrt((0.01^2 + 0.02^2) / 6) = 0.0091287
      Sortino = 0.005 / 0.0091287 * sqrt(252) = 8.694826
    """
    returns = [0.02, -0.01, 0.03, -0.02, 0.01, 0.0]
    equity = 100.0 * np.cumprod([1.0] + [1 + r for r in returns])
    curve = pd.Series(equity, index=pd.date_range("2020-01-01", periods=7, freq="B"))

    results = PerformanceAnalytics(risk_free_rate=0.0).compute_all(curve)

    assert results["sortino_ratio"] == pytest.approx(8.694826, rel=1e-6)
