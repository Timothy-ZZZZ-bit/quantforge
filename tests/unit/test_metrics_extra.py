"""Extra coverage for performance metrics and PSR edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.metrics.performance import (
    annualized_return,
    annualized_volatility,
    calmar,
    hit_rate,
    max_drawdown,
    omega,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    time_underwater,
)
from quantforge.metrics.psr import deflated_sharpe, probabilistic_sharpe


def test_metrics_empty_series_return_nan():
    empty = pd.Series(dtype=float)
    assert np.isnan(annualized_return(empty))
    assert np.isnan(annualized_volatility(empty))
    assert np.isnan(sharpe_ratio(empty))
    assert np.isnan(sortino_ratio(empty))
    assert np.isnan(max_drawdown(empty))
    assert np.isnan(calmar(empty))
    assert np.isnan(hit_rate(empty))


def test_omega_above_and_below_threshold():
    r = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    assert omega(r, threshold=0.0) > 0
    # All returns below a very high threshold -> no gains -> omega 0.
    assert omega(r, threshold=10.0) == 0.0


def test_omega_empty():
    assert np.isnan(omega(pd.Series(dtype=float)))


def test_profit_factor_positive():
    r = pd.Series([0.02, -0.01, 0.03, -0.02])
    assert profit_factor(r) > 0


def test_calmar_zero_drawdown_is_nan():
    r = pd.Series([0.01] * 252)
    assert np.isnan(calmar(r))


def test_sortino_all_positive_returns_nan_denominator():
    r = pd.Series([0.01] * 100)
    # No downside -> downside deviation 0 -> nan.
    assert np.isnan(sortino_ratio(r))


def test_time_underwater_with_drawdown():
    r = pd.Series([0.1, -0.2, -0.1, 0.05, 0.5])
    assert time_underwater(r) >= 1


def test_probabilistic_sharpe_short_series_nan():
    assert np.isnan(probabilistic_sharpe(pd.Series([0.01, 0.02])))


def test_deflated_sharpe_with_sr_estimates():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0008, 0.01, 1500))
    dsr = deflated_sharpe(r, n_trials=20, sr_estimates=[0.1, 0.5, 0.9, 1.2, 0.3])
    assert 0.0 <= dsr <= 1.0


def test_deflated_sharpe_empty_is_nan():
    assert np.isnan(deflated_sharpe(pd.Series(dtype=float), n_trials=5))
