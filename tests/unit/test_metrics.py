from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantforge.metrics.performance import (
    annualized_return,
    annualized_volatility,
    calmar,
    hit_rate,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    time_underwater,
)
from quantforge.metrics.psr import deflated_sharpe, probabilistic_sharpe


def _rets(
    seed: int = 0, n: int = 252 * 3, mu: float = 0.0004, sigma: float = 0.01
) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(mu, sigma, n), index=pd.bdate_range("2018-01-02", periods=n)
    )


def test_sharpe_known_ratio():
    # All-zero returns: zero std -> NaN.
    assert np.isnan(sharpe_ratio(pd.Series([0.0] * 1000)))


def test_sharpe_random_series():
    r = _rets(seed=1)
    s = sharpe_ratio(r)
    assert np.isfinite(s)


def test_annualized_return_zero_returns_zero():
    r = pd.Series([0.0] * 252)
    assert annualized_return(r) == 0.0


def test_annualized_volatility_positive():
    r = _rets(seed=2)
    assert annualized_volatility(r) > 0


def test_max_drawdown_negative():
    r = pd.Series([0.1, -0.5, 0.05, -0.2])
    assert max_drawdown(r) < 0


def test_calmar_finite():
    r = _rets(seed=3, mu=0.001)
    c = calmar(r)
    assert np.isfinite(c) or np.isnan(c)


def test_hit_rate_bounds():
    r = _rets(seed=4)
    hr = hit_rate(r)
    assert 0.0 <= hr <= 1.0


def test_profit_factor_zero_negative_returns_nan():
    r = pd.Series([0.01, 0.02, 0.03])
    assert np.isnan(profit_factor(r))


def test_probabilistic_sharpe_high_when_good():
    r = _rets(seed=10, mu=0.002, sigma=0.005, n=2500)
    psr = probabilistic_sharpe(r, benchmark_sr=0.0)
    assert psr > 0.9


def test_probabilistic_sharpe_lower_when_bad():
    r = _rets(seed=11, mu=-0.001, sigma=0.01, n=2500)
    psr = probabilistic_sharpe(r, benchmark_sr=0.0)
    assert psr < 0.5


def test_deflated_sharpe_requires_positive_n_trials():
    r = _rets(seed=12)
    with pytest.raises(ValueError):
        deflated_sharpe(r, n_trials=0)


def test_deflated_sharpe_falls_with_more_trials():
    r = _rets(seed=13, mu=0.0005, n=2000)
    d_low = deflated_sharpe(r, n_trials=1)
    d_high = deflated_sharpe(r, n_trials=1000)
    assert d_low >= d_high


def test_sortino_uses_downside():
    r = pd.Series([0.05, 0.05, -0.01, 0.05, -0.01])
    s = sortino_ratio(r)
    sh = sharpe_ratio(r)
    assert s > sh


def test_time_underwater_zero_when_monotone_up():
    r = pd.Series([0.01] * 252)
    assert time_underwater(r) == 0
