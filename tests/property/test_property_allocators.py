from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from quantforge.portfolio.hrp import hrp_weights_from_cov
from quantforge.portfolio.risk_parity import solve_erc_spinu


def _psd_cov(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 1, size=(n, n))
    return A @ A.T + 1e-4 * np.eye(n)


@given(
    seed=st.integers(min_value=0, max_value=2**16 - 1),
    n=st.integers(min_value=2, max_value=8),
)
@settings(deadline=None, max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_hrp_weights_sum_to_one(seed: int, n: int) -> None:
    cov = _psd_cov(seed, n)
    w = hrp_weights_from_cov(cov)
    assert w.shape == (n,)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w > -1e-9).all()


@given(
    seed=st.integers(min_value=0, max_value=2**16 - 1),
    n=st.integers(min_value=2, max_value=6),
)
@settings(deadline=None, max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_erc_equal_marginal_risk(seed: int, n: int) -> None:
    cov = _psd_cov(seed, n)
    w = solve_erc_spinu(cov)
    rc = w * (cov @ w)
    # Equal-risk-contribution: dispersion of rc is small.
    assert rc.std(ddof=0) < 1e-4
