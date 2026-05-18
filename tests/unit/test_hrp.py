"""HRP unit tests.

Includes a small reproducibility check that the algorithm yields positive,
normalized weights with reasonable behavior on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantforge.portfolio.hrp import HRPAllocator, hrp_weights_from_cov
from quantforge.portfolio.risk_parity import solve_erc_slsqp, solve_erc_spinu


def _example_cov_from_corr() -> np.ndarray:
    # Two clusters: {0, 1} and {2, 3}; cross-cluster correlation low.
    corr = np.array(
        [
            [1.00, 0.90, 0.10, 0.05],
            [0.90, 1.00, 0.05, 0.10],
            [0.10, 0.05, 1.00, 0.85],
            [0.05, 0.10, 0.85, 1.00],
        ]
    )
    sd = np.array([0.20, 0.18, 0.30, 0.25])
    return corr * np.outer(sd, sd)


def test_hrp_weights_sum_to_one():
    cov = _example_cov_from_corr()
    w = hrp_weights_from_cov(cov)
    assert w.shape == (4,)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w > 0).all()


def test_hrp_concentration_in_low_vol():
    cov = _example_cov_from_corr()
    w = hrp_weights_from_cov(cov)
    # Lower-vol cluster (0, 1) should attract more weight than the higher-vol one (2, 3).
    assert w[:2].sum() > w[2:].sum()


def test_erc_spinu_equal_risk():
    cov = _example_cov_from_corr()
    w = solve_erc_spinu(cov)
    rc = w * (cov @ w)
    # Equal risk contribution -> rc has very low dispersion.
    assert rc.std(ddof=0) < 1e-6


def test_erc_spinu_matches_slsqp():
    cov = _example_cov_from_corr()
    w_spinu = solve_erc_spinu(cov)
    w_slsqp = solve_erc_slsqp(cov)
    # SLSQP and Spinu should agree to ~1e-3 on this small problem.
    np.testing.assert_allclose(w_spinu, w_slsqp, atol=1e-3)


def test_hrp_allocator_returns_series(small_panel):
    wide = small_panel.pivot(
        index="date", columns="ticker", values="adj_close"
    ).sort_index()
    rets = np.log(wide / wide.shift(1)).dropna()
    alphas = pd.Series(1.0, index=wide.columns)
    w = HRPAllocator().allocate(alphas, rets)
    assert isinstance(w, pd.Series)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= 0).all()
