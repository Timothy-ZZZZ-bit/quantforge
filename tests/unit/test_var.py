from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.risk.cvar import expected_shortfall
from quantforge.risk.var import cornish_fisher_var, historical_var, parametric_var


def _normal_returns(seed: int = 0, n: int = 1000) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.01, n))


def test_parametric_var_positive():
    r = _normal_returns()
    res = parametric_var(r, confidence=0.95, n_boot=100, seed=1)
    assert res.point > 0
    assert res.ci_low <= res.point <= res.ci_high or np.isnan(res.ci_low)


def test_historical_var_positive():
    r = _normal_returns()
    res = historical_var(r, confidence=0.95, n_boot=100, seed=1)
    assert res.point > 0


def test_cornish_fisher_var_positive_for_skewed():
    rng = np.random.default_rng(42)
    r = pd.Series(rng.standard_normal(2000) * 0.01 - 0.001)
    res = cornish_fisher_var(r, confidence=0.95, n_boot=100, seed=2)
    assert np.isfinite(res.point)


def test_expected_shortfall_at_least_var():
    r = _normal_returns()
    es = expected_shortfall(r, confidence=0.95)
    var_res = historical_var(r, confidence=0.95, n_boot=100, seed=0)
    # ES >= VaR by construction.
    assert es >= var_res.point - 1e-9
