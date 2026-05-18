from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.metrics.attribution import factor_exposure
from quantforge.metrics.factor_exposure import fama_french_betas


def _factor_data(n: int = 600, seed: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n)
    mkt = rng.normal(0.0003, 0.009, n)
    smb = rng.normal(0.0, 0.004, n)
    hml = rng.normal(0.0, 0.004, n)
    # Portfolio with a known market beta of 1.2 and small alpha.
    port = 0.0002 + 1.2 * mkt + 0.3 * smb + rng.normal(0, 0.002, n)
    factors = pd.DataFrame({"Mkt-RF": mkt, "SMB": smb, "HML": hml}, index=idx)
    return pd.Series(port, index=idx), factors


def test_factor_exposure_recovers_market_beta():
    port, factors = _factor_data()
    res = factor_exposure(port, factors)
    assert abs(res.betas["Mkt-RF"] - 1.2) < 0.15
    assert res.n_obs == len(port)
    assert res.newey_west_lag >= 1


def test_factor_exposure_insufficient_data():
    port = pd.Series([0.01, 0.02, 0.03])
    factors = pd.DataFrame({"Mkt-RF": [0.01, 0.02, 0.03]})
    res = factor_exposure(port, factors)
    assert np.isnan(res.alpha)


def test_fama_french_betas_with_rf_column():
    port, factors = _factor_data()
    factors = factors.reset_index().rename(columns={"index": "date"})
    factors["RF"] = 0.00005
    res = fama_french_betas(port, factors, factors=("Mkt-RF", "SMB", "HML"))
    assert "Mkt-RF" in res.betas.index
    assert np.isfinite(res.r_squared)
