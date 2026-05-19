from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.features.microstructure import amihud_illiquidity, kyle_lambda, roll_spread


def _series(n: int = 120, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    returns = pd.Series(rng.normal(0, 0.01, n), index=idx)
    dollar_volume = pd.Series(rng.uniform(1e6, 5e6, n), index=idx)
    return returns, dollar_volume


def test_amihud_illiquidity_non_negative():
    returns, dv = _series()
    illiq = amihud_illiquidity(returns, dv, window=21)
    assert (illiq.dropna() >= 0).all()
    assert illiq.notna().sum() > 0


def test_amihud_handles_zero_volume():
    returns, dv = _series()
    dv.iloc[:10] = 0.0
    illiq = amihud_illiquidity(returns, dv, window=5)
    # Zero volume becomes NaN, not inf.
    assert not np.isinf(illiq.dropna()).any()


def test_roll_spread_non_negative():
    rng = np.random.default_rng(1)
    close = pd.Series(
        100.0 + np.cumsum(rng.normal(0, 0.5, 200)),
        index=pd.bdate_range("2023-01-02", periods=200),
    )
    spread = roll_spread(close, window=21)
    assert (spread.dropna() >= 0).all()


def test_kyle_lambda_runs():
    returns, dv = _series(n=80)
    signed = dv * np.sign(returns)
    lam = kyle_lambda(returns, signed, window=21)
    assert isinstance(lam, pd.Series)
    assert len(lam) == len(returns)


def test_kyle_lambda_short_history_returns_all_nan():
    returns, dv = _series(n=10)
    lam = kyle_lambda(returns, dv, window=21)
    assert lam.isna().all()
