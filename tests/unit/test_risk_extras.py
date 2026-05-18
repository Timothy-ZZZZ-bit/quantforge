from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.risk.drawdown import (
    drawdown_series,
    max_drawdown,
    recovery_time,
    time_underwater,
)
from quantforge.risk.sizing import fractional_kelly, vol_target_scale
from quantforge.risk.stress import STRESS_WINDOWS, historical_stress


def _returns(seed: int = 0, n: int = 1000) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(0.0003, 0.01, n),
        index=pd.bdate_range("2007-01-03", periods=n),
    )


def test_drawdown_series_non_positive():
    dd = drawdown_series(_returns())
    assert (dd <= 1e-9).all()


def test_drawdown_series_empty():
    assert drawdown_series(pd.Series(dtype=float)).empty


def test_max_drawdown_matches_series_min():
    r = _returns()
    assert max_drawdown(r) == drawdown_series(r).min()


def test_time_underwater_non_negative():
    assert time_underwater(_returns()) >= 0


def test_time_underwater_empty():
    assert time_underwater(pd.Series(dtype=float)) == 0


def test_recovery_time_non_negative():
    assert recovery_time(_returns()) >= 0


def test_recovery_time_empty():
    assert recovery_time(pd.Series(dtype=float)) == 0


def test_fractional_kelly_finite():
    f = fractional_kelly(_returns(), fraction=0.25)
    assert np.isfinite(f)


def test_fractional_kelly_short_history_nan():
    assert np.isnan(fractional_kelly(pd.Series([0.01, 0.02])))


def test_vol_target_scale():
    assert vol_target_scale(0.20, 0.10) == 0.5
    assert vol_target_scale(0.0, 0.10) == 0.0


def test_historical_stress_returns_all_windows():
    # Returns spanning 2007-2010 will overlap the GFC window.
    results = historical_stress(_returns(n=2000))
    assert len(results) == len(STRESS_WINDOWS)
    gfc = next(r for r in results if "GFC" in r.name)
    assert gfc.n_obs > 0


def test_historical_stress_empty_window():
    # Returns far from any stress window -> n_obs == 0.
    r = pd.Series(
        np.random.default_rng(0).normal(0, 0.01, 100),
        index=pd.bdate_range("2030-01-02", periods=100),
    )
    results = historical_stress(r)
    assert all(res.n_obs == 0 for res in results)
