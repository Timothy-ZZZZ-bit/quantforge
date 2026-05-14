from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantforge.features.returns import (
    annualize_volatility,
    cumulative_returns,
    frac_diff_ffd,
    log_returns,
    reconstruct_prices_from_log_returns,
    simple_returns,
)


def test_log_returns_basic():
    prices = pd.Series([100.0, 110.0, 121.0])
    lr = log_returns(prices)
    assert np.isnan(lr.iloc[0])
    assert lr.iloc[1] == pytest.approx(np.log(110.0 / 100.0), rel=1e-12)
    assert lr.iloc[2] == pytest.approx(np.log(121.0 / 110.0), rel=1e-12)


def test_log_returns_rejects_nonpositive():
    with pytest.raises(ValueError):
        log_returns(pd.Series([100.0, 0.0, 121.0]))


def test_simple_returns():
    prices = pd.Series([100.0, 110.0, 99.0])
    sr = simple_returns(prices, fill_na=True)
    assert sr.iloc[0] == pytest.approx(0.10)
    assert sr.iloc[1] == pytest.approx(-0.10)


def test_cumulative_returns_geometric():
    r = pd.Series([0.1, -0.05, 0.2])
    cum = cumulative_returns(r)
    assert cum.iloc[-1] == pytest.approx((1.1 * 0.95 * 1.2) - 1.0)


def test_cumulative_returns_additive():
    r = pd.Series([0.1, 0.05])
    cum = cumulative_returns(r, compounding="additive")
    assert cum.iloc[-1] == pytest.approx(0.15)


def test_cumulative_returns_invalid_mode():
    with pytest.raises(ValueError):
        cumulative_returns(pd.Series([0.0]), compounding="weird")


def test_annualize_volatility():
    assert annualize_volatility(0.01) == pytest.approx(0.01 * np.sqrt(252))
    with pytest.raises(ValueError):
        annualize_volatility(-1e-9)


def test_reconstruct_prices_from_log_returns_roundtrip():
    prices = pd.Series([100.0, 105.0, 110.0, 100.0, 95.0])
    lr = log_returns(prices, fill_na=True)
    reconstructed = reconstruct_prices_from_log_returns(lr, base=prices.iloc[0])
    np.testing.assert_allclose(reconstructed.to_numpy(), prices.iloc[1:].to_numpy(), atol=1e-10)


def test_frac_diff_ffd_preserves_length():
    series = pd.Series(np.log(np.linspace(100.0, 200.0, 500)))
    fd = frac_diff_ffd(series, d=0.5)
    assert len(fd) == len(series)
    assert fd.notna().sum() > 0


def test_frac_diff_d_zero_returns_input():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])
    fd = frac_diff_ffd(series, d=0.0)
    np.testing.assert_allclose(fd.dropna().to_numpy(), series.to_numpy()[-len(fd.dropna()):])


def test_frac_diff_d_one_is_difference():
    series = pd.Series(np.cumsum(np.ones(100)))
    fd = frac_diff_ffd(series, d=1.0).dropna()
    # Should be close to first differences (=1) for cumulative sum of ones.
    np.testing.assert_allclose(fd.iloc[1:].to_numpy(), np.ones(len(fd) - 1), atol=1e-6)


def test_frac_diff_invalid_d():
    series = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        frac_diff_ffd(series, d=2.0)
