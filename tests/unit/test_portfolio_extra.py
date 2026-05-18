"""Extra portfolio coverage: Black-Litterman, MVO edge cases, EW edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.portfolio import BlackLittermanAllocator, EqualWeight, MeanVariance


def _returns_history(panel: pd.DataFrame) -> pd.DataFrame:
    wide = panel.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    return np.log(wide / wide.shift(1)).dropna()


def test_black_litterman_returns_series(small_panel):
    rets = _returns_history(small_panel)
    alphas = pd.Series(np.linspace(-0.02, 0.02, len(rets.columns)), index=rets.columns)
    w = BlackLittermanAllocator().allocate(alphas, rets)
    assert isinstance(w, pd.Series)
    assert not w.empty


def test_black_litterman_empty_alphas():
    w = BlackLittermanAllocator().allocate(pd.Series(dtype=float))
    assert w.empty


def test_black_litterman_no_history(small_panel):
    cols = small_panel["ticker"].unique()
    alphas = pd.Series(0.01, index=cols)
    w = BlackLittermanAllocator().allocate(alphas, None)
    assert isinstance(w, pd.Series)


def test_mvo_empty_alphas():
    assert MeanVariance().allocate(pd.Series(dtype=float)).empty


def test_mvo_no_history(small_panel):
    cols = small_panel["ticker"].unique()
    alphas = pd.Series(np.linspace(0.01, 0.05, len(cols)), index=cols)
    w = MeanVariance().allocate(alphas, None)
    assert w.abs().sum() <= 1.0 + 1e-6


def test_equal_weight_empty():
    assert EqualWeight().allocate(pd.Series(dtype=float)).empty


def test_equal_weight_all_zero_alpha_signed():
    alphas = pd.Series({"A": 0.0, "B": 0.0})
    w = EqualWeight(mode="signed").allocate(alphas)
    assert (w == 0.0).all()


def test_equal_weight_invalid_mode():
    import pytest

    with pytest.raises(ValueError):
        EqualWeight(mode="banana")
