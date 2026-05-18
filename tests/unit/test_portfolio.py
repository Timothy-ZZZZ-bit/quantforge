from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantforge.portfolio import (
    Constraints,
    EqualRiskContribution,
    EqualWeight,
    MeanVariance,
    apply_constraints,
)


def test_equal_weight_signed():
    alphas = pd.Series({"A": 1.0, "B": -1.0, "C": 0.5})
    w = EqualWeight().allocate(alphas)
    assert w.sum() == pytest.approx(
        1.0 / 3.0, abs=1e-9
    )  # 2 longs (+1/3 each), 1 short (-1/3) -> 1/3 net
    assert w.abs().sum() == pytest.approx(1.0, abs=1e-9)


def test_equal_weight_signless():
    alphas = pd.Series({"A": 1.0, "B": 2.0, "C": -3.0})
    w = EqualWeight(mode="signless").allocate(alphas)
    assert (w > 0).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_apply_constraints_clips_and_rescales():
    # Feasible case: caps are loose enough that we can hit the gross leverage.
    w = pd.Series({"A": 0.6, "B": 0.6, "C": -0.6, "D": 0.6, "E": -0.6})
    c = Constraints(max_weight=0.3, min_weight=-0.3, gross_leverage=1.0)
    w2 = apply_constraints(w, c)
    assert (w2.abs() <= 0.3 + 1e-9).all()
    assert w2.abs().sum() == pytest.approx(1.0, abs=1e-6)


def test_apply_constraints_infeasible_respects_caps():
    # Caps make gross leverage 1.0 infeasible on 2 names; caps should win.
    w = pd.Series({"A": 1.0, "B": -1.0})
    c = Constraints(max_weight=0.3, min_weight=-0.3, gross_leverage=1.0)
    w2 = apply_constraints(w, c)
    assert (w2.abs() <= 0.3 + 1e-9).all()
    assert w2.abs().sum() <= 1.0 + 1e-9


def test_apply_constraints_turnover_cap():
    prior = pd.Series({"A": 0.5, "B": 0.5})
    new = pd.Series({"A": 1.0, "B": 0.0})
    c = Constraints(
        max_weight=1.0, min_weight=-1.0, gross_leverage=1.0, turnover_cap=0.5
    )
    w = apply_constraints(new, c, prior_weights=prior)
    assert float((w - prior).abs().sum()) == pytest.approx(0.5, abs=1e-9)


def test_mvo_returns_series(small_panel):
    wide = small_panel.pivot(
        index="date", columns="ticker", values="adj_close"
    ).sort_index()
    rets = np.log(wide / wide.shift(1)).dropna()
    alphas = pd.Series(np.linspace(0.01, 0.05, len(wide.columns)), index=wide.columns)
    w = MeanVariance().allocate(alphas, rets)
    assert isinstance(w, pd.Series)
    assert w.abs().sum() <= 1.01


def test_erc_returns_series(small_panel):
    wide = small_panel.pivot(
        index="date", columns="ticker", values="adj_close"
    ).sort_index()
    rets = np.log(wide / wide.shift(1)).dropna()
    alphas = pd.Series(1.0, index=wide.columns)
    w = EqualRiskContribution().allocate(alphas, rets)
    assert isinstance(w, pd.Series)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
