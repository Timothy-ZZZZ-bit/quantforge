from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantforge.backtest.cpcv import CombinatorialPurgedCV, cpcv_indices


def test_cpcv_split_count_matches_binomial():
    splits = cpcv_indices(n_obs=100, n_groups=6, n_test_groups=2, embargo=0)
    # C(6, 2) = 15
    assert len(splits) == 15


def test_cpcv_train_and_test_disjoint():
    splits = cpcv_indices(n_obs=100, n_groups=5, n_test_groups=2, embargo=3)
    for s in splits:
        assert set(s.train_idx).isdisjoint(set(s.test_idx))


def test_cpcv_embargo_extends_exclusion():
    splits = cpcv_indices(n_obs=100, n_groups=5, n_test_groups=2, embargo=5)
    for s in splits:
        # No index within `embargo` of any test index should be in the train set.
        nearest = np.min(np.abs(np.subtract.outer(s.train_idx, s.test_idx)), axis=1)
        assert (nearest > 5).all()


def test_cpcv_invalid_test_groups():
    with pytest.raises(ValueError):
        cpcv_indices(n_obs=10, n_groups=2, n_test_groups=2)


def test_cpcv_wrapper_split():
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    cv = CombinatorialPurgedCV(idx, n_groups=6, n_test_groups=2, embargo=0)
    pairs = cv.split()
    assert len(pairs) == 15
    assert cv.n_paths() == 5  # 2 * 15 / 6
