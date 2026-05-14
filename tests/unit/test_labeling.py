from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.features.labeling import (
    TripleBarrierConfig,
    meta_labels,
    triple_barrier,
)


def test_triple_barrier_upper_hit():
    # Monotone-up price series should mostly produce +1 labels.
    prices = pd.Series(np.linspace(100.0, 200.0, 100), index=pd.date_range("2024", periods=100))
    vol = pd.Series(0.01, index=prices.index)
    out = triple_barrier(prices, vol, config=TripleBarrierConfig(pt=2.0, sl=2.0, vertical=10))
    assert (out["bin"] >= 0).all()
    assert (out["bin"] > 0).sum() >= 1


def test_triple_barrier_lower_hit():
    prices = pd.Series(np.linspace(200.0, 100.0, 100), index=pd.date_range("2024", periods=100))
    vol = pd.Series(0.01, index=prices.index)
    out = triple_barrier(prices, vol, config=TripleBarrierConfig(pt=2.0, sl=2.0, vertical=10))
    assert (out["bin"] <= 0).all()
    assert (out["bin"] < 0).sum() >= 1


def test_meta_labels():
    pred = pd.Series([1, -1, 1, 0])
    tb = pd.Series([1, -1, -1, 0])
    ml = meta_labels(pred, tb)
    np.testing.assert_array_equal(ml.to_numpy(), [1, 1, 0, 1])
