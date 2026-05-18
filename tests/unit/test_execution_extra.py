from __future__ import annotations

import pandas as pd

from quantforge.backtest.walk_forward import walk_forward_splits
from quantforge.execution.fills import fill_at_next_open
from quantforge.execution.slippage import participation_slippage
from quantforge.risk.cvar import expected_shortfall


def test_fill_at_next_open_caps_participation():
    fill = fill_at_next_open(
        ticker="X",
        target_shares_delta=1_000_000.0,
        next_open=100.0,
        bar_volume=100_000.0,
        participation_cap=0.10,
        cost_per_dollar=0.001,
    )
    # Capped at 10% of 100k volume = 10k shares.
    assert abs(fill.shares) <= 10_000.0 + 1e-6
    assert fill.cost > 0


def test_fill_at_next_open_zero_price():
    fill = fill_at_next_open("X", 100.0, 0.0, 1000.0, 0.1, 0.001)
    assert fill.shares == 0.0


def test_participation_slippage_caps():
    s_small = participation_slippage(100.0, 1_000_000.0)
    s_large = participation_slippage(10_000_000.0, 1_000_000.0)
    assert s_large >= s_small
    assert participation_slippage(100.0, 0.0) == 0.0


def test_walk_forward_empty_index():
    splits = list(walk_forward_splits(pd.DatetimeIndex([])))
    assert splits == []


def test_walk_forward_rejects_unsorted():
    import pytest

    idx = pd.DatetimeIndex(["2020-01-03", "2020-01-02"])
    with pytest.raises(ValueError):
        list(walk_forward_splits(idx))


def test_expected_shortfall_empty():
    import numpy as np

    assert np.isnan(expected_shortfall(pd.Series(dtype=float)))
