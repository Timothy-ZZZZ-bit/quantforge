"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantforge.data import synthetic_equity_panel


@pytest.fixture
def small_panel() -> pd.DataFrame:
    """A small deterministic panel suitable for unit tests."""
    return synthetic_equity_panel(n_tickers=5, n_days=300, seed=7)


@pytest.fixture
def medium_panel() -> pd.DataFrame:
    """A medium deterministic panel for integration-leaning tests."""
    return synthetic_equity_panel(n_tickers=10, n_days=252 * 3, seed=11)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
