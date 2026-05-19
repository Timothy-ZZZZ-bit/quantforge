from __future__ import annotations

import math

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from quantforge.features.returns import log_returns, reconstruct_prices_from_log_returns
from quantforge.metrics.performance import sharpe_ratio


@given(
    base=st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
    pcts=st.lists(
        st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=200,
    ),
)
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None, max_examples=50)
def test_log_returns_roundtrip(base: float, pcts: list[float]) -> None:
    prices = pd.Series(base * np.cumprod([1.0] + [1.0 + p for p in pcts]))
    lr = log_returns(prices, fill_na=True)
    recon = reconstruct_prices_from_log_returns(lr, base=prices.iloc[0])
    np.testing.assert_allclose(recon.to_numpy(), prices.iloc[1:].to_numpy(), atol=1e-9, rtol=1e-9)


@given(
    seed=st.integers(min_value=0, max_value=2**16 - 1),
    n=st.integers(min_value=200, max_value=2000),
    scale=st.floats(min_value=1e-4, max_value=10.0, allow_nan=False),
)
@settings(deadline=None, max_examples=30)
def test_sharpe_invariant_to_scale(seed: int, n: int, scale: float) -> None:
    rng = np.random.default_rng(seed)
    r = pd.Series(rng.normal(0.0005, 0.01, n))
    s1 = sharpe_ratio(r)
    s2 = sharpe_ratio(r * scale)
    # Sharpe is scale-invariant under multiplicative rescaling.
    assert math.isclose(s1, s2, rel_tol=1e-9, abs_tol=1e-9)
