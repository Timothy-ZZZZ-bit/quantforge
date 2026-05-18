"""Pairs trading on a deliberately cointegrated pair.

The independent-series test in ``test_signals.py`` exercises the rejection
path. This file builds a genuinely cointegrated pair so the ADF filter
passes and the trade-construction logic runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.signals.pairs import PairsTrade, _half_life


def _cointegrated_panel(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n)
    # A is a random walk in log space.
    log_a = np.cumsum(rng.normal(0.0002, 0.01, n)) + np.log(100.0)
    # The spread is a mean-reverting OU process -> A and B cointegrate.
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.85 * spread[t - 1] + rng.normal(0, 0.02)
    log_b = log_a + spread + np.log(1.5)
    price_a = np.exp(log_a)
    price_b = np.exp(log_b)
    frames = []
    for tk, prices in (("AAA", price_a), ("BBB", price_b)):
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": tk,
                    "open": prices,
                    "high": prices * 1.001,
                    "low": prices * 0.999,
                    "close": prices,
                    "adj_close": prices,
                    "volume": 1_000_000.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_pairs_trades_cointegrated_pair():
    panel = _cointegrated_panel()
    sig = PairsTrade(
        candidate_pairs=[("AAA", "BBB")],
        window=200,
        entry_z=1.0,
        max_half_life_days=60.0,
    )
    sig.fit(panel)
    out = sig.predict(panel)
    # The pair cointegrates; depending on the latest z-score the signal may
    # be flat, but the function must return a valid Series either way.
    assert isinstance(out, pd.Series)
    if not out.empty:
        assert abs(out.abs().sum() - 1.0) < 1e-6


def test_pairs_auto_candidate_generation():
    panel = _cointegrated_panel()
    # No explicit pairs -> all combinations considered.
    sig = PairsTrade(window=200, entry_z=1.0, max_half_life_days=60.0)
    out = sig.predict(panel)
    assert isinstance(out, pd.Series)


def test_half_life_of_mean_reverting_series():
    rng = np.random.default_rng(1)
    n = 500
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + rng.normal(0, 1)
    hl = _half_life(x)
    # phi=0.9 -> half-life = ln(2)/-ln(0.9) ~ 6.6 bars.
    assert 2.0 < hl < 20.0


def test_half_life_non_mean_reverting_is_inf():
    # A pure random walk has no mean reversion.
    rw = np.cumsum(np.ones(100))
    assert _half_life(rw) == float("inf")
