"""Extra signal coverage: carry proxy, OU mean reversion, ML signal, IC."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.data import synthetic_equity_panel
from quantforge.signals.base import information_coefficient
from quantforge.signals.carry import ETFCarryProxy
from quantforge.signals.mean_reversion import OUMeanReversion
from quantforge.signals.ml_signal import MLSignal


def test_carry_proxy_runs(medium_panel):
    sig = ETFCarryProxy(lookback=63, skip=5)
    sig.fit(medium_panel)
    out = sig.predict(medium_panel)
    assert isinstance(out, pd.Series)
    if not out.empty:
        assert abs(out.abs().sum() - 1.0) < 1e-6


def test_carry_proxy_short_history_empty():
    panel = synthetic_equity_panel(n_tickers=3, n_days=40, seed=1)
    out = ETFCarryProxy(lookback=252).predict(panel)
    assert out.empty


def test_ou_mean_reversion_runs():
    panel = synthetic_equity_panel(n_tickers=6, n_days=300, seed=3)
    # Rename one ticker to SPY so the market proxy exists.
    panel = panel.copy()
    panel.loc[panel["ticker"] == "SYN000", "ticker"] = "SPY"
    sig = OUMeanReversion(market_ticker="SPY", window=63)
    sig.fit(panel)
    out = sig.predict(panel)
    assert isinstance(out, pd.Series)


def test_ou_missing_market_returns_empty(medium_panel):
    sig = OUMeanReversion(market_ticker="NOTHERE", window=63)
    assert sig.predict(medium_panel).empty


def test_ml_signal_fit_predict():
    panel = synthetic_equity_panel(n_tickers=8, n_days=400, seed=5)
    sig = MLSignal(n_estimators=20, max_depth=2)
    sig.fit(panel)
    out = sig.predict(panel)
    assert isinstance(out, pd.Series)


def test_ml_signal_predict_without_fit_returns_empty():
    panel = synthetic_equity_panel(n_tickers=4, n_days=300, seed=6)
    sig = MLSignal()
    # No fit called -> model is None -> empty prediction.
    assert sig.predict(panel).empty


def test_information_coefficient_basic():
    rng = np.random.default_rng(0)
    n = 300
    signal = pd.Series(rng.normal(0, 1, n))
    # Forward returns correlated with the signal.
    fwd = 0.5 * signal + rng.normal(0, 1, n)
    diag = information_coefficient(signal, fwd, n_bootstrap=100, seed=1)
    assert diag.information_coefficient > 0.2
    assert diag.n_obs == n
    lo, hi = diag.ic_bootstrap_ci
    assert lo <= diag.information_coefficient <= hi


def test_information_coefficient_insufficient_data():
    diag = information_coefficient(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))
    assert np.isnan(diag.information_coefficient)
