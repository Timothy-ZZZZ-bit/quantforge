"""Coverage for the OU mean-reversion signal trading logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.signals.mean_reversion import OUMeanReversion, _fit_ou


def _market_plus_meanrev_panel(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n)
    log_mkt = np.cumsum(rng.normal(0.0003, 0.01, n)) + np.log(400.0)
    frames = []
    # SPY market proxy.
    frames.append(_frame("SPY", np.exp(log_mkt), dates))
    # Names whose residual to the market is mean-reverting.
    for k in range(4):
        resid = np.zeros(n)
        for t in range(1, n):
            resid[t] = 0.8 * resid[t - 1] + rng.normal(0, 0.02)
        beta = 0.8 + 0.1 * k
        log_p = beta * log_mkt + resid + np.log(50.0 + 10.0 * k)
        frames.append(_frame(f"NM{k}", np.exp(log_p), dates))
    return pd.concat(frames, ignore_index=True)


def _frame(ticker: str, prices: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "adj_close": prices,
            "volume": 1_000_000.0,
        }
    )


def test_ou_mean_reversion_produces_signal():
    panel = _market_plus_meanrev_panel()
    sig = OUMeanReversion(market_ticker="SPY", window=120, entry_z=0.5, max_half_life_days=60.0)
    sig.fit(panel)
    out = sig.predict(panel)
    assert isinstance(out, pd.Series)


def test_ou_short_history_returns_empty():
    panel = _market_plus_meanrev_panel(n=40)
    sig = OUMeanReversion(market_ticker="SPY", window=120)
    assert sig.predict(panel).empty


def test_fit_ou_recovers_mean_reverting_params():
    rng = np.random.default_rng(7)
    n = 2000
    x = np.zeros(n)
    mu_true = 0.0
    for t in range(1, n):
        x[t] = mu_true + 0.9 * (x[t - 1] - mu_true) + rng.normal(0, 0.1)
    theta, mu, sigma = _fit_ou(x)
    assert np.isfinite(theta) and theta > 0
    assert abs(mu - mu_true) < 0.2
    assert sigma > 0


def test_fit_ou_non_mean_reverting_returns_nan():
    rw = np.cumsum(np.ones(200))
    theta, mu, sigma = _fit_ou(rw)
    assert np.isnan(theta)
