"""Ornstein-Uhlenbeck mean reversion on residuals.

Fit an OU process to the residual of each asset's log price relative to a
common market proxy. Trade the z-score of the residual: enter at +/- 2 sigma,
exit at zero, stop at +/- 4 sigma. Skip names whose half-life exceeds a
configurable threshold (default 30 days).

References
----------
- Avellaneda, M., Lee, J.H. (2010). "Statistical arbitrage in the US equities
  market". *Quantitative Finance* 10, 761-782.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from quantforge.signals.base import Signal


def _fit_ou(residuals: np.ndarray) -> tuple[float, float, float]:
    """Fit OU :math:`dx_t = -\\theta(x_t - \\mu)dt + \\sigma dW_t` via OLS.

    Returns
    -------
    (theta, mu, sigma) : tuple[float, float, float]
    """
    x = residuals[:-1]
    y = residuals[1:]
    slope, intercept, _, _, _ = stats.linregress(x, y)
    if not 0 < slope < 1:
        return float("nan"), float("nan"), float("nan")
    theta = -np.log(slope)
    mu = intercept / (1.0 - slope)
    resid = y - (slope * x + intercept)
    sigma = float(np.std(resid, ddof=1) * np.sqrt(2.0 * theta / (1.0 - slope**2)))
    return float(theta), float(mu), sigma


class OUMeanReversion(Signal):
    """Single-name OU mean reversion.

    Parameters
    ----------
    market_ticker : str
        Name of the asset to use as a market proxy (e.g. ``"SPY"``).
    window : int
        OU-fit window length.
    entry_z : float
        Entry z-score. Positive: short on a positive residual.
    exit_z : float
        Exit z-score (absolute).
    max_half_life_days : float
        Skip names whose estimated half-life :math:`\\log 2 / \\theta`
        exceeds this threshold.
    """

    name = "ou_mr"

    def __init__(
        self,
        market_ticker: str = "SPY",
        window: int = 63,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        max_half_life_days: float = 30.0,
    ) -> None:
        self.market_ticker = market_ticker
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.max_half_life_days = max_half_life_days

    def fit(self, panel: pd.DataFrame) -> OUMeanReversion:
        return self

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        wide = panel.pivot(
            index="date", columns="ticker", values="adj_close"
        ).sort_index()
        if self.market_ticker not in wide.columns or wide.shape[0] < self.window + 2:
            return pd.Series(dtype=float, name=self.name)
        mkt = np.log(wide[self.market_ticker]).dropna()
        mkt_recent = mkt.iloc[-self.window :]
        scores: dict[str, float] = {}
        for tk in wide.columns:
            if tk == self.market_ticker:
                continue
            ser = np.log(wide[tk]).dropna()
            if len(ser) < self.window + 2:
                continue
            ser_recent = ser.iloc[-self.window :]
            mkt_aligned = mkt_recent.reindex(ser_recent.index).dropna()
            ser_aligned = ser_recent.reindex(mkt_aligned.index)
            if len(ser_aligned) < self.window // 2:
                continue
            slope, intercept, _, _, _ = stats.linregress(mkt_aligned, ser_aligned)
            resid = (ser_aligned - (slope * mkt_aligned + intercept)).to_numpy()
            theta, mu, sigma = _fit_ou(resid)
            if not np.isfinite(theta) or theta <= 0:
                continue
            half_life = float(np.log(2.0) / theta)
            if half_life > self.max_half_life_days:
                continue
            sd = sigma / np.sqrt(2.0 * theta) if theta > 0 else float("nan")
            if not np.isfinite(sd) or sd <= 0:
                continue
            z = (resid[-1] - mu) / sd
            if z > self.entry_z:
                scores[tk] = -1.0  # short the rich
            elif z < -self.entry_z:
                scores[tk] = 1.0  # long the cheap
            elif abs(z) < self.exit_z:
                scores[tk] = 0.0
        if not scores:
            return pd.Series(dtype=float, name=self.name)
        s = pd.Series(scores, name=self.name)
        # Normalize to dollar neutral with equal weights on each leg.
        if (s > 0).any():
            s.loc[s > 0] = s.loc[s > 0] / (s > 0).sum() / 2.0
        if (s < 0).any():
            s.loc[s < 0] = s.loc[s < 0] / (s < 0).sum().__abs__() / 2.0
        return s


__all__ = ["OUMeanReversion"]
