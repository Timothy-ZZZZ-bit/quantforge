"""Time-series momentum (Moskowitz-Ooi-Pedersen 2012).

For each asset independently:

1. Compute a trailing total return over the lookback window.
2. The signal is the sign of that return.
3. Scale by inverse realized volatility so each leg contributes the same
   ex-ante risk to the portfolio.

References
----------
Moskowitz, T., Ooi, Y.H., Pedersen, L.H. (2012). "Time Series Momentum".
*Journal of Financial Economics* 104, 228-250.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.constants import TRADING_DAYS_PER_YEAR
from quantforge.features.technical import momentum
from quantforge.signals.base import Signal


class TimeSeriesMomentum(Signal):
    """Inverse-vol scaled time-series momentum.

    Parameters
    ----------
    lookback : int
        Bars over which to compute trailing total return. 252 is the canonical
        12-month value on daily data.
    skip : int
        Number of most-recent bars to skip when computing momentum.
        21 (one trading month) is standard to avoid short-term reversal noise.
    vol_window : int
        Window for the inverse-vol scaling.
    target_vol_annual : float
        Per-asset target annualized volatility for the scaled position.
    """

    name = "tsmom"

    def __init__(
        self,
        lookback: int = TRADING_DAYS_PER_YEAR,
        skip: int = 21,
        vol_window: int = 63,
        target_vol_annual: float = 0.10,
    ) -> None:
        self.lookback = lookback
        self.skip = skip
        self.vol_window = vol_window
        self.target_vol_annual = target_vol_annual

    def fit(self, panel: pd.DataFrame) -> TimeSeriesMomentum:  # stateless
        return self

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        """Return inverse-vol-scaled signed momentum at the latest date."""
        scores: dict[str, float] = {}
        for tk, sub in panel.groupby("ticker", observed=True):
            sub = sub.sort_values("date")
            adj = sub["adj_close"]
            if len(adj) < self.lookback + self.skip + self.vol_window:
                continue
            mom_series = momentum(adj.reset_index(drop=True), self.lookback, self.skip)
            cc_returns = pd.Series(
                np.log((adj / adj.shift(1)).to_numpy()), index=adj.index
            ).dropna()
            sigma_d = cc_returns.rolling(self.vol_window).std(ddof=1).iloc[-1]
            if not np.isfinite(sigma_d) or sigma_d <= 0:
                continue
            sigma_annual = float(sigma_d * np.sqrt(TRADING_DAYS_PER_YEAR))
            mom = mom_series.iloc[-1]
            if not np.isfinite(mom):
                continue
            scaled = float(np.sign(mom) * (self.target_vol_annual / sigma_annual))
            scores[str(tk)] = scaled
        return pd.Series(scores, name="tsmom")


__all__ = ["TimeSeriesMomentum"]
