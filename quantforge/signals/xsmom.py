"""Cross-sectional momentum (Jegadeesh-Titman 1993).

Rank the universe by trailing (12, 1) total return and form a long-short
portfolio of the top decile against the bottom decile. Returns are equal-
weighted within each leg.

References
----------
Jegadeesh, N., Titman, S. (1993). "Returns to Buying Winners and Selling
Losers: Implications for Stock Market Efficiency". *Journal of Finance*
48, 65-91.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.features.technical import momentum
from quantforge.signals.base import Signal


class CrossSectionalMomentum(Signal):
    """Long-short decile cross-sectional momentum.

    Parameters
    ----------
    lookback : int
        Lookback in bars.
    skip : int
        Recency skip in bars.
    decile : float
        Quantile cutoff for each leg (0.10 means top/bottom 10%).
    """

    name = "xsmom"

    def __init__(self, lookback: int = 252, skip: int = 21, decile: float = 0.10) -> None:
        if not 0 < decile < 0.5:
            raise ValueError("decile must be in (0, 0.5)")
        self.lookback = lookback
        self.skip = skip
        self.decile = decile

    def fit(self, panel: pd.DataFrame) -> CrossSectionalMomentum:  # stateless
        return self

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        """Long top decile, short bottom decile, equal weights within."""
        latest_date = panel["date"].max()
        moms: dict[str, float] = {}
        for tk, sub in panel.groupby("ticker", observed=True):
            sub = sub.sort_values("date")
            if len(sub) < self.lookback + self.skip + 1:
                continue
            adj = sub["adj_close"].reset_index(drop=True)
            mom = momentum(adj, self.lookback, self.skip).iloc[-1]
            if np.isfinite(mom):
                moms[str(tk)] = float(mom)
        if not moms:
            return pd.Series(dtype=float, name="xsmom")
        s = pd.Series(moms, name="xsmom")
        n = len(s)
        top_n = max(1, int(np.floor(self.decile * n)))
        ranked = s.rank(method="first")
        longs = ranked > (n - top_n)
        shorts = ranked <= top_n
        out = pd.Series(0.0, index=s.index, name="xsmom")
        if longs.any():
            out.loc[longs] = 1.0 / longs.sum()
        if shorts.any():
            out.loc[shorts] = -1.0 / shorts.sum()
        # Use latest_date for traceability only.
        _ = latest_date
        return out


__all__ = ["CrossSectionalMomentum"]
