"""A simple ETF-based carry proxy.

True FX/futures carry requires forward/futures curves. For a free-data
backtest we use a 12-1 momentum proxy on fixed-income vs equity ETFs as a
crude stand-in for term-premium exposure. The README is explicit that this
is a proxy, not a true carry signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.features.technical import momentum
from quantforge.signals.base import Signal


class ETFCarryProxy(Signal):
    """Carry proxy: long fixed income when its 12m momentum exceeds equity's."""

    name = "carry_proxy"

    def __init__(self, lookback: int = 252, skip: int = 21) -> None:
        self.lookback = lookback
        self.skip = skip

    def fit(self, panel: pd.DataFrame) -> ETFCarryProxy:
        return self

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        scores: dict[str, float] = {}
        for tk, sub in panel.groupby("ticker", observed=True):
            sub = sub.sort_values("date")
            if len(sub) < self.lookback + self.skip + 1:
                continue
            adj = sub["adj_close"].reset_index(drop=True)
            mom = momentum(adj, self.lookback, self.skip).iloc[-1]
            if np.isfinite(mom):
                scores[tk] = float(np.tanh(mom))
        if not scores:
            return pd.Series(dtype=float, name=self.name)
        s = pd.Series(scores, name=self.name)
        gross = s.abs().sum()
        if gross > 0:
            s = s / gross
        return s


__all__ = ["ETFCarryProxy"]
