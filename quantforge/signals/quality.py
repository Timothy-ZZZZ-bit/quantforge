"""A simple price-based quality factor (Piotroski-style proxy on ETF data).

Free fundamental data is unreliable for backtest-grade research. Instead we
expose a Piotroski-style *behavioral* proxy built from price information
only: low volatility, high trailing risk-adjusted return, and low drawdown.
The aggregate is z-scored cross-sectionally and emitted as an alpha score.

This is intentionally documented as a proxy; the README is explicit that the
true Piotroski F-Score requires accounting data outside our scope.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.constants import TRADING_DAYS_PER_YEAR
from quantforge.features.cross_sectional import cs_zscore
from quantforge.signals.base import Signal


class QualityFactor(Signal):
    """Price-only quality proxy.

    Combines three z-scored components: trailing-year Sharpe, negative of
    realized vol, and negative of trailing drawdown.

    Parameters
    ----------
    lookback : int
        Window in bars over which the components are evaluated.
    weights : tuple[float, float, float]
        Component weights (Sharpe, -vol, -drawdown).
    """

    name = "quality"

    def __init__(
        self,
        lookback: int = TRADING_DAYS_PER_YEAR,
        weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
    ) -> None:
        self.lookback = lookback
        self.weights = weights

    def fit(self, panel: pd.DataFrame) -> QualityFactor:
        return self

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        rows: list[tuple[str, float, float, float]] = []
        for tk, sub in panel.groupby("ticker", observed=True):
            sub = sub.sort_values("date")
            if len(sub) < self.lookback + 1:
                continue
            adj = sub["adj_close"].iloc[-self.lookback - 1 :]
            r = np.log(adj / adj.shift(1)).dropna()
            mean = r.mean() * TRADING_DAYS_PER_YEAR
            vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
            if vol <= 0:
                continue
            sharpe = float(mean / vol)
            wealth = (1.0 + r).cumprod()
            dd = float((wealth / wealth.cummax() - 1.0).min())
            rows.append((tk, sharpe, float(vol), dd))
        if not rows:
            return pd.Series(dtype=float, name=self.name)
        df = pd.DataFrame(rows, columns=["ticker", "sharpe", "vol", "dd"]).set_index("ticker")
        df["date"] = panel["date"].max()  # synthetic date column for cs_zscore
        z_sharpe = cs_zscore(df.reset_index(), "sharpe", by="date").to_numpy()
        z_negvol = -cs_zscore(df.reset_index(), "vol", by="date").to_numpy()
        z_negdd = -cs_zscore(df.reset_index(), "dd", by="date").to_numpy()
        w1, w2, w3 = self.weights
        agg = w1 * z_sharpe + w2 * z_negvol + w3 * z_negdd
        out = pd.Series(agg, index=df.index, name=self.name).fillna(0.0)
        gross = out.abs().sum()
        if gross > 0:
            out = out / gross
        return out


__all__ = ["QualityFactor"]
