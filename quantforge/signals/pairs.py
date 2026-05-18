"""Engle-Granger pairs trading on the cointegration residual.

Steps:

1. For each pair (i, j) in the candidate set, run an OLS of log prices:
   :math:`\\log p^i_t = \\alpha + \\beta \\log p^j_t + \\epsilon_t`.
2. ADF-test the residual; retain pairs with p < 0.05.
3. Fit OU to the residual to estimate the half-life and target sigma.
4. Trade the z-score of the residual: long the residual at -2, short at +2,
   exit at 0. Skip pairs with half-life > 30 trading days.

References
----------
- Engle, R.F., Granger, C.W.J. (1987). "Co-integration and error correction".
  *Econometrica* 55, 251-276.
- Avellaneda, M., Lee, J.H. (2010). "Statistical arbitrage in the US equities
  market". *Quantitative Finance* 10, 761-782.
"""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from quantforge.signals.base import Signal


def _half_life(residuals: np.ndarray) -> float:
    """OLS-estimated OU half-life."""
    x = residuals[:-1]
    y = np.diff(residuals)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    if slope >= 0 or not np.isfinite(slope):
        return float("inf")
    return float(-np.log(2.0) / slope)


def _adf_pvalue(residuals: np.ndarray) -> float:
    if len(residuals) < 30:
        return 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(adfuller(residuals, autolag="AIC")[1])
        except Exception:
            return 1.0


class PairsTrade(Signal):
    """Engle-Granger pairs trade on a candidate set of pairs.

    Parameters
    ----------
    candidate_pairs : list[tuple[str, str]] | None
        Explicit pair list. If None, all unique pairs in the universe are
        considered (use with care; combinatorial explosion).
    window : int
        Window for ADF and OU fit.
    entry_z : float
    exit_z : float
    stop_z : float
    max_half_life_days : float
    adf_pvalue_threshold : float
    """

    name = "pairs"

    def __init__(
        self,
        candidate_pairs: list[tuple[str, str]] | None = None,
        window: int = 126,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
        max_half_life_days: float = 30.0,
        adf_pvalue_threshold: float = 0.05,
    ) -> None:
        self.candidate_pairs = candidate_pairs
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.max_half_life_days = max_half_life_days
        self.adf_pvalue_threshold = adf_pvalue_threshold

    def fit(self, panel: pd.DataFrame) -> PairsTrade:
        return self

    def _pairs_to_consider(self, tickers: list[str]) -> list[tuple[str, str]]:
        if self.candidate_pairs is not None:
            return self.candidate_pairs
        return list(combinations(sorted(tickers), 2))

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        wide = panel.pivot(
            index="date", columns="ticker", values="adj_close"
        ).sort_index()
        if wide.shape[0] < self.window + 2:
            return pd.Series(dtype=float, name=self.name)
        log_wide = pd.DataFrame(
            np.log(wide.to_numpy()), index=wide.index, columns=wide.columns
        )
        recent = log_wide.iloc[-self.window :]
        scores: dict[str, float] = {}
        pairs = self._pairs_to_consider(list(wide.columns))
        for a, b in pairs:
            if a not in recent.columns or b not in recent.columns:
                continue
            sub = recent[[a, b]].dropna()
            if len(sub) < self.window // 2:
                continue
            slope, intercept, _, _, _ = stats.linregress(sub[b], sub[a])
            resid = (sub[a] - (slope * sub[b] + intercept)).to_numpy()
            if _adf_pvalue(resid) > self.adf_pvalue_threshold:
                continue
            hl = _half_life(resid)
            if not np.isfinite(hl) or hl > self.max_half_life_days:
                continue
            sd = resid.std(ddof=1)
            if sd <= 0:
                continue
            z = float((resid[-1] - resid.mean()) / sd)
            if z > self.entry_z and abs(z) < self.stop_z:
                scores[a] = scores.get(a, 0.0) - 0.5
                scores[b] = scores.get(b, 0.0) + 0.5 * float(slope)
            elif z < -self.entry_z and abs(z) < self.stop_z:
                scores[a] = scores.get(a, 0.0) + 0.5
                scores[b] = scores.get(b, 0.0) - 0.5 * float(slope)
        if not scores:
            return pd.Series(dtype=float, name=self.name)
        out = pd.Series(scores, name=self.name)
        gross = out.abs().sum()
        if gross > 0:
            out = out / gross
        return out


__all__ = ["PairsTrade"]
