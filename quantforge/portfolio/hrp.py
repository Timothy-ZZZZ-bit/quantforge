"""Hierarchical Risk Parity (Lopez de Prado 2016).

Three-step algorithm:

1. **Tree clustering**: build a single-linkage hierarchical cluster tree on
   the correlation distance :math:`d_{ij} = \\sqrt{(1 - \\rho_{ij}) / 2}`.
2. **Quasi-diagonalization**: reorder the covariance matrix according to the
   linkage to place correlated assets adjacent.
3. **Recursive bisection**: split the ordered list, allocate inverse-variance
   weights between halves, recurse.

This implementation reproduces the toy example from the paper to six
decimal places (see ``tests/unit/test_hrp.py``).

References
----------
Lopez de Prado, M. (2016). "Building Diversified Portfolios that Outperform
Out-of-Sample". *Journal of Portfolio Management* 42, 59-69.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from quantforge.portfolio.base import Allocator


def _correl_distance(corr: np.ndarray) -> np.ndarray:
    """Lopez de Prado correlation distance."""
    return np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))


def _quasi_diag(link: np.ndarray) -> list[int]:
    """Lopez de Prado quasi-diagonalization."""
    link = link.astype(int)
    n_clusters = link.shape[0] + 1
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = n_clusters
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.to_numpy() - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _ivp(cov: np.ndarray) -> np.ndarray:
    """Inverse-variance portfolio (within a cluster)."""
    ivp = 1.0 / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


def _recursive_bisection(cov: np.ndarray, sort_ix: list[int]) -> np.ndarray:
    """HRP recursive bisection step."""
    w = np.ones(len(sort_ix))
    clusters = [sort_ix]
    while clusters:
        clusters = [c[i:j] for c in clusters for i, j in (
            (0, len(c) // 2), (len(c) // 2, len(c))
        ) if len(c) > 1 and len(c[i:j]) > 0]
        # Iterate in pairs.
        for i in range(0, len(clusters), 2):
            if i + 1 >= len(clusters):
                break
            c0, c1 = clusters[i], clusters[i + 1]
            cov0 = cov[np.ix_(c0, c0)]
            cov1 = cov[np.ix_(c1, c1)]
            w0 = _ivp(cov0)
            w1 = _ivp(cov1)
            var0 = float(w0 @ cov0 @ w0)
            var1 = float(w1 @ cov1 @ w1)
            alpha = 1.0 - var0 / (var0 + var1)
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha
    return w


def hrp_weights_from_cov(cov: np.ndarray) -> np.ndarray:
    """Return HRP weights from a covariance matrix.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix. Must be square, symmetric, and positive-definite.

    Returns
    -------
    np.ndarray
        Weights in input ordering, summing to 1.
    """
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be square")
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    np.fill_diagonal(corr, 1.0)
    dist = _correl_distance(corr)
    link = linkage(squareform(dist, checks=False), method="single")
    sort_ix = _quasi_diag(link)
    w_sorted = _recursive_bisection(cov, sort_ix)
    out = np.zeros_like(w_sorted)
    out[sort_ix] = w_sorted[sort_ix]
    # The bisection already operates in-place by index, so 'w_sorted' is in
    # input order. The block above is defensive.
    return w_sorted


class HRPAllocator(Allocator):
    """HRP allocator wrapping :func:`hrp_weights_from_cov`."""

    name = "hrp"

    def __init__(self, gross_leverage: float = 1.0) -> None:
        self.gross_leverage = gross_leverage

    def allocate(
        self,
        alphas: pd.Series,
        returns_history: pd.DataFrame | None = None,
    ) -> pd.Series:
        tickers = alphas.dropna().index.tolist()
        if returns_history is None or returns_history.empty:
            n = len(tickers)
            if n == 0:
                return pd.Series(dtype=float)
            return pd.Series(self.gross_leverage / n, index=tickers)
        rh = returns_history.reindex(columns=tickers).dropna(how="any")
        if rh.shape[0] < 30 or rh.shape[1] < 2:
            n = len(tickers)
            if n == 0:
                return pd.Series(dtype=float)
            return pd.Series(self.gross_leverage / n, index=tickers)
        cov = rh.cov().to_numpy()
        cov = (cov + cov.T) / 2.0 + 1e-10 * np.eye(cov.shape[0])
        w = hrp_weights_from_cov(cov)
        return pd.Series(w * self.gross_leverage, index=tickers, name="hrp_weights")


__all__ = ["HRPAllocator", "hrp_weights_from_cov"]
