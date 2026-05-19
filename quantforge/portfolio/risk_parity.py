"""Equal-Risk-Contribution (ERC) portfolio.

Each name contributes the same marginal risk to the portfolio:

.. math::
    \\text{RC}_i(w) := w_i \\cdot (\\Sigma w)_i \\; = \\; \\frac{1}{n} \\cdot w^\\top \\Sigma w.

Implemented with Spinu's cyclic-coordinate-descent and verified against an
SLSQP-based reference solver in :func:`solve_erc_slsqp`.

References
----------
- Maillard, S., Roncalli, T., Teiletche, J. (2010). "The Properties of
  Equally Weighted Risk Contribution Portfolios". *JPM* 36, 60-70.
- Spinu, F. (2013). "An Algorithm for Computing Risk Parity Weights".
  SSRN 2297383.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from quantforge.portfolio.base import Allocator


def solve_erc_spinu(cov: np.ndarray, *, tol: float = 1e-10, max_iter: int = 5000) -> np.ndarray:
    """Cyclic coordinate descent to ERC weights (Spinu 2013).

    The algorithm updates each weight as the positive solution to the
    coordinate-wise quadratic obtained by requiring
    ``w_i * (cov @ w)_i = c`` for a common constant ``c``. Any positive
    constant works; we use ``1/n`` for numerical balance. At convergence
    the weights are rescaled to sum to 1.
    """
    n = cov.shape[0]
    w = np.ones(n) / np.sqrt(n)  # unnormalized initial guess
    c = 1.0 / n
    for _ in range(max_iter):
        w_old = w.copy()
        for i in range(n):
            a = cov[i, i]
            b = float(cov[i] @ w) - cov[i, i] * w[i]
            disc = b * b + 4.0 * a * c
            x = (-b + np.sqrt(disc)) / (2.0 * a)
            w[i] = max(x, 1e-12)
        if np.linalg.norm(w - w_old) < tol:
            break
    return w / w.sum()


def solve_erc_slsqp(cov: np.ndarray) -> np.ndarray:
    """Reference ERC solver via SLSQP minimization of risk-contribution dispersion.

    Uses a vol-inverse warm start and a relative-dispersion loss to avoid
    the stationary point at the equal-weight start when assets have
    asymmetric volatilities.
    """
    n = cov.shape[0]

    def loss(w: np.ndarray) -> float:
        rc = w * (cov @ w)
        total = rc.sum()
        if total <= 0:
            return float("inf")
        # Loss in *relative* contributions to avoid scale dependence.
        return float(((rc / total - 1.0 / n) ** 2).sum())

    cons = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]
    bnds = [(1e-9, 1.0) for _ in range(n)]
    # Inverse-vol warm start, scaled to sum to 1.
    sd = np.sqrt(np.diag(cov))
    w0 = (1.0 / sd) / (1.0 / sd).sum()
    res = minimize(
        loss,
        w0,
        bounds=bnds,
        constraints=cons,
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return res.x if res.success else w0


class EqualRiskContribution(Allocator):
    """Long-only ERC allocator.

    Parameters
    ----------
    gross_leverage : float
        Resultant weights sum-to-``gross_leverage``.
    use_alpha_sign : bool
        If True, multiply each weight by the sign of the alpha. Default
        False to preserve the pure ERC interpretation.
    """

    name = "erc"

    def __init__(self, gross_leverage: float = 1.0, use_alpha_sign: bool = False) -> None:
        self.gross_leverage = gross_leverage
        self.use_alpha_sign = use_alpha_sign

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
            cov = np.eye(n)
        else:
            rh = returns_history.reindex(columns=tickers).dropna(how="any")
            if rh.shape[0] < 30:
                cov = np.eye(len(tickers))
            else:
                cov = LedoitWolf().fit(rh.to_numpy()).covariance_
        cov = (cov + cov.T) / 2.0 + 1e-10 * np.eye(cov.shape[0])
        w = solve_erc_spinu(cov)
        if self.use_alpha_sign:
            w = w * np.sign(alphas.reindex(tickers).fillna(0.0).to_numpy())
        gross = np.sum(np.abs(w))
        if gross > 0:
            w = w * self.gross_leverage / gross
        return pd.Series(w, index=tickers, name="erc_weights")


__all__ = ["EqualRiskContribution", "solve_erc_slsqp", "solve_erc_spinu"]
