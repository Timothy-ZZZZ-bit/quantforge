"""Mean-variance optimization with Ledoit-Wolf shrinkage covariance.

The MVO problem solved here is

.. math::
    \\max_w \\; w^\\top \\mu - \\frac{\\lambda}{2} w^\\top \\Sigma w
    \\quad \\text{s.t.} \\quad |w_i| \\le c, \\; \\sum_i |w_i| \\le L.

We use a SciPy SLSQP solver to keep dependencies minimal (no cvxpy). The
covariance is the Ledoit-Wolf shrinkage estimator, which strictly dominates
the sample covariance in expected-Frobenius-loss sense at small sample
sizes.

References
----------
- Markowitz, H. (1952). "Portfolio Selection". *Journal of Finance* 7, 77-91.
- Ledoit, O., Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix".
  *Journal of Portfolio Management* 30, 110-119.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from quantforge.portfolio.base import Allocator


class MeanVariance(Allocator):
    """Mean-variance allocator.

    Parameters
    ----------
    risk_aversion : float
        Trade-off parameter :math:`\\lambda` in the objective. Higher means
        more conservative.
    max_weight : float
        Per-name upper bound.
    min_weight : float
        Per-name lower bound (can be negative for short).
    gross_leverage : float
        Cap on sum of absolute weights.
    """

    name = "mvo"

    def __init__(
        self,
        risk_aversion: float = 5.0,
        max_weight: float = 0.10,
        min_weight: float = -0.10,
        gross_leverage: float = 1.0,
    ) -> None:
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.gross_leverage = gross_leverage

    def allocate(
        self,
        alphas: pd.Series,
        returns_history: pd.DataFrame | None = None,
    ) -> pd.Series:
        tickers = alphas.dropna().index.tolist()
        if not tickers:
            return pd.Series(dtype=float)
        mu = alphas.reindex(tickers).to_numpy(dtype=float)
        if returns_history is None or returns_history.empty:
            sigma = np.eye(len(tickers))
        else:
            rh = returns_history.reindex(columns=tickers).dropna(how="any")
            if rh.shape[0] < 20:
                sigma = np.eye(len(tickers))
            else:
                lw = LedoitWolf().fit(rh.to_numpy())
                sigma = lw.covariance_

        # Ensure PSD
        sigma = (sigma + sigma.T) / 2.0
        # Tiny ridge for numerical stability.
        sigma += 1e-8 * np.eye(sigma.shape[0])

        n = len(tickers)

        def negative_utility(w: np.ndarray) -> float:
            return float(-(w @ mu) + 0.5 * self.risk_aversion * w @ sigma @ w)

        def grad(w: np.ndarray) -> np.ndarray:
            return -mu + self.risk_aversion * sigma @ w

        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]
        # Use the gross-leverage cap as an inequality.
        constraints = [
            {"type": "ineq", "fun": lambda w: self.gross_leverage - float(np.sum(np.abs(w)))}
        ]
        w0 = np.zeros(n)
        res = minimize(
            negative_utility,
            w0,
            jac=grad,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": 200, "ftol": 1e-9},
        )
        w = res.x if res.success else np.zeros(n)
        return pd.Series(w, index=tickers, name="mvo_weights")


__all__ = ["MeanVariance"]
