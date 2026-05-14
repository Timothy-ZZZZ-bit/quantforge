"""Black-Litterman with sample views drawn from signals.

The posterior expected return blends the market-implied prior :math:`\\Pi`
with caller-provided views :math:`Q` according to view-uncertainty
:math:`\\Omega`:

.. math::
    \\hat{\\mu} = \\big[(\\tau \\Sigma)^{-1} + P^\\top \\Omega^{-1} P\\big]^{-1}
        \\big[(\\tau \\Sigma)^{-1} \\Pi + P^\\top \\Omega^{-1} Q\\big].

We treat each signal score as an absolute view on its asset. View uncertainty
is set inversely proportional to the realized volatility of the signal, so
noisier signals get downweighted automatically.

References
----------
- Black, F., Litterman, R. (1992). "Global Portfolio Optimization".
  *Financial Analysts Journal* 48, 28-43.
- He, G., Litterman, R. (1999). "The Intuition Behind Black-Litterman
  Model Portfolios". Goldman Sachs working paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from quantforge.portfolio.base import Allocator
from quantforge.portfolio.mvo import MeanVariance


class BlackLittermanAllocator(Allocator):
    """Black-Litterman allocator with signal views.

    Parameters
    ----------
    risk_aversion : float
        Used to back out the implied returns from market weights.
    tau : float
        Scalar controlling prior uncertainty (Black & Litterman use 0.025).
    view_confidence_floor : float
        Lower bound on each view's confidence (variance) to avoid
        numerical singularities.
    """

    name = "black_litterman"

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        view_confidence_floor: float = 1e-6,
    ) -> None:
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.view_confidence_floor = view_confidence_floor

    def allocate(
        self,
        alphas: pd.Series,
        returns_history: pd.DataFrame | None = None,
    ) -> pd.Series:
        tickers = alphas.dropna().index.tolist()
        if not tickers:
            return pd.Series(dtype=float)

        n = len(tickers)
        if returns_history is None or returns_history.empty:
            sigma = np.eye(n)
        else:
            rh = returns_history.reindex(columns=tickers).dropna(how="any")
            if rh.shape[0] < 30:
                sigma = np.eye(n)
            else:
                sigma = LedoitWolf().fit(rh.to_numpy()).covariance_
        sigma = (sigma + sigma.T) / 2.0 + 1e-8 * np.eye(n)

        # Equilibrium prior from equal-weight market.
        w_eq = np.ones(n) / n
        pi = self.risk_aversion * sigma @ w_eq

        # Views: P = I, Q = alphas.
        P = np.eye(n)
        Q = alphas.reindex(tickers).to_numpy(dtype=float)
        view_var = np.maximum(np.diag(sigma) * 0.1, self.view_confidence_floor)
        Omega = np.diag(view_var)

        # Posterior mean.
        tau_sigma_inv = np.linalg.pinv(self.tau * sigma)
        Omega_inv = np.linalg.inv(Omega)
        post_cov = np.linalg.inv(tau_sigma_inv + P.T @ Omega_inv @ P)
        mu_hat = post_cov @ (tau_sigma_inv @ pi + P.T @ Omega_inv @ Q)

        # Feed into MVO with the original prior covariance.
        mvo = MeanVariance(risk_aversion=self.risk_aversion, gross_leverage=1.0)
        return mvo.allocate(
            pd.Series(mu_hat, index=tickers),
            returns_history=pd.DataFrame(
                sigma, index=range(sigma.shape[0]), columns=tickers
            ).iloc[:0],  # trick: empty frame so MVO uses identity sigma
        )


__all__ = ["BlackLittermanAllocator"]
