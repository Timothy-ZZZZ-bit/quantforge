"""Performance attribution: factor exposures with Newey-West standard errors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class FactorExposureResult:
    """Result of a regression of returns onto factor returns."""

    alpha: float
    alpha_tstat: float
    betas: pd.Series
    beta_tstats: pd.Series
    r_squared: float
    n_obs: int
    newey_west_lag: int


def _optimal_lag(n: int) -> int:
    """Newey-West optimal lag (Andrews 1991, rule of thumb)."""
    return max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def factor_exposure(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> FactorExposureResult:
    r"""Regress portfolio returns onto factor returns with Newey-West SEs.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Per-bar portfolio returns.
    factor_returns : pd.DataFrame
        Factor returns. The constant is added automatically; do not include
        it manually.

    Returns
    -------
    FactorExposureResult
        Alpha, betas, t-stats, R-squared, Newey-West lag used.
    """
    df = pd.concat([portfolio_returns.rename("r"), factor_returns], axis=1).dropna()
    if df.shape[0] < max(20, factor_returns.shape[1] + 5):
        return FactorExposureResult(
            alpha=float("nan"),
            alpha_tstat=float("nan"),
            betas=pd.Series(dtype=float),
            beta_tstats=pd.Series(dtype=float),
            r_squared=float("nan"),
            n_obs=df.shape[0],
            newey_west_lag=0,
        )
    y = df["r"].to_numpy()
    X = sm.add_constant(df[factor_returns.columns].to_numpy())
    lag = _optimal_lag(len(y))
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    params = model.params
    tvals = model.tvalues
    betas = pd.Series(params[1:], index=factor_returns.columns, name="beta")
    beta_t = pd.Series(tvals[1:], index=factor_returns.columns, name="beta_tstat")
    return FactorExposureResult(
        alpha=float(params[0]),
        alpha_tstat=float(tvals[0]),
        betas=betas,
        beta_tstats=beta_t,
        r_squared=float(model.rsquared),
        n_obs=len(y),
        newey_west_lag=int(lag),
    )


__all__ = ["FactorExposureResult", "factor_exposure"]
