"""Fama-French factor regression helpers."""

from __future__ import annotations

import pandas as pd

from quantforge.metrics.attribution import FactorExposureResult, factor_exposure


def fama_french_betas(
    portfolio_returns: pd.Series,
    ff_factors: pd.DataFrame,
    factors: tuple[str, ...] = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"),
) -> FactorExposureResult:
    """Regress portfolio returns on the chosen Fama-French + MOM factor set.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns indexed by date.
    ff_factors : pd.DataFrame
        Output of :func:`quantforge.data.load_fama_french` with a ``date``
        column and factor columns. The ``RF`` column, if present, is used to
        compute the excess portfolio return.
    factors : tuple[str, ...]
        Names of factors to include from ``ff_factors`` columns.
    """
    if "date" in ff_factors.columns:
        ff = ff_factors.set_index("date")
    else:
        ff = ff_factors
    cols = [c for c in factors if c in ff.columns]
    rf = ff["RF"] if "RF" in ff.columns else pd.Series(0.0, index=ff.index)
    excess = portfolio_returns.subtract(rf.reindex(portfolio_returns.index).fillna(0.0))
    return factor_exposure(excess, ff[cols])


__all__ = ["fama_french_betas"]
