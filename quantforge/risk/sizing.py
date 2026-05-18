"""Position-sizing rules: fractional Kelly and vol targeting."""

from __future__ import annotations

import pandas as pd

from quantforge.constants import TRADING_DAYS_PER_YEAR


def fractional_kelly(
    returns: pd.Series,
    fraction: float = 0.25,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    r"""Fractional Kelly sizing.

    Mathematical Definition
    -----------------------
    Full Kelly: :math:`f^* = \mu / \sigma^2`. Fractional Kelly scales by
    ``fraction`` to reduce drawdowns (e.g. half-Kelly at ``fraction=0.5``).

    Parameters
    ----------
    returns : pd.Series
        Per-period returns (typically daily).
    fraction : float
        Fraction of full Kelly to deploy.
    periods_per_year : int

    Returns
    -------
    float
        Multiplier on the strategy's gross exposure.

    Notes
    -----
    Kelly is sensitive to estimation error. ``fraction <= 0.5`` is the norm
    in production. We default to a conservative 0.25.
    """
    r = returns.dropna()
    if len(r) < 30:
        return float("nan")
    mu = r.mean() * periods_per_year
    var = r.var(ddof=1) * periods_per_year
    if var <= 0:
        return float("nan")
    return float(fraction * mu / var)


def vol_target_scale(realized_vol_annual: float, target_vol_annual: float) -> float:
    """Multiplier that brings realized vol to a target annual level."""
    if realized_vol_annual <= 0:
        return 0.0
    return float(target_vol_annual / realized_vol_annual)


__all__ = ["fractional_kelly", "vol_target_scale"]
