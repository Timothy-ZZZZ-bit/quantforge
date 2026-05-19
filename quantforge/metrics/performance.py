"""Headline performance metrics.

All functions take per-bar simple or log returns and a periods-per-year
constant. The convention throughout is daily returns and 252.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.constants import TRADING_DAYS_PER_YEAR


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Geometric annualized return.

    :math:`(1 + r_d)^N - 1` where :math:`r_d` is the realized geometric mean.
    """
    r = returns.dropna()
    if r.empty:
        return float("nan")
    total = float(np.prod(1.0 + r.to_numpy(dtype=float)))
    n_years = len(r) / periods_per_year
    if n_years <= 0:
        return float("nan")
    return total ** (1.0 / n_years) - 1.0


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Annualized volatility, computed with ddof=1."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    rf_per_period: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio.

    .. math::
        SR = \\frac{\\mathbb{E}[r] - r_f}{\\sigma(r)} \\cdot \\sqrt{N}.
    """
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    excess = r - rf_per_period
    denom = excess.std(ddof=1)
    # Guard against numerical noise on a constant series: any std smaller than
    # the relative-precision of the mean is effectively zero.
    mean_abs = float(np.abs(excess.mean()))
    if denom < max(1e-15, mean_abs * 1e-12):
        return float("nan")
    return float(excess.mean() / denom * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    rf_per_period: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Sortino with downside deviation (negative-only)."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    excess = r - rf_per_period
    downside = excess.clip(upper=0.0)
    denom = np.sqrt((downside**2).mean())
    if denom == 0:
        return float("nan")
    return float(excess.mean() / denom * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum compounded drawdown."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    wealth = (1.0 + r).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    return float(dd.min())


def calmar(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized return divided by absolute max drawdown."""
    ar = annualized_return(returns, periods_per_year)
    mdd = max_drawdown(returns)
    if not np.isfinite(mdd) or mdd == 0:
        return float("nan")
    return float(ar / abs(mdd))


def hit_rate(returns: pd.Series) -> float:
    """Fraction of positive return bars."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    return float((r > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    """Sum of positive returns over absolute sum of negative returns."""
    r = returns.dropna()
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    if neg == 0:
        return float("nan")
    return float(pos / neg)


def omega(returns: pd.Series, threshold: float = 0.0) -> float:
    """Omega ratio above ``threshold``."""
    r = returns.dropna() - threshold
    if r.empty:
        return float("nan")
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    if neg == 0:
        return float("nan")
    return float(pos / neg)


def time_underwater(returns: pd.Series) -> int:
    """Longest stretch of consecutive bars below the previous peak."""
    r = returns.dropna()
    if r.empty:
        return 0
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    underwater = (wealth < peak).astype(int).to_numpy()
    longest = 0
    current = 0
    for u in underwater:
        if u:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


__all__ = [
    "annualized_return",
    "annualized_volatility",
    "calmar",
    "hit_rate",
    "max_drawdown",
    "omega",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
    "time_underwater",
]
