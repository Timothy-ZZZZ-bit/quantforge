"""Cross-sectional feature operators.

A common pattern in factor research is to take a raw measure (e.g., a
12-month return) and convert it into a cross-sectional signal by ranking,
z-scoring, or sector-neutralizing within each rebalance date. These three
operations are implemented here in pandas-vectorized form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cs_rank(panel: pd.DataFrame, value_col: str, by: str = "date") -> pd.Series:
    """Cross-sectional rank, scaled to ``[-0.5, +0.5]`` within each date.

    Parameters
    ----------
    panel : pd.DataFrame
        Long-format panel with at least ``[by, value_col]``.
    value_col : str
        Column to rank.
    by : str
        Grouping column (default ``"date"``).

    Returns
    -------
    pd.Series
        Index-aligned to ``panel`` with rank in ``[-0.5, +0.5]``. NaN inputs
        remain NaN.
    """
    grouped = panel.groupby(by, observed=True)[value_col]
    ranks = grouped.rank(method="average", pct=True) - 0.5
    return ranks


def cs_zscore(panel: pd.DataFrame, value_col: str, by: str = "date") -> pd.Series:
    """Cross-sectional z-score (winsorize-friendly base operation).

    Notes
    -----
    Standard deviation is computed with ``ddof=0`` so that the result is
    well-defined for cross sections with as few as two assets.
    """
    grouped = panel.groupby(by, observed=True)[value_col]
    mean = grouped.transform("mean")
    std = grouped.transform(lambda s: s.std(ddof=0))
    z = (panel[value_col] - mean) / std.replace(0.0, np.nan)
    return z


def sector_neutralize(
    panel: pd.DataFrame,
    value_col: str,
    sector_col: str,
    by: str = "date",
) -> pd.Series:
    r"""Sector neutralization by within-sector demeaning.

    Mathematical Definition
    -----------------------
    For each date :math:`d` and asset :math:`i` in sector :math:`s(i)`,

    .. math::
        \tilde{x}_{d,i} = x_{d,i} - \frac{1}{|\{j: s(j) = s(i)\}|}
            \sum_{j: s(j) = s(i)} x_{d,j}.

    Equivalent to regressing :math:`x` onto sector dummies and returning the
    residual, but implemented as a fast group transform.
    """
    if sector_col not in panel.columns:
        raise KeyError(f"sector column not found: {sector_col!r}")
    grouped = panel.groupby([by, sector_col], observed=True)[value_col]
    sector_mean = grouped.transform("mean")
    return panel[value_col] - sector_mean


__all__ = ["cs_rank", "cs_zscore", "sector_neutralize"]
