"""Expected shortfall (CVaR)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.constants import DEFAULT_CONFIDENCE_LEVEL


def expected_shortfall(
    returns: pd.Series, confidence: float = DEFAULT_CONFIDENCE_LEVEL
) -> float:
    r"""Empirical expected shortfall at confidence :math:`\\alpha`.

    .. math::
        \text{ES}_\alpha = -\mathbb{E}[r \mid r \le -\text{VaR}_\alpha].
    """
    r = returns.dropna()
    if r.empty:
        return float("nan")
    q = 1.0 - confidence
    threshold = np.quantile(r, q)
    tail = r[r <= threshold]
    if tail.empty:
        return float("nan")
    return float(-tail.mean())


__all__ = ["expected_shortfall"]
