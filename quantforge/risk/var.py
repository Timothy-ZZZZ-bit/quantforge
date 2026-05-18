"""Value-at-risk estimators with bootstrap confidence intervals.

Three flavors:
- ``parametric_var``: assumes normal returns; uses sample mean and SD.
- ``historical_var``: empirical quantile of past returns.
- ``cornish_fisher_var``: normal quantile adjusted for sample skew/kurtosis.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from quantforge.constants import DEFAULT_BOOTSTRAP_REPLICATES, DEFAULT_CONFIDENCE_LEVEL


@dataclass(frozen=True)
class VaRResult:
    """Point estimate plus bootstrap CI for a VaR figure."""

    point: float
    ci_low: float
    ci_high: float
    confidence: float
    method: str


def _bootstrap_ci(
    values: npt.NDArray[np.float64],
    statistic: Callable[[npt.NDArray[np.float64]], float],
    n_boot: int,
    seed: int,
    ci_level: float,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    samples = rng.choice(values, size=(n_boot, n), replace=True)
    stats_arr = np.array([statistic(samples[i]) for i in range(n_boot)])
    lo = float(np.quantile(stats_arr, (1.0 - ci_level) / 2.0))
    hi = float(np.quantile(stats_arr, 1.0 - (1.0 - ci_level) / 2.0))
    return lo, hi


def parametric_var(
    returns: pd.Series,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    n_boot: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = 0,
) -> VaRResult:
    r"""Normal-distribution VaR.

    :math:`\text{VaR}_\alpha = -(\mu + \sigma z_\alpha)` for the
    :math:`(1 - \alpha)` lower tail.
    """
    r = returns.dropna().to_numpy()
    if len(r) < 30:
        return VaRResult(
            float("nan"), float("nan"), float("nan"), confidence, "parametric"
        )
    z = stats.norm.ppf(1.0 - confidence)

    def stat(s: np.ndarray) -> float:
        return float(-(s.mean() + s.std(ddof=1) * z))

    point = stat(r)
    lo, hi = _bootstrap_ci(r, stat, n_boot, seed, 0.95)
    return VaRResult(point, lo, hi, confidence, "parametric")


def historical_var(
    returns: pd.Series,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    n_boot: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = 0,
) -> VaRResult:
    """Empirical-quantile VaR."""
    r = returns.dropna().to_numpy()
    if len(r) < 30:
        return VaRResult(
            float("nan"), float("nan"), float("nan"), confidence, "historical"
        )
    q = 1.0 - confidence

    def stat(s: np.ndarray) -> float:
        return float(-np.quantile(s, q))

    point = stat(r)
    lo, hi = _bootstrap_ci(r, stat, n_boot, seed, 0.95)
    return VaRResult(point, lo, hi, confidence, "historical")


def cornish_fisher_var(
    returns: pd.Series,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    n_boot: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = 0,
) -> VaRResult:
    r"""Cornish-Fisher adjusted VaR.

    Adjusts the normal quantile :math:`z` for sample skew :math:`S` and
    excess kurtosis :math:`K`:

    .. math::
        z^* = z + \frac{1}{6}(z^2 - 1)S + \frac{1}{24}(z^3 - 3z) K
            - \frac{1}{36}(2 z^3 - 5 z) S^2.
    """
    r = returns.dropna().to_numpy()
    if len(r) < 30:
        return VaRResult(
            float("nan"), float("nan"), float("nan"), confidence, "cornish_fisher"
        )
    z = stats.norm.ppf(1.0 - confidence)

    def stat(s: np.ndarray) -> float:
        m = float(np.mean(s))
        sd = float(np.std(s, ddof=1))
        S = float(stats.skew(s, bias=False))
        K = float(stats.kurtosis(s, fisher=True, bias=False))
        zs = (
            z
            + (1.0 / 6.0) * (z**2 - 1.0) * S
            + (1.0 / 24.0) * (z**3 - 3.0 * z) * K
            - (1.0 / 36.0) * (2.0 * z**3 - 5.0 * z) * (S**2)
        )
        return float(-(m + sd * zs))

    point = stat(r)
    lo, hi = _bootstrap_ci(r, stat, n_boot, seed, 0.95)
    return VaRResult(point, lo, hi, confidence, "cornish_fisher")


__all__ = [
    "VaRResult",
    "cornish_fisher_var",
    "historical_var",
    "parametric_var",
]
