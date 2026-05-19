"""Probabilistic Sharpe Ratio and Deflated Sharpe Ratio.

The PSR adjusts the t-statistic on a Sharpe estimate for finite-sample bias
introduced by the skew and excess kurtosis of the return distribution. The
DSR additionally adjusts for the implicit multiple testing introduced when
many strategies are tried during research.

References
----------
- Bailey, D.H., Lopez de Prado, M. (2012). "The Sharpe Ratio Efficient
  Frontier". *Journal of Risk*, 15(2), 3-44.
- Bailey, D.H., Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio".
  *Journal of Portfolio Management* 40, 94-107.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from quantforge.constants import TRADING_DAYS_PER_YEAR
from quantforge.metrics.performance import sharpe_ratio


def _moments(returns: pd.Series) -> tuple[float, float, float, float]:
    r = returns.dropna()
    n = len(r)
    if n < 4:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(r.mean())
    sd = float(r.std(ddof=1))
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=True, bias=False))
    return mean, sd, skew, kurt


def probabilistic_sharpe(
    returns: pd.Series,
    benchmark_sr: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    r"""Bailey-Lopez de Prado PSR against a benchmark Sharpe.

    Mathematical Definition
    -----------------------
    .. math::
        \mathrm{PSR}(SR^*) = \Phi\!\left(
            \frac{(\hat{SR} - SR^*) \sqrt{n - 1}}
                 {\sqrt{1 - \gamma_3 \hat{SR} + \frac{\gamma_4 - 1}{4} \hat{SR}^2}}
            \right),

    where :math:`\hat{SR}` is the per-period Sharpe of the sample,
    :math:`\gamma_3` is its skew, and :math:`\gamma_4` is its excess kurtosis.

    Returns
    -------
    float
        Probability the true Sharpe exceeds the benchmark.
    """
    r = returns.dropna()
    n = len(r)
    if n < 4:
        return float("nan")
    _, _, skew, kurt = _moments(r)
    # Period-level Sharpe; annualization cancels in the t-statistic.
    sr_hat = sharpe_ratio(r, periods_per_year=1)  # raw per-period
    sr_bench = benchmark_sr / np.sqrt(periods_per_year)
    numerator = (sr_hat - sr_bench) * math.sqrt(n - 1)
    denom_sq = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2
    if denom_sq <= 0 or not np.isfinite(denom_sq):
        return float("nan")
    z = numerator / math.sqrt(denom_sq)
    return float(stats.norm.cdf(z))


def deflated_sharpe(
    returns: pd.Series,
    n_trials: int,
    sr_estimates: list[float] | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    r"""Bailey-Lopez de Prado Deflated Sharpe Ratio.

    Parameters
    ----------
    returns : pd.Series
        Returns of the strategy in question.
    n_trials : int
        Number of strategy variants tested during research. This is the
        deflation factor: the more variants, the higher the multiple-testing
        bar.
    sr_estimates : list[float] | None
        If provided, the variance of these Sharpe estimates is used in the
        expected-maximum approximation. If None, a unit standard deviation
        is assumed.
    periods_per_year : int

    Notes
    -----
    DSR > 0.95 is a credible threshold for declaring a finding statistically
    significant under multiple testing. The function returns the raw DSR
    probability; callers decide on a threshold.
    """
    r = returns.dropna()
    if r.empty:
        return float("nan")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    sr_std = 1.0 if sr_estimates is None else float(np.std(sr_estimates, ddof=1))
    if not np.isfinite(sr_std) or sr_std <= 0:
        sr_std = 1.0
    # Expected maximum of N iid standard normals (Gumbel approximation).
    # E[max(N)] ~ sqrt(2 ln N) - (ln ln N + ln 4 pi) / (2 sqrt(2 ln N)).
    gamma = 0.5772156649  # Euler-Mascheroni
    z_max = (1.0 - gamma) * stats.norm.ppf(1.0 - 1.0 / n_trials) + gamma * stats.norm.ppf(
        1.0 - 1.0 / (n_trials * math.e)
    )
    sr_bench_per_period = (sr_std / np.sqrt(periods_per_year)) * z_max
    sr_bench_annual = sr_bench_per_period * np.sqrt(periods_per_year)
    return probabilistic_sharpe(
        r, benchmark_sr=float(sr_bench_annual), periods_per_year=periods_per_year
    )


__all__ = ["deflated_sharpe", "probabilistic_sharpe"]
