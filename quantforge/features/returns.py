"""Return transforms: log, simple, and fixed-width-window fractional differencing.

Fractional differencing preserves long memory while reducing the order of
integration to a level at which standard tests of stationarity (ADF) pass.
This matters because most ML signals require approximately stationary
inputs while still containing the price information that the model is
expected to exploit.

References
----------
- Hosking, J.R.M. (1981). "Fractional Differencing". *Biometrika* 68, 165-176.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, Ch. 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.constants import EPS


def log_returns(prices: pd.Series, fill_na: bool = False) -> pd.Series:
    r"""Compute one-step log returns of a price series.

    Mathematical Definition
    -----------------------
    :math:`r_t = \log(p_t / p_{t-1})`.

    Parameters
    ----------
    prices : pd.Series
        Strictly positive price series indexed by date.
    fill_na : bool
        If True, drop the leading NaN.

    Returns
    -------
    pd.Series
        Log returns aligned to ``prices.index``.
    """
    if (prices <= 0).any():
        raise ValueError("log_returns requires strictly positive prices")
    r = pd.Series(
        np.log((prices / prices.shift(1)).to_numpy()),
        index=prices.index,
        name=prices.name,
    )
    return r.dropna() if fill_na else r


def simple_returns(prices: pd.Series, fill_na: bool = False) -> pd.Series:
    r"""Compute one-step simple returns.

    :math:`r_t = p_t / p_{t-1} - 1`.
    """
    r = prices / prices.shift(1) - 1.0
    return r.dropna() if fill_na else r


def cumulative_returns(returns: pd.Series, compounding: str = "geometric") -> pd.Series:
    """Wealth-relative cumulative return.

    Parameters
    ----------
    returns : pd.Series
        Per-period simple or log returns.
    compounding : {"geometric", "additive"}
        Geometric (default) assumes simple returns and yields
        ``(1 + r).cumprod() - 1``. Additive assumes log returns and yields
        ``r.cumsum()``.
    """
    if compounding == "geometric":
        return (1.0 + returns).cumprod() - 1.0
    if compounding == "additive":
        return returns.cumsum()
    raise ValueError(f"unknown compounding mode: {compounding!r}")


def _ffd_weights(d: float, thresh: float) -> np.ndarray:
    r"""Compute fixed-width-window FFD weights.

    The weight at lag :math:`k` is

    .. math::
        \omega_k = (-1)^k \binom{d}{k} = \omega_{k-1} \cdot \frac{-(d - k + 1)}{k}.

    The window is truncated at the smallest :math:`k` such that
    :math:`|\omega_k| < \text{thresh}`.
    """
    weights: list[float] = [1.0]
    k = 1
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        weights.append(w_k)
        k += 1
        if k > 10_000:  # safety cap
            break
    return np.array(weights[::-1])  # apply most recent at the right


def frac_diff_ffd(series: pd.Series, d: float, thresh: float = 1e-4) -> pd.Series:
    r"""Fixed-width-window fractional differencing.

    Parameters
    ----------
    series : pd.Series
        Real-valued series, typically a log-price.
    d : float
        Differencing order in [0, 1]. ``d=0`` is identity; ``d=1`` is first
        differencing.
    thresh : float
        Weight truncation threshold. Smaller values widen the window.

    Returns
    -------
    pd.Series
        Fractionally differenced series aligned to ``series.index``. The
        first ``len(weights) - 1`` entries are NaN.

    Mathematical Definition
    -----------------------
    .. math::
        (\nabla^d X)_t = \sum_{k=0}^{K} \omega_k X_{t-k}
        \quad \text{with} \quad
        \omega_k = (-1)^k \binom{d}{k}.

    References
    ----------
    Lopez de Prado AFML Ch. 5, Algorithm 5.3.
    """
    if not 0.0 <= d <= 1.0:
        raise ValueError("d must be in [0, 1]")
    weights = _ffd_weights(d, thresh)
    width = len(weights)
    x = series.to_numpy(dtype=float)
    out = np.full_like(x, np.nan)
    if len(x) >= width:
        # Vectorize with a sliding window inner product.
        win = np.lib.stride_tricks.sliding_window_view(x, window_shape=width)
        out[width - 1 :] = win @ weights
    return pd.Series(out, index=series.index, name=series.name)


def annualize_volatility(daily_vol: float, periods_per_year: int = 252) -> float:
    """Annualize a daily volatility estimate."""
    if daily_vol < 0:
        raise ValueError("daily_vol must be non-negative")
    return float(daily_vol * np.sqrt(periods_per_year))


def reconstruct_prices_from_log_returns(log_rets: pd.Series, base: float = 1.0) -> pd.Series:
    """Reconstruct a price path from log returns (used in tests)."""
    if abs(base) < EPS:
        raise ValueError("base must be non-zero")
    return base * np.exp(log_rets.cumsum())


__all__ = [
    "annualize_volatility",
    "cumulative_returns",
    "frac_diff_ffd",
    "log_returns",
    "reconstruct_prices_from_log_returns",
    "simple_returns",
]
