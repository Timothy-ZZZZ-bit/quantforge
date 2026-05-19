"""Technical features: momentum and four realized-volatility estimators.

Realized volatility estimators differ in which intra-bar information they
exploit. Yang-Zhang dominates the others under fairly mild conditions because
it is drift-free and uses open, high, low and close jointly; Garman-Klass is
nearly as efficient when overnight returns can be ignored; Parkinson uses
high-low only and is robust to opening jumps but biased downward; close-to-
close is the simplest and noisiest.

References
----------
- Parkinson, M. (1980). "The extreme value method for estimating the variance
  of the rate of return". *Journal of Business* 53, 61-65.
- Garman, M., Klass, M. (1980). "On the estimation of security price
  volatilities from historical data". *Journal of Business* 53, 67-78.
- Yang, D., Zhang, Q. (2000). "Drift-Independent Volatility Estimation Based
  on High, Low, Open and Close Prices". *Journal of Business* 73, 477-491.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

VolEstimator = Literal["close_to_close", "parkinson", "garman_klass", "yang_zhang"]


def momentum(prices: pd.Series, lookback: int, skip: int = 0) -> pd.Series:
    r"""Trailing total-return momentum.

    Mathematical Definition
    -----------------------
    :math:`\text{mom}_t(L, s) = p_{t-s} / p_{t-s-L} - 1`.

    Parameters
    ----------
    prices : pd.Series
        Adjusted close.
    lookback : int
        Lookback window in bars.
    skip : int
        Number of most-recent bars to skip (e.g. 1-month skip to avoid the
        well-documented short-term reversal).
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if skip < 0:
        raise ValueError("skip must be non-negative")
    return prices.shift(skip) / prices.shift(skip + lookback) - 1.0


def _log_series(x: pd.Series) -> pd.Series:
    """Element-wise log that mypy recognizes as returning a Series.

    ``np.log`` on a Series returns a Series at runtime, but the numpy stubs
    type the result as ``ndarray``. This wrapper restores the Series type.
    """
    return pd.Series(np.log(x.to_numpy(dtype=float)), index=x.index, name=x.name)


def _close_to_close_sd(close: pd.Series, window: int) -> pd.Series:
    r"""Close-to-close standard deviation.

    .. math::
        \hat{\sigma}_t = \sqrt{\frac{1}{n-1}\sum_{i=t-n+1}^{t} (r_i - \bar{r})^2}.
    """
    r = _log_series(close / close.shift(1))
    return r.rolling(window).std(ddof=1)


def _parkinson_sd(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    r"""Parkinson estimator.

    .. math::
        \hat{\sigma}^2 = \frac{1}{4 n \ln 2}\sum_i (\ln(h_i / l_i))^2.
    """
    if (high <= 0).any() or (low <= 0).any():
        raise ValueError("Parkinson requires strictly positive high and low")
    hl = _log_series(high / low)
    var = (hl**2).rolling(window).mean() / (4.0 * np.log(2.0))
    return pd.Series(np.sqrt(var.to_numpy()), index=var.index)


def _garman_klass_sd(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    r"""Garman-Klass estimator.

    .. math::
        \hat{\sigma}^2 = \frac{1}{n}\sum_i\Big[ \tfrac{1}{2}(\ln h_i/l_i)^2
            - (2\ln 2 - 1)(\ln c_i/o_i)^2 \Big].
    """
    if any((x <= 0).any() for x in (open_, high, low, close)):
        raise ValueError("Garman-Klass requires strictly positive prices")
    hl = _log_series(high / low) ** 2
    co = _log_series(close / open_) ** 2
    var = (0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co).rolling(window).mean()
    return pd.Series(np.sqrt(var.clip(lower=0.0).to_numpy()), index=var.index)


def _yang_zhang_sd(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    r"""Yang-Zhang estimator (drift-independent).

    .. math::
        \hat{\sigma}^2 = \sigma_o^2 + k\, \sigma_c^2 + (1-k)\, \sigma_{rs}^2,

    where :math:`\sigma_o^2` is the overnight-return variance, :math:`\sigma_c^2`
    is the open-to-close variance, :math:`\sigma_{rs}^2` is the Rogers-Satchell
    variance, and :math:`k = 0.34 / (1.34 + (n+1)/(n-1))`.
    """
    if any((x <= 0).any() for x in (open_, high, low, close)):
        raise ValueError("Yang-Zhang requires strictly positive prices")
    o_c_prev = _log_series(open_ / close.shift(1))
    c_o = _log_series(close / open_)
    rs = (_log_series(high / close) * _log_series(high / open_)) + (
        _log_series(low / close) * _log_series(low / open_)
    )

    sigma_o_sq = o_c_prev.rolling(window).var(ddof=1)
    sigma_c_sq = c_o.rolling(window).var(ddof=1)
    sigma_rs_sq = rs.rolling(window).mean()

    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = sigma_o_sq + k * sigma_c_sq + (1.0 - k) * sigma_rs_sq
    return pd.Series(np.sqrt(var.clip(lower=0.0).to_numpy()), index=var.index)


def realized_vol(
    df: pd.DataFrame,
    window: int = 21,
    method: VolEstimator = "close_to_close",
) -> pd.Series:
    """Realized volatility via the requested estimator.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``adj_close``. For non-close-to-close methods, also
        requires ``open``, ``high``, ``low``.
    window : int
        Rolling window in bars.
    method : str
        One of ``"close_to_close"``, ``"parkinson"``, ``"garman_klass"``,
        ``"yang_zhang"``.

    Returns
    -------
    pd.Series
        Rolling standard deviation of log returns under the chosen estimator.

    Notes
    -----
    The output is a per-bar standard deviation. Multiply by
    :math:`\\sqrt{252}` (or import ``constants.VOL_ANNUALIZATION_FACTOR``)
    for annualization.
    """
    if method == "close_to_close":
        return _close_to_close_sd(df["adj_close"], window)
    if method == "parkinson":
        return _parkinson_sd(df["high"], df["low"], window)
    if method == "garman_klass":
        return _garman_klass_sd(df["open"], df["high"], df["low"], df["adj_close"], window)
    if method == "yang_zhang":
        return _yang_zhang_sd(df["open"], df["high"], df["low"], df["adj_close"], window)
    raise ValueError(f"unknown method: {method!r}")


__all__ = ["VolEstimator", "momentum", "realized_vol"]
