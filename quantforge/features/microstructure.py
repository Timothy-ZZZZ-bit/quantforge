"""Daily-bar microstructure proxies.

True microstructure work uses tick or order-book data; on daily bars we
construct proxies that correlate well with the true quantities and serve as
useful conditioning variables for liquidity and adverse-selection risk.

References
----------
- Amihud, Y. (2002). "Illiquidity and stock returns". *Journal of Financial
  Markets* 5, 31-56.
- Roll, R. (1984). "A simple implicit measure of the effective bid-ask
  spread". *Journal of Finance* 39, 1127-1139.
- Kyle, A.S. (1985). "Continuous Auctions and Insider Trading".
  *Econometrica* 53, 1315-1336.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def amihud_illiquidity(
    returns: pd.Series, dollar_volume: pd.Series, window: int = 21
) -> pd.Series:
    r"""Amihud (2002) illiquidity proxy.

    Mathematical Definition
    -----------------------
    .. math::
        \text{ILLIQ}_t = \frac{1}{n}\sum_{i=t-n+1}^{t} \frac{|r_i|}{V_i},

    where :math:`r_i` is the return and :math:`V_i` is the dollar volume of
    bar :math:`i`.

    Parameters
    ----------
    returns : pd.Series
        Period returns (simple or log).
    dollar_volume : pd.Series
        Period dollar volume.
    window : int
        Rolling window length.
    """
    safe_dv = dollar_volume.replace(0.0, np.nan)
    daily = returns.abs() / safe_dv
    return daily.rolling(window).mean()


def roll_spread(close: pd.Series, window: int = 21) -> pd.Series:
    r"""Roll (1984) effective spread.

    Mathematical Definition
    -----------------------
    .. math::
        \hat{s}_t = 2 \sqrt{-\text{Cov}(\Delta p_t, \Delta p_{t-1})}\quad
        \text{if the covariance is negative; 0 otherwise}.

    The estimator is meaningful only when serial covariance of price changes
    is negative, indicating bid-ask bouncing.
    """
    dp = close.diff()
    cov = dp.rolling(window).cov(dp.shift(1))
    spread = 2.0 * np.sqrt((-cov).clip(lower=0.0))
    return spread


def kyle_lambda(
    returns: pd.Series, signed_dollar_volume: pd.Series, window: int = 21
) -> pd.Series:
    r"""Kyle's lambda price-impact proxy.

    Fits :math:`r_t = \lambda \, V^{\text{signed}}_t + \epsilon_t` over a
    rolling window and returns the slope estimate. Signed volume is most
    naturally obtained via the tick rule; on daily bars we accept any caller-
    constructed signed-volume series.

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    signed_dollar_volume : pd.Series
        Signed dollar volume aligned with ``returns``.
    window : int
        Rolling-window length.
    """
    aligned = pd.concat({"r": returns, "v": signed_dollar_volume}, axis=1).dropna()
    out = pd.Series(np.nan, index=returns.index)
    if len(aligned) < window:
        return out
    r = aligned["r"].to_numpy()
    v = aligned["v"].to_numpy()
    for end in range(window, len(aligned) + 1):
        rr = r[end - window : end]
        vv = v[end - window : end]
        denom = (vv**2).sum()
        if denom <= 0:
            continue
        loc = aligned.index.get_loc(aligned.index[end - 1])
        if isinstance(loc, int | np.integer):
            out.iloc[int(loc)] = float((rr * vv).sum() / denom)
    return out


__all__ = ["amihud_illiquidity", "kyle_lambda", "roll_spread"]
