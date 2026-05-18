"""Triple-barrier method and meta-labeling (Lopez de Prado AFML Ch. 3).

The triple-barrier method labels a trade by which of three barriers is
touched first: upper profit-take, lower stop-loss, or vertical time barrier.
Meta-labels supplement a primary model's predictions with a secondary
classifier deciding whether to act on the primary signal at all.

References
----------
Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, Ch. 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TripleBarrierConfig:
    """Configuration for :func:`triple_barrier`.

    Parameters
    ----------
    pt : float
        Upper-barrier multiplier, expressed as a multiple of bar-level
        target volatility. A value of 2.0 means take profit at 2 sigma.
    sl : float
        Lower-barrier multiplier (always positive; the barrier is at
        ``-sl * vol``).
    vertical : int
        Vertical barrier in number of bars.
    min_ret : float
        Minimum absolute return required to keep the label; below this
        the bar is marked as 0 (no trade).
    """

    pt: float = 2.0
    sl: float = 2.0
    vertical: int = 21
    min_ret: float = 0.0


def triple_barrier(
    prices: pd.Series,
    target_vol: pd.Series,
    events: pd.Series | None = None,
    config: TripleBarrierConfig | None = None,
) -> pd.DataFrame:
    """Apply the triple-barrier method to a price series.

    Parameters
    ----------
    prices : pd.Series
        Adjusted close prices indexed by date.
    target_vol : pd.Series
        Per-bar target volatility (same index as ``prices``). Typically a
        rolling realized vol estimate.
    events : pd.Series, optional
        Boolean mask: True at the bars where a label should be computed.
        Defaults to all bars where ``target_vol`` is not NaN.
    config : TripleBarrierConfig, optional
        Barrier configuration. Defaults to ``TripleBarrierConfig()``.

    Returns
    -------
    pd.DataFrame
        Columns ``[t1, ret, bin]`` indexed by event date.

        - ``t1``: date at which one of the barriers was first touched.
        - ``ret``: log return from event to ``t1``.
        - ``bin``: -1 (lower hit), +1 (upper hit), 0 (vertical hit / no edge).

    Notes
    -----
    Implementation is a straightforward Python loop; correctness is more
    important than throughput here, and the function is typically called
    once per backtest train/test split.
    """
    cfg = config or TripleBarrierConfig()
    if events is None:
        events = pd.Series(True, index=prices.index)

    log_p = np.log(prices.to_numpy(dtype=float))
    vol = target_vol.reindex(prices.index).to_numpy(dtype=float)
    idx = prices.index
    n = len(idx)
    event_mask = events.reindex(prices.index, fill_value=False).to_numpy(dtype=bool)

    rows: list[tuple[pd.Timestamp, pd.Timestamp, float, int]] = []
    for i in range(n):
        if not event_mask[i]:
            continue
        if not np.isfinite(vol[i]):
            continue
        upper = cfg.pt * vol[i]
        lower = -cfg.sl * vol[i]
        end_idx = min(i + cfg.vertical, n - 1)
        slc = log_p[i + 1 : end_idx + 1] - log_p[i]
        if slc.size == 0:
            rows.append((idx[i], idx[end_idx], 0.0, 0))
            continue
        up_hits = np.where(slc >= upper)[0]
        dn_hits = np.where(slc <= lower)[0]
        t_up = up_hits[0] if up_hits.size else np.inf
        t_dn = dn_hits[0] if dn_hits.size else np.inf
        if t_up < t_dn:
            t1_offset = int(t_up) + 1
            ret = float(slc[int(t_up)])
            label = 1
        elif t_dn < t_up:
            t1_offset = int(t_dn) + 1
            ret = float(slc[int(t_dn)])
            label = -1
        else:
            t1_offset = len(slc)
            ret = float(slc[-1])
            label = 0
        t1 = idx[min(i + t1_offset, n - 1)]
        if abs(ret) < cfg.min_ret:
            label = 0
        rows.append((idx[i], t1, ret, label))

    if not rows:
        return pd.DataFrame(columns=["t1", "ret", "bin"])
    out = pd.DataFrame(rows, columns=["t0", "t1", "ret", "bin"]).set_index("t0")
    return out


def meta_labels(
    primary_predictions: pd.Series,
    triple_barrier_bins: pd.Series,
) -> pd.Series:
    """Construct meta-labels: 1 if the primary prediction agrees with the
    realized triple-barrier sign, 0 otherwise.

    Parameters
    ----------
    primary_predictions : pd.Series
        Signed primary signal (e.g. {-1, +1}).
    triple_barrier_bins : pd.Series
        Realized triple-barrier label in {-1, 0, +1}.

    Returns
    -------
    pd.Series
        Binary series aligned to the intersection of the inputs.
    """
    joined = pd.concat(
        {"pred": primary_predictions, "bin": triple_barrier_bins}, axis=1
    ).dropna()
    return (np.sign(joined["pred"]) == np.sign(joined["bin"])).astype(int)


__all__ = ["TripleBarrierConfig", "meta_labels", "triple_barrier"]
