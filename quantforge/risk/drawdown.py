"""Drawdown utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Per-bar drawdown from the running peak."""
    r = returns.dropna()
    if r.empty:
        return pd.Series(dtype=float)
    wealth = (1.0 + r).cumprod()
    return wealth / wealth.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Worst drawdown over the sample."""
    dd = drawdown_series(returns)
    if dd.empty:
        return float("nan")
    return float(dd.min())


def time_underwater(returns: pd.Series) -> int:
    """Longest stretch (in bars) under the previous peak."""
    dd = drawdown_series(returns)
    if dd.empty:
        return 0
    underwater = (dd < 0).astype(int).to_numpy()
    longest, run = 0, 0
    for u in underwater:
        if u:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return int(longest)


def recovery_time(returns: pd.Series) -> int:
    """Bars from the trough of the worst drawdown to recovery, or 0 if not recovered."""
    dd = drawdown_series(returns)
    if dd.empty:
        return 0
    trough = int(np.argmin(dd.to_numpy()))
    after = dd.iloc[trough:]
    recovered = after[after >= -1e-9]
    if recovered.empty:
        return 0
    loc = after.index.get_loc(recovered.index[0])
    return int(loc) if isinstance(loc, int | np.integer) else 0


__all__ = ["drawdown_series", "max_drawdown", "recovery_time", "time_underwater"]
