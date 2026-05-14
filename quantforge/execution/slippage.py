"""Volume-participation slippage helpers."""

from __future__ import annotations

import numpy as np


def participation_slippage(
    shares: float, bar_volume: float, spread_bps: float = 5.0, max_part: float = 0.10
) -> float:
    """Slippage as a fraction of price, capped at full bar participation.

    Parameters
    ----------
    shares : float
        Absolute shares to trade.
    bar_volume : float
        Bar volume in shares.
    spread_bps : float
        Half-spread in basis points at full participation.
    max_part : float
        Upper bound on participation; orders larger than this are spread
        across bars upstream of this function.
    """
    if bar_volume <= 0:
        return 0.0
    part = min(abs(shares) / bar_volume, max_part)
    return float(spread_bps * np.sqrt(part) / 10_000.0)


__all__ = ["participation_slippage"]
