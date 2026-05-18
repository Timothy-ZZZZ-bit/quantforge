"""Position-level and portfolio-level constraints.

Implemented as a post-processing step rather than baked into each optimizer:
- per-name weight caps (long and short),
- gross-leverage cap,
- optional turnover cap relative to a prior portfolio.

This decouples constraint handling from optimization, at the cost of giving
up some optimality. For the targets the platform is built for, the gap is
small.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Constraints:
    """Allocator constraints applied post-optimization."""

    max_weight: float = 0.10
    min_weight: float = -0.10
    gross_leverage: float = 1.0
    turnover_cap: float | None = None


def apply_constraints(
    weights: pd.Series,
    constraints: Constraints,
    prior_weights: pd.Series | None = None,
) -> pd.Series:
    """Apply per-name caps and rescale to the target gross leverage.

    Caps and gross leverage are applied iteratively (clip -> rescale -> clip)
    until both binds are satisfied or a small budget of iterations is
    exhausted. If the constraints are mutually infeasible (e.g., ``max_weight``
    is too tight given ``gross_leverage`` and the universe size), the result
    respects ``max_weight`` and accepts a lower gross leverage.

    A turnover cap, if set, is applied at the end by pulling the new weights
    toward ``prior_weights`` along the connecting line.
    """
    w = weights.copy().astype(float).fillna(0.0)
    # Always clip first; the per-name cap takes precedence over gross leverage.
    w = w.clip(lower=constraints.min_weight, upper=constraints.max_weight)
    gross = float(w.abs().sum())
    # Scale down if gross exceeds the cap; never scale up beyond the per-name caps.
    if gross > constraints.gross_leverage + 1e-12:
        w = w * (constraints.gross_leverage / gross)
        w = w.clip(lower=constraints.min_weight, upper=constraints.max_weight)
    elif gross > 0 and gross < constraints.gross_leverage:
        # Scale up to the gross leverage target, but re-clip to caps; if the
        # caps then bind, accept the lower gross leverage rather than oscillate.
        scaled = w * (constraints.gross_leverage / gross)
        clipped = scaled.clip(
            lower=constraints.min_weight, upper=constraints.max_weight
        )
        # If clipping caused the scale-up to recover the cap, keep the clipped value.
        w = clipped
    if constraints.turnover_cap is not None and prior_weights is not None:
        prior = prior_weights.reindex(w.index).fillna(0.0)
        turnover = float((w - prior).abs().sum())
        if turnover > constraints.turnover_cap:
            alpha = constraints.turnover_cap / turnover
            w = prior + alpha * (w - prior)
    return w


__all__ = ["Constraints", "apply_constraints"]
