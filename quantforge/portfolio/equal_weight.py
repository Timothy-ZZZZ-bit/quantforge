"""Equal-weight allocator with optional sign from alphas."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.portfolio.base import Allocator


class EqualWeight(Allocator):
    """Equal weight per name.

    Modes:
    - ``signless``: long every name in ``alphas`` with weight ``1/n``.
    - ``signed``: weight ``sign(alpha) / n_nonzero``.
    """

    name = "equal_weight"

    def __init__(self, mode: str = "signed", gross_leverage: float = 1.0) -> None:
        if mode not in ("signless", "signed"):
            raise ValueError(f"unknown mode: {mode!r}")
        self.mode = mode
        self.gross_leverage = gross_leverage

    def allocate(
        self,
        alphas: pd.Series,
        returns_history: pd.DataFrame | None = None,
    ) -> pd.Series:
        s = alphas.dropna()
        if s.empty:
            return pd.Series(dtype=float)
        if self.mode == "signless":
            w = pd.Series(1.0 / len(s), index=s.index)
        else:
            signs = np.sign(s)
            nz = (signs != 0).sum()
            if nz == 0:
                return pd.Series(0.0, index=s.index)
            w = signs / nz
        return w * self.gross_leverage


__all__ = ["EqualWeight"]
