"""Base class for portfolio allocators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Allocator(ABC):
    """Abstract allocator.

    An allocator takes
    - an alpha view (per-asset score or expected return), and optionally
    - a returns history (used to estimate the covariance matrix),
    and returns target weights summing to a specified gross leverage.
    """

    name: str = "allocator"

    @abstractmethod
    def allocate(
        self,
        alphas: pd.Series,
        returns_history: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Return target weights indexed by ticker."""


__all__ = ["Allocator"]
