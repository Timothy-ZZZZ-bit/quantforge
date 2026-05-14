"""Walk-forward splits for rolling and expanding-window evaluation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from quantforge.constants import TRADING_DAYS_PER_YEAR


@dataclass(frozen=True)
class WalkForwardSplit:
    """A single train/test split for walk-forward validation."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_splits(
    dates: pd.DatetimeIndex,
    train_years: int = 5,
    test_years: int = 1,
    mode: Literal["rolling", "expanding"] = "rolling",
    step_years: int | None = None,
) -> Iterator[WalkForwardSplit]:
    """Yield walk-forward train/test splits.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Strictly increasing date index of the available data.
    train_years : int
        Length of the train window in years.
    test_years : int
        Length of each out-of-sample fold.
    mode : {"rolling", "expanding"}
        ``rolling`` slides a fixed-width train window; ``expanding`` grows it.
    step_years : int, optional
        Step between folds; defaults to ``test_years`` (non-overlapping folds).
    """
    if len(dates) == 0:
        return
    if not dates.is_monotonic_increasing:
        raise ValueError("dates must be sorted")
    step = step_years if step_years is not None else test_years
    train_bars = train_years * TRADING_DAYS_PER_YEAR
    test_bars = test_years * TRADING_DAYS_PER_YEAR
    step_bars = step * TRADING_DAYS_PER_YEAR

    i = train_bars
    while i + test_bars <= len(dates):
        train_lo = 0 if mode == "expanding" else max(0, i - train_bars)
        train_hi = i - 1
        test_lo = i
        test_hi = i + test_bars - 1
        yield WalkForwardSplit(
            train_start=dates[train_lo],
            train_end=dates[train_hi],
            test_start=dates[test_lo],
            test_end=dates[test_hi],
        )
        i += step_bars


__all__ = ["WalkForwardSplit", "walk_forward_splits"]
