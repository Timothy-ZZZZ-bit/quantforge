"""Combinatorial Purged Cross-Validation (Lopez de Prado AFML Ch. 12).

CPCV produces many out-of-sample paths from a single dataset by combining
multiple disjoint test groups, then purging and embargoing observations
adjacent to each test group to prevent label leakage.

References
----------
Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, Ch. 12.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from quantforge.constants import DEFAULT_CPCV_GROUPS, DEFAULT_CPCV_TEST_GROUPS


@dataclass(frozen=True)
class CPCVSplit:
    """One combination of test groups and the corresponding train indices."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    test_groups: tuple[int, ...]


def _embargo_mask(test_idx: np.ndarray, n: int, embargo: int) -> np.ndarray:
    """Return a boolean mask of indices to remove from the train set.

    Indices removed: anything within ``embargo`` bars of any test index.
    """
    if embargo <= 0:
        return np.zeros(n, dtype=bool)
    keep = np.zeros(n, dtype=bool)
    for t in test_idx:
        lo = max(0, t - embargo)
        hi = min(n, t + embargo + 1)
        keep[lo:hi] = True
    return keep


def cpcv_indices(
    n_obs: int,
    n_groups: int = DEFAULT_CPCV_GROUPS,
    n_test_groups: int = DEFAULT_CPCV_TEST_GROUPS,
    embargo: int = 5,
) -> list[CPCVSplit]:
    """Generate index-level CPCV splits.

    Parameters
    ----------
    n_obs : int
        Number of observations.
    n_groups : int
        Number of contiguous groups to partition observations into.
    n_test_groups : int
        Number of groups assigned to the test side in each combination.
    embargo : int
        Bars to embargo on either side of every test group.

    Returns
    -------
    list[CPCVSplit]
        One split per :math:`\\binom{N}{k}` combination of test groups.
    """
    if n_test_groups >= n_groups:
        raise ValueError("n_test_groups must be < n_groups")
    if n_obs <= 0:
        return []
    group_bounds = np.array_split(np.arange(n_obs), n_groups)
    splits: list[CPCVSplit] = []
    for combo in combinations(range(n_groups), n_test_groups):
        test_idx = np.concatenate([group_bounds[g] for g in combo])
        test_idx.sort()
        emb = _embargo_mask(test_idx, n_obs, embargo)
        emb[test_idx] = True  # also remove the test indices themselves
        train_idx = np.where(~emb)[0]
        splits.append(
            CPCVSplit(train_idx=train_idx, test_idx=test_idx, test_groups=combo)
        )
    return splits


class CombinatorialPurgedCV:
    """Thin convenience wrapper around :func:`cpcv_indices`.

    Parameters
    ----------
    index : pd.Index
        Observation index of the dataset being validated.
    n_groups : int
    n_test_groups : int
    embargo : int

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range("2020-01-01", periods=100, freq="B")
    >>> cv = CombinatorialPurgedCV(idx, n_groups=5, n_test_groups=2, embargo=2)
    >>> len(list(cv.split()))
    10
    """

    def __init__(
        self,
        index: pd.Index,
        n_groups: int = DEFAULT_CPCV_GROUPS,
        n_test_groups: int = DEFAULT_CPCV_TEST_GROUPS,
        embargo: int = 5,
    ) -> None:
        self.index = index
        self.splits = cpcv_indices(
            n_obs=len(index),
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            embargo=embargo,
        )

    def split(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(s.train_idx, s.test_idx) for s in self.splits]

    def n_paths(self) -> int:
        """Number of OOS paths produced by this CPCV layout.

        Lopez de Prado, AFML eq. 12.4: ``(k * C(N, k)) / N`` where
        ``k = n_test_groups`` and ``N = n_groups``.
        """
        # The number of times a single group appears as test across all combos.
        if not self.splits:
            return 0
        n_groups = max(max(s.test_groups) for s in self.splits) + 1
        k = len(self.splits[0].test_groups)
        return int(len(self.splits) * k / n_groups)


__all__ = ["CPCVSplit", "CombinatorialPurgedCV", "cpcv_indices"]
