"""Feature engineering primitives.

The features fall into four buckets:

- ``returns``        - log/simple returns and fractional differencing.
- ``technical``      - momentum, drawdown, multi-estimator realized volatility.
- ``microstructure`` - Amihud illiquidity, Roll spread, Kyle lambda proxy.
- ``cross_sectional``- rank, z-score, sector-neutralization.
- ``labeling``       - triple-barrier method and meta-labels.
"""

from __future__ import annotations

from quantforge.features.cross_sectional import cs_rank, cs_zscore, sector_neutralize
from quantforge.features.labeling import meta_labels, triple_barrier
from quantforge.features.microstructure import (
    amihud_illiquidity,
    kyle_lambda,
    roll_spread,
)
from quantforge.features.returns import (
    cumulative_returns,
    frac_diff_ffd,
    log_returns,
    simple_returns,
)
from quantforge.features.technical import momentum, realized_vol

__all__ = [
    "amihud_illiquidity",
    "cs_rank",
    "cs_zscore",
    "cumulative_returns",
    "frac_diff_ffd",
    "kyle_lambda",
    "log_returns",
    "meta_labels",
    "momentum",
    "realized_vol",
    "roll_spread",
    "sector_neutralize",
    "simple_returns",
    "triple_barrier",
]
