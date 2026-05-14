"""Historical scenario replay for stress testing.

Canonical stress windows:
- 2008 GFC: 2008-09-01 to 2009-03-31.
- COVID-19 crash: 2020-02-15 to 2020-04-15.
- 2022 inflation drawdown: 2022-01-01 to 2022-10-31.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from quantforge.metrics.performance import max_drawdown, sharpe_ratio
from quantforge.risk.cvar import expected_shortfall


@dataclass(frozen=True)
class StressWindow:
    name: str
    start: str
    end: str


STRESS_WINDOWS: Final[tuple[StressWindow, ...]] = (
    StressWindow("GFC 2008", "2008-09-01", "2009-03-31"),
    StressWindow("COVID 2020", "2020-02-15", "2020-04-15"),
    StressWindow("Inflation 2022", "2022-01-01", "2022-10-31"),
)


@dataclass(frozen=True)
class StressResult:
    name: str
    start: str
    end: str
    total_return: float
    sharpe: float
    max_drawdown: float
    expected_shortfall_95: float
    n_obs: int


def historical_stress(
    returns: pd.Series, windows: tuple[StressWindow, ...] = STRESS_WINDOWS
) -> list[StressResult]:
    """Apply each stress window to a returns series and summarize."""
    out: list[StressResult] = []
    for w in windows:
        sub = returns.loc[w.start : w.end].dropna()
        if sub.empty:
            out.append(
                StressResult(
                    name=w.name,
                    start=w.start,
                    end=w.end,
                    total_return=float("nan"),
                    sharpe=float("nan"),
                    max_drawdown=float("nan"),
                    expected_shortfall_95=float("nan"),
                    n_obs=0,
                )
            )
            continue
        total = float(np.exp(np.log(1.0 + sub).sum()) - 1.0)
        out.append(
            StressResult(
                name=w.name,
                start=w.start,
                end=w.end,
                total_return=total,
                sharpe=sharpe_ratio(sub),
                max_drawdown=max_drawdown(sub),
                expected_shortfall_95=expected_shortfall(sub, confidence=0.95),
                n_obs=len(sub),
            )
        )
    return out


__all__ = ["STRESS_WINDOWS", "StressResult", "StressWindow", "historical_stress"]
