"""Abstract base class shared by every signal."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class SignalDiagnostics:
    """Diagnostic summary for a fitted signal.

    Attributes
    ----------
    information_coefficient : float
        Spearman rank correlation of signal with forward returns.
    ic_p_value : float
        Two-sided p-value for IC under the null of independence.
    ic_bootstrap_ci : tuple[float, float]
        95% bootstrap CI on the IC.
    forward_horizon : int
        Bars used for the forward-return computation.
    n_obs : int
        Number of stacked (date, ticker) IC observations.
    """

    information_coefficient: float
    ic_p_value: float
    ic_bootstrap_ci: tuple[float, float]
    forward_horizon: int
    n_obs: int


def information_coefficient(
    signal: pd.Series,
    forward_returns: pd.Series,
    n_bootstrap: int = 500,
    seed: int = 0,
) -> SignalDiagnostics:
    """Spearman rank IC plus bootstrap CI.

    Parameters
    ----------
    signal : pd.Series
        Signal indexed by ``[date, ticker]`` or panel index.
    forward_returns : pd.Series
        Realized forward returns aligned to ``signal``.
    n_bootstrap : int
        Bootstrap replicates for the 95% CI.
    seed : int
        Random seed for the bootstrap.
    """
    df = pd.concat({"s": signal, "r": forward_returns}, axis=1).dropna()
    if len(df) < 30:
        return SignalDiagnostics(
            information_coefficient=float("nan"),
            ic_p_value=float("nan"),
            ic_bootstrap_ci=(float("nan"), float("nan")),
            forward_horizon=0,
            n_obs=len(df),
        )
    rho, p = stats.spearmanr(df["s"], df["r"])
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rhos = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(idx, size=len(idx), replace=True)
        rhos[b], _ = stats.spearmanr(df["s"].iloc[sample], df["r"].iloc[sample])
    ci_lo, ci_hi = float(np.quantile(rhos, 0.025)), float(np.quantile(rhos, 0.975))
    return SignalDiagnostics(
        information_coefficient=float(rho),
        ic_p_value=float(p),
        ic_bootstrap_ci=(ci_lo, ci_hi),
        forward_horizon=0,
        n_obs=len(df),
    )


class Signal(ABC):
    """Abstract base for alpha-generating strategies.

    Subclasses implement :meth:`fit` and :meth:`predict`. ``fit`` may be a
    no-op for stateless signals (e.g., pure momentum). ``predict`` returns a
    per-asset alpha score for the most recent bar in ``panel``.
    """

    name: str = "signal"

    @abstractmethod
    def fit(self, panel: pd.DataFrame) -> Signal:
        """Fit any internal parameters using strictly past data."""

    @abstractmethod
    def predict(self, panel: pd.DataFrame) -> pd.Series:
        """Return a Series of per-ticker alpha scores for the latest date."""


__all__ = ["Signal", "SignalDiagnostics", "information_coefficient"]
