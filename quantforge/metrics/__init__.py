"""Performance metrics, PSR/DSR, and factor attribution."""

from __future__ import annotations

from quantforge.metrics.attribution import factor_exposure
from quantforge.metrics.factor_exposure import fama_french_betas
from quantforge.metrics.performance import (
    annualized_return,
    annualized_volatility,
    calmar,
    hit_rate,
    omega,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)
from quantforge.metrics.psr import deflated_sharpe, probabilistic_sharpe

__all__ = [
    "annualized_return",
    "annualized_volatility",
    "calmar",
    "deflated_sharpe",
    "factor_exposure",
    "fama_french_betas",
    "hit_rate",
    "omega",
    "probabilistic_sharpe",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
]
