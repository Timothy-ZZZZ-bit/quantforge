"""Numeric and convention constants used across QuantForge.

Every magic number that appears more than once or carries economic meaning
lives here with a comment citing its source. Modules should import from this
file rather than redefining.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# Trading-day conventions.

TRADING_DAYS_PER_YEAR: Final[int] = 252
"""US equity calendar convention. See Bodie/Kane/Marcus, *Investments*."""

TRADING_DAYS_PER_MONTH: Final[int] = 21
"""Standard approximation for monthly horizon at the daily bar level."""

TRADING_DAYS_PER_WEEK: Final[int] = 5

# Risk-free proxy.

DEFAULT_RF_TICKER: Final[str] = "DGS3MO"
"""3-month Treasury constant maturity from FRED, used as a risk-free proxy."""

# Volatility annualization (sqrt of trading-days-per-year).

VOL_ANNUALIZATION_FACTOR: Final[float] = TRADING_DAYS_PER_YEAR**0.5  # sqrt(252)

# Strategy execution defaults.

DEFAULT_PARTICIPATION_CAP: Final[float] = 0.10
"""Cap our order size at 10% of the bar's traded volume."""

DEFAULT_COMMISSION_BPS: Final[float] = 1.0
"""1 basis point per side as a coarse, lenient retail/HFT-leaning default."""

DEFAULT_SLIPPAGE_BPS: Final[float] = 5.0
"""Linear slippage component, 5bps. See Almgren-Chriss (2000)."""

DEFAULT_IMPACT_COEF: Final[float] = 0.10
"""Square-root impact coefficient. Calibrated to be conservative for US equities.

Almgren et al. (2005), Direct Estimation of Equity Market Impact, find values
in the range 0.05 to 0.15 for daily volumes; we adopt 0.10 as a neutral default.
"""

DEFAULT_BORROW_BPS_ANNUAL: Final[float] = 50.0
"""Generic stock-borrow cost: 50bps per annum on the short notional."""

# Numerical tolerances.

EPS: Final[float] = 1e-12
"""Default tolerance for floating-point equality checks."""

POSITIVE_TOL: Final[float] = 1e-10
"""Tolerance below which a quantity is treated as zero/non-positive."""

# Statistical defaults.

DEFAULT_CONFIDENCE_LEVEL: Final[float] = 0.95
"""Default confidence level for VaR, CVaR, bootstrap CIs."""

DEFAULT_BOOTSTRAP_BLOCK: Final[int] = 21
"""Average block length for stationary bootstrap. Politis-Romano (1994).

Roughly one trading month: long enough to capture autocorrelation in daily returns,
short enough to preserve sample-size statistical power.
"""

DEFAULT_BOOTSTRAP_REPLICATES: Final[int] = 1_000

# CPCV defaults: Lopez de Prado AFML Ch. 12 recommendation.

DEFAULT_CPCV_GROUPS: Final[int] = 10
DEFAULT_CPCV_TEST_GROUPS: Final[int] = 2

# Project-relative cache and run-artifact roots.

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_CACHE_DIR: Final[Path] = PROJECT_ROOT / "data" / "cache"
RUN_ARTIFACTS_DIR: Final[Path] = PROJECT_ROOT / "run_artifacts"
RESEARCH_LOG_PATH: Final[Path] = PROJECT_ROOT / "research_log.jsonl"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"

__all__ = [
    "DATA_CACHE_DIR",
    "DEFAULT_BOOTSTRAP_BLOCK",
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BORROW_BPS_ANNUAL",
    "DEFAULT_COMMISSION_BPS",
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_CPCV_GROUPS",
    "DEFAULT_CPCV_TEST_GROUPS",
    "DEFAULT_IMPACT_COEF",
    "DEFAULT_PARTICIPATION_CAP",
    "DEFAULT_RF_TICKER",
    "DEFAULT_SLIPPAGE_BPS",
    "EPS",
    "POSITIVE_TOL",
    "PROJECT_ROOT",
    "REPORTS_DIR",
    "RESEARCH_LOG_PATH",
    "RUN_ARTIFACTS_DIR",
    "TRADING_DAYS_PER_MONTH",
    "TRADING_DAYS_PER_WEEK",
    "TRADING_DAYS_PER_YEAR",
    "VOL_ANNUALIZATION_FACTOR",
]
