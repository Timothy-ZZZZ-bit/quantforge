"""Portfolio construction: allocators that turn alpha scores into weights."""

from __future__ import annotations

from quantforge.portfolio.base import Allocator
from quantforge.portfolio.black_litterman import BlackLittermanAllocator
from quantforge.portfolio.constraints import Constraints, apply_constraints
from quantforge.portfolio.equal_weight import EqualWeight
from quantforge.portfolio.hrp import HRPAllocator
from quantforge.portfolio.mvo import MeanVariance
from quantforge.portfolio.risk_parity import EqualRiskContribution

__all__ = [
    "Allocator",
    "BlackLittermanAllocator",
    "Constraints",
    "EqualRiskContribution",
    "EqualWeight",
    "HRPAllocator",
    "MeanVariance",
    "apply_constraints",
]
