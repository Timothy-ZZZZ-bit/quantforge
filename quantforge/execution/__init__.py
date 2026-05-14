"""Realistic execution-cost models.

Three components, combined linearly by the backtest engine:

- ``commissions`` (basis points per traded notional);
- ``slippage`` (linear in participation rate);
- ``impact`` (square-root in participation rate, Almgren-Chriss style).
"""

from __future__ import annotations

from quantforge.execution.costs import (
    CostModel,
    almgren_chriss_impact,
    commission_cost,
    linear_slippage,
)
from quantforge.execution.fills import fill_at_next_open
from quantforge.execution.slippage import participation_slippage

__all__ = [
    "CostModel",
    "almgren_chriss_impact",
    "commission_cost",
    "fill_at_next_open",
    "linear_slippage",
    "participation_slippage",
]
