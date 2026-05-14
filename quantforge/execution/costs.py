"""Transaction-cost models.

The combined cost of a trade is decomposed into

.. math::
    c = \\text{commission}(\\text{notional}) + \\text{slippage}(\\text{notional}, \\text{participation})
        + \\text{impact}(\\text{notional}, \\text{participation}) + \\text{borrow}.

Each component is modeled in isolation, then linearly aggregated.

References
----------
- Almgren, Chriss (2000). "Optimal Execution of Portfolio Transactions".
- Almgren, Thum, Hauptmann, Li (2005). "Direct Estimation of Equity Market Impact".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantforge.constants import (
    DEFAULT_BORROW_BPS_ANNUAL,
    DEFAULT_COMMISSION_BPS,
    DEFAULT_IMPACT_COEF,
    DEFAULT_SLIPPAGE_BPS,
    TRADING_DAYS_PER_YEAR,
)


def commission_cost(traded_notional: float, bps: float = DEFAULT_COMMISSION_BPS) -> float:
    """Flat-rate commission in basis points of traded notional."""
    return abs(traded_notional) * bps / 10_000.0


def linear_slippage(
    traded_notional: float,
    participation: float,
    bps_at_full_participation: float = DEFAULT_SLIPPAGE_BPS,
) -> float:
    r"""Linear slippage proportional to participation rate.

    .. math::
        s = |N| \cdot \frac{p}{1} \cdot \frac{\text{bps}}{10000},

    where :math:`p` is the participation rate (order size / bar volume).
    """
    if participation < 0:
        raise ValueError("participation must be non-negative")
    return abs(traded_notional) * participation * bps_at_full_participation / 10_000.0


def almgren_chriss_impact(
    traded_notional: float,
    participation: float,
    coef: float = DEFAULT_IMPACT_COEF,
) -> float:
    r"""Square-root temporary market impact.

    .. math::
        I = \eta \cdot |N| \cdot \sqrt{p},

    where :math:`\eta` is the impact coefficient and :math:`p` is the
    participation rate. The coefficient default of 0.10 follows Almgren et al.
    (2005) for liquid US equities; see ``constants.DEFAULT_IMPACT_COEF`` for
    the citation.
    """
    if participation < 0:
        raise ValueError("participation must be non-negative")
    return abs(traded_notional) * coef * float(np.sqrt(participation))


def daily_borrow_cost(
    short_notional: float, annual_bps: float = DEFAULT_BORROW_BPS_ANNUAL
) -> float:
    """Daily borrow cost for a short position (per bar)."""
    if short_notional > 0:
        return 0.0
    return abs(short_notional) * (annual_bps / 10_000.0) / TRADING_DAYS_PER_YEAR


@dataclass(frozen=True)
class CostModel:
    """Composable transaction-cost model used by the backtest engine.

    Parameters
    ----------
    commission_bps : float
        Per-side commission in basis points.
    slippage_bps : float
        Linear slippage in basis points at full bar participation.
    impact_coef : float
        Square-root impact coefficient.
    borrow_bps_annual : float
        Stock-borrow cost on the short notional, in basis points per annum.
    """

    commission_bps: float = DEFAULT_COMMISSION_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    impact_coef: float = DEFAULT_IMPACT_COEF
    borrow_bps_annual: float = DEFAULT_BORROW_BPS_ANNUAL

    def trade_cost(self, traded_notional: float, participation: float) -> float:
        return (
            commission_cost(traded_notional, self.commission_bps)
            + linear_slippage(traded_notional, participation, self.slippage_bps)
            + almgren_chriss_impact(traded_notional, participation, self.impact_coef)
        )

    def carry_cost(self, position_notional_by_asset: dict[str, float]) -> float:
        """Sum of borrow costs across short positions for one bar."""
        return sum(daily_borrow_cost(n, self.borrow_bps_annual) for n in position_notional_by_asset.values())

    def scale(self, multiplier: float) -> CostModel:
        """Return a copy with every per-trade component scaled by ``multiplier``.

        Used for the cost-sensitivity stress test in the validation protocol.
        Borrow cost is left unscaled (it is a financing cost, not a trade cost).
        """
        return CostModel(
            commission_bps=self.commission_bps * multiplier,
            slippage_bps=self.slippage_bps * multiplier,
            impact_coef=self.impact_coef * multiplier,
            borrow_bps_annual=self.borrow_bps_annual,
        )


__all__ = [
    "CostModel",
    "almgren_chriss_impact",
    "commission_cost",
    "daily_borrow_cost",
    "linear_slippage",
]
