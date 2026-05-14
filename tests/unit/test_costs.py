from __future__ import annotations

import math

import pytest

from quantforge.execution.costs import (
    CostModel,
    almgren_chriss_impact,
    commission_cost,
    daily_borrow_cost,
    linear_slippage,
)


def test_commission_cost_linear_in_notional():
    assert commission_cost(1_000_000.0, bps=1.0) == pytest.approx(100.0)


def test_linear_slippage_zero_at_zero_participation():
    assert linear_slippage(1_000_000.0, participation=0.0) == 0.0


def test_almgren_chriss_impact_sqrt_in_participation():
    a = almgren_chriss_impact(1.0, participation=0.04, coef=1.0)
    b = almgren_chriss_impact(1.0, participation=0.16, coef=1.0)
    # sqrt(0.16) / sqrt(0.04) = 2.0
    assert b == pytest.approx(2.0 * a, rel=1e-9)


def test_negative_participation_rejected():
    with pytest.raises(ValueError):
        linear_slippage(1.0, participation=-0.1)
    with pytest.raises(ValueError):
        almgren_chriss_impact(1.0, participation=-0.1)


def test_daily_borrow_cost_only_on_shorts():
    assert daily_borrow_cost(short_notional=100_000.0) == 0.0
    assert daily_borrow_cost(short_notional=-100_000.0) > 0


def test_cost_model_scale_multiplies():
    m = CostModel(commission_bps=1.0, slippage_bps=5.0, impact_coef=0.10)
    m2 = m.scale(2.0)
    assert math.isclose(m2.commission_bps, 2.0)
    assert math.isclose(m2.slippage_bps, 10.0)
    assert math.isclose(m2.impact_coef, 0.20)


def test_cost_model_trade_cost_positive():
    m = CostModel()
    assert m.trade_cost(1_000_000.0, participation=0.05) > 0
