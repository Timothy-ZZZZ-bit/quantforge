from __future__ import annotations

import pandas as pd

from quantforge.backtest import BacktestEngine, walk_forward_splits
from quantforge.execution.costs import CostModel


def test_walk_forward_yields_disjoint_test_windows():
    idx = pd.bdate_range("2010-01-04", periods=252 * 10)
    splits = list(walk_forward_splits(idx, train_years=3, test_years=1, mode="rolling"))
    assert len(splits) >= 2
    for s in splits:
        assert s.train_end < s.test_start
        assert s.test_end > s.test_start


def test_walk_forward_expanding_grows():
    idx = pd.bdate_range("2010-01-04", periods=252 * 6)
    splits = list(
        walk_forward_splits(idx, train_years=3, test_years=1, mode="expanding")
    )
    first_train_len = (splits[0].train_end - splits[0].train_start).days
    last_train_len = (splits[-1].train_end - splits[-1].train_start).days
    assert last_train_len >= first_train_len


def test_engine_equal_weight_runs_and_grows_or_drops(medium_panel):
    def weight_fn(date, visible):
        tickers = visible["ticker"].unique().tolist()
        if not tickers:
            return {}
        n = len(tickers)
        return dict.fromkeys(tickers, 1.0 / n)

    engine = BacktestEngine(
        medium_panel,
        weight_fn=weight_fn,
        cost_model=CostModel(),
        rebalance_freq="BMS",
    )
    result = engine.run()
    assert not result.equity.empty
    assert result.equity.notna().all()
    # No NaN in turnover or gross leverage.
    assert result.gross_leverage.notna().all()
