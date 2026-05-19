"""Coverage for the tearsheet builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantforge.backtest import BacktestEngine
from quantforge.data import synthetic_equity_panel
from quantforge.execution.costs import CostModel
from quantforge.reporting import TearsheetData, build_tearsheet, summarize_results


def _equal_weight_fn(date, visible):
    tickers = visible["ticker"].unique().tolist()
    if not tickers:
        return {}
    return dict.fromkeys(tickers, 1.0 / len(tickers))


def _run() -> tuple:
    panel = synthetic_equity_panel(n_tickers=6, n_days=600, seed=21)
    engine = BacktestEngine(panel, _equal_weight_fn, cost_model=CostModel())
    result = engine.run()
    bench = panel[panel["ticker"] == "SYN000"].set_index("date")["adj_close"]
    return result, bench / bench.iloc[0] * 1_000_000.0


def test_summarize_results_keys():
    result, bench = _run()
    summary = summarize_results(result, bench)
    for key in ("cagr", "vol", "sharpe", "sortino", "calmar", "max_drawdown"):
        assert key in summary
    assert "benchmark_cagr" in summary


def test_build_tearsheet_writes_html(tmp_path: Path):
    result, bench = _run()
    cpcv_sharpes = pd.Series([0.5, 0.7, 0.9, -0.1, 0.3])
    data = TearsheetData(
        run_name="test_run",
        description="coverage test",
        run_id="testid",
        config_hash="hash123",
        config_yaml="name: test",
        equity=result.equity,
        benchmark_equity=bench,
        turnover=result.turnover,
        weights=result.weights,
        n_trials_for_dsr=12,
        cpcv_sharpes=cpcv_sharpes,
    )
    out = build_tearsheet(data, tmp_path / "ts.html")
    assert out.exists()
    html = out.read_text()
    assert "test_run" in html
    assert "Deflated Sharpe" in html
    assert "CPCV" in html


def test_build_tearsheet_without_benchmark_or_cpcv(tmp_path: Path):
    result, _ = _run()
    data = TearsheetData(
        run_name="minimal",
        description="",
        run_id="id",
        config_hash="h",
        config_yaml="x: 1",
        equity=result.equity,
    )
    out = build_tearsheet(data, tmp_path / "min.html")
    assert out.exists()
    assert "CPCV not computed" in out.read_text()
