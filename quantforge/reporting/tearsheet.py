"""Tearsheet builder.

A tearsheet is a single self-contained HTML file that summarizes a backtest.
The same data structure can also be exported to PDF via WeasyPrint (optional
dependency).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plotly.subplots import make_subplots

import quantforge
from quantforge.backtest.engine import BacktestResult
from quantforge.metrics.performance import (
    annualized_return,
    annualized_volatility,
    calmar,
    hit_rate,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    time_underwater,
)
from quantforge.metrics.psr import deflated_sharpe, probabilistic_sharpe
from quantforge.risk.drawdown import drawdown_series
from quantforge.risk.stress import StressResult, historical_stress

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _kpi(label: str, value: float, fmt: str = ".2f", sub: str = "") -> dict[str, Any]:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        formatted = "n/a"
    else:
        formatted = format(value, fmt)
    return {"label": label, "value": formatted, "sub": sub}


@dataclass
class TearsheetData:
    """All inputs required to render a tearsheet.

    Attributes
    ----------
    run_name : str
    description : str
    run_id : str
    config_hash : str
    config_yaml : str
    equity : pd.Series
    benchmark_equity : pd.Series | None
    turnover : pd.Series | None
    weights : pd.DataFrame | None
    n_trials_for_dsr : int
    cpcv_sharpes : pd.Series | None
    factor_table : pd.DataFrame | None
    """

    run_name: str
    description: str
    run_id: str
    config_hash: str
    config_yaml: str
    equity: pd.Series
    benchmark_equity: pd.Series | None = None
    turnover: pd.Series | None = None
    weights: pd.DataFrame | None = None
    n_trials_for_dsr: int = 1
    cpcv_sharpes: pd.Series | None = None
    factor_table: pd.DataFrame | None = None
    stress: list[StressResult] = field(default_factory=list)


def summarize_results(result: BacktestResult, benchmark: pd.Series | None = None) -> dict[str, float]:
    """Return a flat dict of summary statistics."""
    r = result.returns()
    out = {
        "cagr": annualized_return(r),
        "vol": annualized_volatility(r),
        "sharpe": sharpe_ratio(r),
        "sortino": sortino_ratio(r),
        "calmar": calmar(r),
        "max_drawdown": max_drawdown(r),
        "hit_rate": hit_rate(r),
        "profit_factor": profit_factor(r),
        "time_underwater": float(time_underwater(r)),
    }
    if benchmark is not None and not benchmark.empty:
        bench_r = benchmark.pct_change().dropna()
        out["benchmark_cagr"] = annualized_return(bench_r)
        out["benchmark_sharpe"] = sharpe_ratio(bench_r)
    return out


def _equity_chart(
    equity: pd.Series, benchmark_equity: pd.Series | None
) -> str:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(
        go.Scatter(x=equity.index, y=equity / equity.iloc[0], mode="lines", name="Strategy"),
        row=1, col=1,
    )
    if benchmark_equity is not None and not benchmark_equity.empty:
        bench = benchmark_equity.reindex(equity.index).dropna()
        if not bench.empty:
            fig.add_trace(
                go.Scatter(x=bench.index, y=bench / bench.iloc[0], mode="lines", name="Benchmark"),
                row=1, col=1,
            )
    fig.update_yaxes(type="log", row=1, col=1, title="Wealth (log scale)")
    dd = drawdown_series(equity.pct_change().dropna())
    fig.add_trace(
        go.Scatter(x=dd.index, y=dd, mode="lines", fill="tozeroy", name="Drawdown", line=dict(color="#d62728")),
        row=2, col=1,
    )
    fig.update_yaxes(title="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_layout(template="plotly_white", height=520, margin=dict(l=20, r=20, t=20, b=20))
    return fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="eq-chart")


def _drawdown_chart(equity: pd.Series) -> str:
    dd = drawdown_series(equity.pct_change().dropna())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", fill="tozeroy", line=dict(color="#d62728")))
    fig.update_layout(template="plotly_white", height=240, yaxis_tickformat=".0%", margin=dict(l=20, r=20, t=20, b=20))
    return fig.to_html(include_plotlyjs=False, full_html=False, div_id="dd-chart")


def _rolling_chart(equity: pd.Series, benchmark_equity: pd.Series | None) -> str:
    r = equity.pct_change().dropna()
    roll_sharpe = (r.rolling(252).mean() / r.rolling(252).std(ddof=1)) * np.sqrt(252)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe, mode="lines", name="Rolling 12m Sharpe"), row=1, col=1)
    if benchmark_equity is not None and not benchmark_equity.empty:
        br = benchmark_equity.reindex(equity.index).pct_change().dropna()
        cov = r.rolling(252).cov(br)
        var = br.rolling(252).var(ddof=1)
        beta = (cov / var).reindex(r.index)
        fig.add_trace(go.Scatter(x=beta.index, y=beta, mode="lines", name="Rolling beta to benchmark"), row=2, col=1)
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=20, r=20, t=20, b=20))
    return fig.to_html(include_plotlyjs=False, full_html=False, div_id="rolling-chart")


def _dist_chart(equity: pd.Series) -> str:
    r = equity.pct_change().dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r, nbinsx=60, histnorm="probability density", name="Empirical"))
    xs = np.linspace(r.min(), r.max(), 200)
    mu, sd = r.mean(), r.std(ddof=1)
    fig.add_trace(go.Scatter(x=xs, y=(1.0 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sd) ** 2),
                             mode="lines", name="Normal fit"))
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=20, b=20))
    return fig.to_html(include_plotlyjs=False, full_html=False, div_id="dist-chart")


def _cpcv_chart(cpcv_sharpes: pd.Series | None) -> str:
    if cpcv_sharpes is None or cpcv_sharpes.empty:
        return "<p class='note'>CPCV not computed for this run.</p>"
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=cpcv_sharpes, nbinsx=20, name="OOS Sharpe"))
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=20, r=20, t=20, b=20),
                      xaxis_title="OOS Sharpe", yaxis_title="Count")
    return fig.to_html(include_plotlyjs=False, full_html=False, div_id="cpcv-chart")


def _per_year_table(equity: pd.Series) -> str:
    r = equity.pct_change().dropna()
    if r.empty:
        return "<p class='note'>No returns.</p>"
    yearly = (1.0 + r).groupby(r.index.year).prod() - 1.0
    df = yearly.to_frame("Return")
    df["Vol"] = r.groupby(r.index.year).std(ddof=1) * np.sqrt(252)
    df["Sharpe"] = r.groupby(r.index.year).apply(
        lambda s: (s.mean() / s.std(ddof=1)) * np.sqrt(252) if s.std(ddof=1) > 0 else float("nan")
    )
    df = df.map(lambda v: format(v, ".2%") if abs(v) < 5 and isinstance(v, float) else format(v, ".2f"))
    return df.to_html(border=0, classes="per-year")


def _stress_table(stress: list[StressResult]) -> str:
    if not stress:
        return "<p class='note'>No stress windows evaluated.</p>"
    rows = pd.DataFrame([s.__dict__ for s in stress])
    rows["total_return"] = rows["total_return"].map(lambda v: format(v, ".2%") if np.isfinite(v) else "n/a")
    for c in ("sharpe", "max_drawdown", "expected_shortfall_95"):
        rows[c] = rows[c].map(lambda v: format(v, ".2f") if np.isfinite(v) else "n/a")
    return rows.to_html(border=0, index=False)


def _factor_table(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "<p class='note'>Factor attribution not computed.</p>"
    return df.to_html(border=0, float_format=lambda v: f"{v:.4f}")


def build_tearsheet(data: TearsheetData, output_path: Path) -> Path:
    """Render the tearsheet to ``output_path`` and return it.

    Parameters
    ----------
    data : TearsheetData
    output_path : Path
        Destination for the HTML file.
    """
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    tpl = env.get_template("tearsheet.html.j2")

    r = data.equity.pct_change().dropna()
    kpis = [
        _kpi("CAGR", annualized_return(r), ".2%"),
        _kpi("Volatility", annualized_volatility(r), ".2%"),
        _kpi("Sharpe", sharpe_ratio(r)),
        _kpi("Deflated Sharpe", deflated_sharpe(r, n_trials=data.n_trials_for_dsr), ".3f",
             sub=f"n_trials={data.n_trials_for_dsr}"),
        _kpi("Sortino", sortino_ratio(r)),
        _kpi("Calmar", calmar(r)),
        _kpi("Max DD", max_drawdown(r), ".2%"),
        _kpi("Time underwater", float(time_underwater(r)), ".0f", sub="bars"),
        _kpi("Hit rate", hit_rate(r), ".2%"),
        _kpi("Profit factor", profit_factor(r)),
        _kpi("PSR vs zero", probabilistic_sharpe(r, benchmark_sr=0.0), ".3f"),
    ]
    if data.turnover is not None and not data.turnover.empty:
        kpis.append(_kpi("Avg turnover", float(data.turnover.mean()), ".2f", sub="rebalance"))

    out = tpl.render(
        run_name=data.run_name,
        description=data.description,
        run_id=data.run_id,
        config_hash=data.config_hash,
        config_yaml=data.config_yaml,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        kpis=kpis,
        equity_chart=_equity_chart(data.equity, data.benchmark_equity),
        drawdown_chart=_drawdown_chart(data.equity),
        rolling_chart=_rolling_chart(data.equity, data.benchmark_equity),
        return_dist_chart=_dist_chart(data.equity),
        per_year_table=_per_year_table(data.equity),
        factor_table=_factor_table(data.factor_table),
        stress_table=_stress_table(data.stress or historical_stress(r)),
        cpcv_chart=_cpcv_chart(data.cpcv_sharpes),
        version=quantforge.__version__,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(out, encoding="utf-8")
    return output_path


__all__ = ["TearsheetData", "build_tearsheet", "summarize_results"]
