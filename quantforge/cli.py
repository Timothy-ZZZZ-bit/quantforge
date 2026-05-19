"""Typer-driven CLI: ``quantforge run-config configs/foo.yaml``."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import yaml

from quantforge.backtest.engine import BacktestResult
from quantforge.config import RunConfig, load_config
from quantforge.constants import REPORTS_DIR, RESEARCH_LOG_PATH, RUN_ARTIFACTS_DIR
from quantforge.data import load_equity_panel, synthetic_equity_panel
from quantforge.execution import CostModel
from quantforge.logging import get_logger, new_run_id
from quantforge.portfolio.base import Allocator
from quantforge.reporting import TearsheetData, build_tearsheet
from quantforge.signals.base import Signal

app = typer.Typer(help="QuantForge command-line interface")
_log = get_logger(__name__)


def _load_panel(cfg: RunConfig) -> pd.DataFrame:
    if cfg.universe.name.lower().startswith("synthetic"):
        n = int(cfg.universe.tickers[0]) if cfg.universe.tickers else 10
        return synthetic_equity_panel(n_tickers=n, n_days=252 * 5, seed=cfg.seed)
    panel = load_equity_panel(
        tickers=cfg.universe.tickers + [cfg.universe.benchmark],
        start=cfg.universe.start,
        end=cfg.universe.end,
    )
    return panel


def _build_signal_stack(cfg: RunConfig) -> list[tuple[float, Signal]]:
    from quantforge.signals import (
        CrossSectionalMomentum,
        OUMeanReversion,
        PairsTrade,
        QualityFactor,
        TimeSeriesMomentum,
    )

    factories = {
        "tsmom": TimeSeriesMomentum,
        "xsmom": CrossSectionalMomentum,
        "ou": OUMeanReversion,
        "pairs": PairsTrade,
        "quality": QualityFactor,
    }
    stack: list[tuple[float, Signal]] = []
    for s in cfg.signals:
        cls = factories.get(s.kind)
        if cls is None:
            raise ValueError(f"unknown signal kind: {s.kind!r}")
        stack.append((s.weight, cls(**s.params)))
    return stack


def _build_allocator(cfg: RunConfig) -> Allocator:
    from quantforge.portfolio import (
        BlackLittermanAllocator,
        EqualRiskContribution,
        EqualWeight,
        HRPAllocator,
        MeanVariance,
    )

    factories = {
        "equal_weight": EqualWeight,
        "mvo": MeanVariance,
        "erc": EqualRiskContribution,
        "hrp": HRPAllocator,
        "black_litterman": BlackLittermanAllocator,
    }
    cls = factories.get(cfg.allocator.kind)
    if cls is None:
        raise ValueError(f"unknown allocator: {cfg.allocator.kind!r}")
    return cls(**cfg.allocator.params)


def _benchmark_equity(panel: pd.DataFrame, benchmark: str) -> pd.Series:
    sub = panel[panel["ticker"] == benchmark].sort_values("date").set_index("date")["adj_close"]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub / sub.iloc[0] * 1_000_000.0


def _run_one(cfg: RunConfig, panel: pd.DataFrame, run_id: str) -> BacktestResult:  # noqa: ARG001
    from quantforge.backtest import BacktestEngine
    from quantforge.portfolio import Constraints, apply_constraints

    signals = _build_signal_stack(cfg)
    allocator = _build_allocator(cfg)
    constraints = Constraints(
        max_weight=cfg.constraints.max_weight,
        min_weight=cfg.constraints.min_weight,
        gross_leverage=cfg.constraints.gross_leverage,
        turnover_cap=cfg.constraints.turnover_cap,
    )

    wide_close = panel.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    returns_hist = pd.DataFrame(
        np.log((wide_close / wide_close.shift(1)).to_numpy()),
        index=wide_close.index,
        columns=wide_close.columns,
    )

    bench = cfg.universe.benchmark
    trading_panel = panel[panel["ticker"] != bench]

    prev_weights = pd.Series(dtype=float)

    def weight_fn(date: pd.Timestamp, visible: pd.DataFrame) -> dict[str, float]:
        # Blend signals with configured weights, then run allocator + constraints.
        blended: pd.Series = pd.Series(dtype=float)
        for w, sig in signals:
            sig.fit(visible)
            alpha = sig.predict(visible)
            blended = blended.add(w * alpha, fill_value=0.0)
        if blended.empty:
            return {}
        hist_slice = returns_hist.loc[:date].iloc[-126:]
        weights = allocator.allocate(blended.dropna(), hist_slice)
        if weights.empty:
            return {}
        nonlocal prev_weights
        weights = apply_constraints(weights, constraints, prior_weights=prev_weights)
        prev_weights = weights.copy()
        return {str(k): float(v) for k, v in weights.to_dict().items()}

    engine = BacktestEngine(
        trading_panel,
        weight_fn,
        cost_model=CostModel(
            commission_bps=cfg.costs.commission_bps,
            slippage_bps=cfg.costs.slippage_bps,
            impact_coef=cfg.costs.impact_coef,
            borrow_bps_annual=cfg.costs.borrow_bps_annual,
        ),
        initial_cash=1_000_000.0,
        rebalance_freq=(
            "BMS" if cfg.rebalance == "M" else ("W-FRI" if cfg.rebalance == "W" else "D")
        ),
        participation_cap=cfg.costs.participation_cap,
    )
    return engine.run()


@app.command("run-config")
def run_config(config_path: Path) -> None:
    """Execute a backtest from a YAML config and emit run artifacts."""
    cfg = load_config(config_path)
    run_id = new_run_id()
    panel = _load_panel(cfg)
    _log.info("run.start", run_id=run_id, config=cfg.name, hash=cfg.config_hash())
    result = _run_one(cfg, panel, run_id)
    artifact_dir = RUN_ARTIFACTS_DIR / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result.equity.to_frame("equity").to_parquet(artifact_dir / "equity.parquet")
    if not result.weights.empty:
        result.weights.to_parquet(artifact_dir / "weights.parquet")
    if not result.turnover.empty:
        result.turnover.to_frame("turnover").to_parquet(artifact_dir / "turnover.parquet")
    (artifact_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    # Append to the research log.
    RESEARCH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESEARCH_LOG_PATH.open("a", encoding="utf-8") as fh:
        rec = {
            "run_id": run_id,
            "config_name": cfg.name,
            "config_hash": cfg.config_hash(),
            "timestamp": datetime.now(UTC).isoformat(),
            "n_bars": len(result.equity),
            "final_equity": (float(result.equity.iloc[-1]) if len(result.equity) else float("nan")),
        }
        fh.write(json.dumps(rec) + "\n")
    typer.echo(f"run_id={run_id} artifacts={artifact_dir}")


@app.command("tearsheet")
def tearsheet(config_path: Path, output: Path | None = None) -> None:
    """Run a backtest and produce an HTML tearsheet."""
    cfg = load_config(config_path)
    run_id = new_run_id()
    panel = _load_panel(cfg)
    result = _run_one(cfg, panel, run_id)
    out = output or REPORTS_DIR / f"{cfg.name}_tearsheet.html"
    data = TearsheetData(
        run_name=cfg.name,
        description=cfg.description,
        run_id=run_id,
        config_hash=cfg.config_hash(),
        config_yaml=yaml.safe_dump(cfg.model_dump(mode="json")),
        equity=result.equity,
        benchmark_equity=_benchmark_equity(panel, cfg.universe.benchmark),
        turnover=result.turnover,
        weights=result.weights,
        n_trials_for_dsr=_count_research_log_trials(),
    )
    path = build_tearsheet(data, out)
    typer.echo(str(path))


def _count_research_log_trials() -> int:
    if not RESEARCH_LOG_PATH.exists():
        return 1
    return max(1, sum(1 for _ in RESEARCH_LOG_PATH.open("r", encoding="utf-8")))


@app.command("validate")
def validate(strategy: str) -> None:
    """Validation protocol stub. See ``docs/methodology/05_*.md``."""
    typer.echo(f"validate {strategy!r}: see docs/methodology/05_validation_and_overfitting.md")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
