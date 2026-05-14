"""Generate the headline tearsheet + equity-curve PNG for the README.

Uses a deterministic synthetic universe so the script runs offline. If you
have populated the Parquet cache via ``scripts/download_data.py``, pass
``--real`` to use real ETF data instead.

Usage:
    .venv/bin/python scripts/generate_headline.py
    .venv/bin/python scripts/generate_headline.py --real
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from quantforge.backtest import BacktestEngine, CombinatorialPurgedCV  # noqa: E402
from quantforge.config import load_config  # noqa: E402
from quantforge.constants import REPORTS_DIR  # noqa: E402
from quantforge.data import (  # noqa: E402
    load_equity_panel,
    synthetic_equity_panel,
)
from quantforge.execution import CostModel  # noqa: E402
from quantforge.logging import new_run_id  # noqa: E402
from quantforge.metrics import (  # noqa: E402
    annualized_return,
    annualized_volatility,
    calmar,
    deflated_sharpe,
    sharpe_ratio,
    sortino_ratio,
)
from quantforge.portfolio import (  # noqa: E402
    Constraints,
    HRPAllocator,
    apply_constraints,
)
from quantforge.reporting import TearsheetData, build_tearsheet  # noqa: E402
from quantforge.signals import (  # noqa: E402
    CrossSectionalMomentum,
    OUMeanReversion,
    QualityFactor,
    TimeSeriesMomentum,
)


def _build_panel(use_real: bool, cfg) -> tuple[pd.DataFrame, str]:
    if use_real:
        all_tickers = cfg.universe.tickers + [cfg.universe.benchmark]
        panel = load_equity_panel(all_tickers, cfg.universe.start, cfg.universe.end)
        return panel, "real"
    return synthetic_equity_panel(
        n_tickers=len(cfg.universe.tickers), n_days=252 * 8, seed=cfg.seed
    ), "synthetic"


def _build_signals(cfg) -> list[tuple[float, object]]:
    factories = {
        "tsmom": TimeSeriesMomentum,
        "xsmom": CrossSectionalMomentum,
        "ou": OUMeanReversion,
        "quality": QualityFactor,
    }
    sigs = []
    for s in cfg.signals:
        cls = factories.get(s.kind)
        if cls is None:
            continue
        sigs.append((s.weight, cls(**s.params)))
    return sigs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "multi_strategy_blend.yaml"),
    )
    parser.add_argument("--real", action="store_true", help="use real ETF data instead of synthetic")
    parser.add_argument(
        "--reports-dir", default=str(REPORTS_DIR), help="where to write reports/headline_*.html/png"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    panel, source = _build_panel(args.real, cfg)
    out_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signals = _build_signals(cfg)
    allocator = HRPAllocator()
    constraints = Constraints(
        max_weight=cfg.constraints.max_weight,
        min_weight=cfg.constraints.min_weight,
        gross_leverage=cfg.constraints.gross_leverage,
        turnover_cap=cfg.constraints.turnover_cap,
    )

    bench_ticker = cfg.universe.benchmark
    trading_panel = panel[panel["ticker"] != bench_ticker]
    wide = trading_panel.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    rets_hist = np.log(wide / wide.shift(1))
    prev_weights = pd.Series(dtype=float)

    def weight_fn(date, visible):
        blended = pd.Series(dtype=float)
        for w_s, sig in signals:
            sig.fit(visible)
            alpha = sig.predict(visible)
            blended = blended.add(w_s * alpha, fill_value=0.0)
        if blended.empty:
            return {}
        hist_slice = rets_hist.loc[:date].iloc[-126:]
        weights = allocator.allocate(blended.dropna(), hist_slice)
        if weights.empty:
            return {}
        nonlocal prev_weights
        weights = apply_constraints(weights, constraints, prior_weights=prev_weights)
        prev_weights = weights.copy()
        return weights.to_dict()

    engine = BacktestEngine(
        trading_panel,
        weight_fn,
        cost_model=CostModel(
            commission_bps=cfg.costs.commission_bps,
            slippage_bps=cfg.costs.slippage_bps,
            impact_coef=cfg.costs.impact_coef,
            borrow_bps_annual=cfg.costs.borrow_bps_annual,
        ),
        rebalance_freq="BMS",
        participation_cap=cfg.costs.participation_cap,
    )
    print(f"running backtest on {source} panel with {len(trading_panel['ticker'].unique())} tickers")
    result = engine.run()
    r = result.returns()
    print(f"Sharpe={sharpe_ratio(r):.2f} CAGR={annualized_return(r):.2%} Vol={annualized_volatility(r):.2%}")

    # Benchmark equity curve.
    bench_panel = panel[panel["ticker"] == bench_ticker].set_index("date")["adj_close"]
    if bench_panel.empty:
        bench_eq = None
    else:
        bench_eq = bench_panel / bench_panel.iloc[0] * 1_000_000.0

    # Lightweight CPCV: distribution of per-fold Sharpe on the (in-sample) returns.
    cpcv = CombinatorialPurgedCV(r.index, n_groups=10, n_test_groups=2, embargo=5)
    cpcv_sharpes = []
    for _, test_idx in cpcv.split():
        if len(test_idx) < 30:
            continue
        cpcv_sharpes.append(sharpe_ratio(r.iloc[test_idx]))
    cpcv_series = pd.Series(cpcv_sharpes).dropna()

    # DSR with n_trials from the on-disk research log; default 25.
    research_log = Path("research_log.jsonl")
    n_trials = max(1, sum(1 for _ in research_log.open(encoding="utf-8"))) if research_log.exists() else 25

    data = TearsheetData(
        run_name=cfg.name,
        description=cfg.description,
        run_id=new_run_id(),
        config_hash=cfg.config_hash(),
        config_yaml=yaml.safe_dump(cfg.model_dump(mode="json")),
        equity=result.equity,
        benchmark_equity=bench_eq,
        turnover=result.turnover,
        weights=result.weights,
        n_trials_for_dsr=n_trials,
        cpcv_sharpes=cpcv_series,
    )
    html_path = out_dir / "headline_tearsheet.html"
    build_tearsheet(data, html_path)
    print(f"wrote {html_path}")

    # Equity-curve PNG for the README.
    fig, ax = plt.subplots(figsize=(9, 5))
    norm_eq = result.equity / result.equity.iloc[0]
    ax.plot(norm_eq.index, norm_eq, label="Multi-strategy blend", linewidth=2.0, color="#1f77b4")
    if bench_eq is not None and not bench_eq.empty:
        bnorm = bench_eq.reindex(result.equity.index).dropna()
        if not bnorm.empty:
            ax.plot(bnorm.index, bnorm / bnorm.iloc[0], label=bench_ticker, linewidth=1.5, color="#7f7f7f")
    ax.set_yscale("log")
    ax.set_title("QuantForge: multi-strategy blend OOS equity (log scale)")
    ax.set_ylabel("Wealth (normalized)")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png_path = out_dir / "headline_equity_curve.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"wrote {png_path}")

    # Summary JSON.
    summary = {
        "cagr": annualized_return(r),
        "vol": annualized_volatility(r),
        "sharpe": sharpe_ratio(r),
        "sortino": sortino_ratio(r),
        "calmar": calmar(r),
        "dsr_n_trials": n_trials,
        "dsr": deflated_sharpe(r, n_trials=n_trials),
        "source": source,
    }
    (out_dir / "headline_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'headline_summary.json'}")


if __name__ == "__main__":
    main()
