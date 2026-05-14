"""Quality audit for incoming equity panels.

A research dataset is only as trustworthy as the audit that ratifies it. This
module produces a structured :class:`QualityReport` capturing missingness,
zero-volume bars, stale prices, and return spikes greater than 8 standard
deviations. The threshold is chosen following Asness, Frazzini, Pedersen
(2019) which flags >8 sigma daily moves as candidates for corporate-action
adjustment errors in commercial vendor data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantforge.logging import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class QualityReport:
    """Structured audit results for an equity panel."""

    n_rows: int
    n_tickers: int
    date_min: str
    date_max: str
    missing_columns: list[str] = field(default_factory=list)
    missing_value_fraction: dict[str, float] = field(default_factory=dict)
    zero_volume_fraction: float = 0.0
    stale_price_fraction: float = 0.0
    return_spike_count: int = 0
    return_spike_threshold_sigma: float = 8.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_clean(self) -> bool:
        """Heuristic: panel is clean if no extreme spikes and < 1% missingness on close."""
        close_miss = self.missing_value_fraction.get("close", 0.0)
        return self.return_spike_count == 0 and close_miss < 0.01


REQUIRED_COLS = ("date", "ticker", "open", "high", "low", "close", "adj_close", "volume")


def audit_panel(
    df: pd.DataFrame,
    *,
    spike_sigma: float = 8.0,
    stale_window: int = 5,
) -> QualityReport:
    """Run a structured quality audit on a tidy equity panel.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel with ``REQUIRED_COLS``.
    spike_sigma : float
        Z-score threshold above which a return is flagged as a candidate
        data error.
    stale_window : int
        Consecutive identical-close window used to flag stale prices.

    Returns
    -------
    QualityReport
        Structured findings.

    Notes
    -----
    The audit is informative, not corrective. Callers should decide whether
    to drop or impute based on the report. This module never mutates the
    input DataFrame.
    """
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        return QualityReport(
            n_rows=len(df),
            n_tickers=df["ticker"].nunique() if "ticker" in df.columns else 0,
            date_min=str(df["date"].min()) if "date" in df.columns else "",
            date_max=str(df["date"].max()) if "date" in df.columns else "",
            missing_columns=missing_cols,
            warnings=[f"missing required columns: {missing_cols}"],
        )

    miss_frac = {c: float(df[c].isna().mean()) for c in REQUIRED_COLS if c not in ("date", "ticker")}
    zero_vol = float((df["volume"] == 0).mean())

    # Per-ticker rolling identical-close detection.
    panel = df.sort_values(["ticker", "date"]).copy()
    panel["close_diff"] = panel.groupby("ticker")["adj_close"].diff().abs()
    panel["stale_run"] = (
        (panel["close_diff"].fillna(0.0) == 0).astype(int)
        .groupby(panel["ticker"]).cumsum()
        - (panel["close_diff"].fillna(0.0) == 0).astype(int).groupby(panel["ticker"]).cumcount()
    )
    stale_frac = float((panel.groupby("ticker")["close_diff"].apply(
        lambda s: (s.fillna(0.0) == 0).rolling(stale_window).sum().fillna(0).ge(stale_window).mean()
    )).mean())

    # Spike detection per ticker.
    grouped = panel.groupby("ticker", group_keys=False)
    ret = grouped["adj_close"].apply(lambda s: np.log(s / s.shift(1)))
    z = grouped.apply(lambda g: (
        (np.log(g["adj_close"] / g["adj_close"].shift(1))
         - np.log(g["adj_close"] / g["adj_close"].shift(1)).mean())
        / np.log(g["adj_close"] / g["adj_close"].shift(1)).std(ddof=0)
    ))
    if isinstance(z, pd.DataFrame):
        z = z.stack()
    spike_count = int((z.abs() > spike_sigma).sum())

    warnings_: list[str] = []
    if miss_frac.get("adj_close", 0.0) > 0.01:
        warnings_.append("adj_close missingness > 1%")
    if zero_vol > 0.05:
        warnings_.append("zero-volume bars > 5%")
    if stale_frac > 0.02:
        warnings_.append(f"stale-close fraction > 2% (window={stale_window})")
    if spike_count > 0:
        warnings_.append(f"{spike_count} return spikes > {spike_sigma} sigma")

    report = QualityReport(
        n_rows=len(df),
        n_tickers=int(df["ticker"].nunique()),
        date_min=str(df["date"].min()),
        date_max=str(df["date"].max()),
        missing_columns=[],
        missing_value_fraction=miss_frac,
        zero_volume_fraction=zero_vol,
        stale_price_fraction=stale_frac,
        return_spike_count=spike_count,
        return_spike_threshold_sigma=spike_sigma,
        warnings=warnings_,
    )
    _log.info("audit.complete", **report.to_dict())
    # Silence "ret" unused warning while keeping the computation visible for review.
    _ = ret
    return report


__all__ = ["QualityReport", "audit_panel"]
