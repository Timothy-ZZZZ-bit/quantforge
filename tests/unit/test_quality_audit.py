from __future__ import annotations

import pandas as pd

from quantforge.data import audit_panel, synthetic_equity_panel


def test_audit_clean_synthetic_panel():
    panel = synthetic_equity_panel(n_tickers=3, n_days=200, seed=0)
    report = audit_panel(panel)
    assert report.is_clean
    assert report.n_tickers == 3
    assert report.return_spike_count == 0


def test_audit_detects_missing_columns():
    df = pd.DataFrame({"date": [], "ticker": []})
    report = audit_panel(df)
    assert report.missing_columns


def test_audit_flags_zero_volume():
    panel = synthetic_equity_panel(n_tickers=2, n_days=80, seed=2)
    panel.loc[panel.index % 7 == 0, "volume"] = 0
    report = audit_panel(panel)
    assert report.zero_volume_fraction > 0.05
    assert any("zero-volume" in w for w in report.warnings)
