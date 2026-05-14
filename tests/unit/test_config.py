from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from quantforge.config import RunConfig, load_config


def test_load_config_roundtrip(tmp_path: Path):
    yaml_text = """
name: t
description: test
seed: 7
rebalance: M
universe:
  name: synthetic-test
  tickers: ["10"]
  start: "2020-01-02"
  end: "2024-12-31"
  benchmark: SPY
signals:
  - name: x
    kind: tsmom
    params: {}
allocator:
  kind: equal_weight
  params: {}
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_text)
    cfg = load_config(p)
    assert isinstance(cfg, RunConfig)
    assert cfg.name == "t"
    assert cfg.config_hash() == cfg.config_hash()


def test_load_config_rejects_extra_keys(tmp_path: Path):
    yaml_text = """
name: t
description: test
seed: 7
rebalance: M
unknown_key: 1
universe: { name: x, tickers: [], start: "2020-01-02", end: "2024-12-31", benchmark: SPY }
signals: []
allocator: { kind: equal_weight, params: {} }
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValidationError):
        load_config(p)
