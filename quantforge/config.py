"""Pydantic-validated configuration objects for QuantForge runs.

A run configuration is a single YAML file that fully specifies a backtest:
universe, signal stack, allocator, execution model, and validation protocol.
The configuration is hashed and committed to the research log so that every
result can be linked back to the exact specification that produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from quantforge.constants import (
    DEFAULT_BORROW_BPS_ANNUAL,
    DEFAULT_COMMISSION_BPS,
    DEFAULT_CPCV_GROUPS,
    DEFAULT_CPCV_TEST_GROUPS,
    DEFAULT_IMPACT_COEF,
    DEFAULT_PARTICIPATION_CAP,
    DEFAULT_SLIPPAGE_BPS,
)

RebalanceFreq = Literal["D", "W", "M"]


class UniverseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Universe identifier, e.g. SPY-components or ETF-basket")
    tickers: list[str] = Field(default_factory=list)
    start: str
    end: str
    benchmark: str = "SPY"

    @field_validator("start", "end")
    @classmethod
    def _date_format(cls, v: str) -> str:
        # Light validation: pandas will fully parse later.
        if len(v) < 8:
            raise ValueError(f"date string too short: {v!r}")
        return v


class SignalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str = Field(..., description="One of: tsmom, xsmom, pairs, ou, ml, carry, quality")
    params: dict[str, Any] = Field(default_factory=dict)
    weight: float = 1.0


class AllocatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(..., description="equal_weight, mvo, erc, hrp, black_litterman")
    params: dict[str, Any] = Field(default_factory=dict)


class CostsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commission_bps: float = DEFAULT_COMMISSION_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    impact_coef: float = DEFAULT_IMPACT_COEF
    borrow_bps_annual: float = DEFAULT_BORROW_BPS_ANNUAL
    participation_cap: float = DEFAULT_PARTICIPATION_CAP


class ConstraintsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_weight: float = 0.10
    min_weight: float = -0.10
    gross_leverage: float = 1.0
    net_leverage: float | None = None
    turnover_cap: float | None = None
    sector_cap: float | None = None


class CPCVConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_groups: int = DEFAULT_CPCV_GROUPS
    n_test_groups: int = DEFAULT_CPCV_TEST_GROUPS
    embargo: int = 5


class WalkForwardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_years: int = 5
    test_years: int = 1
    mode: Literal["rolling", "expanding"] = "rolling"


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    cpcv: CPCVConfig = Field(default_factory=CPCVConfig)
    cost_multipliers: list[float] = Field(default_factory=lambda: [0.0, 1.0, 2.0, 5.0])
    bootstrap_replicates: int = 1_000


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    seed: int = 1729
    rebalance: RebalanceFreq = "M"
    universe: UniverseConfig
    signals: list[SignalConfig]
    allocator: AllocatorConfig
    costs: CostsConfig = Field(default_factory=CostsConfig)
    constraints: ConstraintsConfig = Field(default_factory=ConstraintsConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    def config_hash(self) -> str:
        """Stable hash of the full configuration."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a YAML configuration file.

    Parameters
    ----------
    path : str | Path
        Path to a YAML run config.

    Returns
    -------
    RunConfig
        Validated configuration.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return RunConfig.model_validate(raw)


__all__ = [
    "AllocatorConfig",
    "CPCVConfig",
    "ConstraintsConfig",
    "CostsConfig",
    "RunConfig",
    "SignalConfig",
    "UniverseConfig",
    "ValidationConfig",
    "WalkForwardConfig",
    "load_config",
]
