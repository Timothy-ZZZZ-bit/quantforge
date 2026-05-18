"""Structured logging configured once per process.

Every backtest run obtains a unique ``run_id`` (UUID4) that is bound into
the logger and emitted on every record, so logs from concurrent runs can be
demultiplexed downstream.
"""

from __future__ import annotations

import logging
import sys
import uuid
from typing import Any

import structlog

_CONFIGURED: bool = False


def configure(level: int = logging.INFO) -> None:
    """Configure structlog once. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def new_run_id() -> str:
    """Generate a fresh run identifier."""
    return uuid.uuid4().hex[:12]


def get_logger(name: str | None = None, **bound: Any) -> structlog.stdlib.BoundLogger:
    """Return a bound logger with optional contextual fields."""
    configure()
    log = structlog.get_logger(name)
    if bound:
        log = log.bind(**bound)
    return log


__all__ = ["configure", "get_logger", "new_run_id"]
