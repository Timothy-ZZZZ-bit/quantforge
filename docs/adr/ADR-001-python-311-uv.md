# ADR-001: Python 3.11 + uv

## Status

Accepted.

## Context

We need a Python runtime, a dependency manager, and a virtual-environment
manager. Options considered: `pip + venv`, `poetry`, `pdm`, `uv`.

## Decision

Python 3.11 with `uv` for environment and dependency management.

## Rationale

- 3.11 has measurable interpreter-level speedups over 3.10 on the
  pandas-heavy workload this library runs.
- `uv` is roughly an order of magnitude faster than `pip` on resolving
  and installing dependencies, which matters for a `make install` that
  must complete in under 30 minutes on a fresh clone.
- `uv` ships its own Python installer, which sidesteps the need for
  pyenv-style version management.

## Consequences

The repository depends on `uv` being installed locally. We document the
single-line installer in `README.md` and `scripts/reproduce_all.sh` checks
for its presence.
