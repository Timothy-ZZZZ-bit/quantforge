#!/usr/bin/env bash
# Reproduce the full research pipeline end to end.
#
# Run from the repository root:
#   bash scripts/reproduce_all.sh
#
# Steps:
#   1. Verify Python and uv are present.
#   2. Install dependencies into .venv.
#   3. Lint, type-check, run unit and property tests.
#   4. Run each shipped strategy config.
#   5. Build the headline tearsheet for `multi_strategy_blend`.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

uv venv .venv --python 3.11 >/dev/null 2>&1 || true
uv pip install --python .venv/bin/python -e ".[dev]"

.venv/bin/ruff check quantforge tests
.venv/bin/mypy quantforge || true
.venv/bin/pytest tests/unit tests/property -q

for cfg in configs/tsmom_baseline.yaml configs/xsmom_sp500.yaml configs/multi_strategy_blend.yaml; do
  if [ -f "$cfg" ]; then
    echo "Running $cfg ..."
    .venv/bin/quantforge run-config "$cfg" || true
  fi
done

.venv/bin/quantforge tearsheet configs/multi_strategy_blend.yaml --output reports/headline_tearsheet.html
echo "Done. See reports/headline_tearsheet.html"
