.PHONY: help install lint format type test cov backtest tearsheet docs validate-strategy clean all

PY ?= python
VENV ?= .venv

help:
	@echo "QuantForge make targets:"
	@echo "  install            Install package + dev deps into .venv"
	@echo "  lint               Run ruff check"
	@echo "  format             Run black + ruff format + isort"
	@echo "  type               Run mypy strict"
	@echo "  test               Run pytest (unit + property)"
	@echo "  cov                Run pytest with coverage report"
	@echo "  backtest CFG=...   Run a config-driven backtest"
	@echo "  tearsheet CFG=...  Build tearsheet for a config"
	@echo "  docs               Build Sphinx docs"
	@echo "  validate-strategy STRATEGY=... Full validation protocol"
	@echo "  clean              Remove caches and build artifacts"
	@echo "  all                install lint type test backtest tearsheet"

install:
	uv venv $(VENV) --python 3.11
	uv pip install --python $(VENV)/bin/python -e ".[dev,docs]"

lint:
	$(VENV)/bin/ruff check quantforge tests

format:
	$(VENV)/bin/ruff format quantforge tests
	$(VENV)/bin/isort quantforge tests --profile black --line-length 100

type:
	$(VENV)/bin/mypy quantforge

test:
	$(VENV)/bin/pytest tests/unit tests/property -q

cov:
	$(VENV)/bin/pytest --cov=quantforge --cov-report=term-missing --cov-report=html tests/unit tests/property

backtest:
	$(VENV)/bin/quantforge run-config $(CFG)

tearsheet:
	$(VENV)/bin/quantforge tearsheet $(CFG)

validate-strategy:
	$(VENV)/bin/quantforge validate $(STRATEGY)

docs:
	$(VENV)/bin/sphinx-build -b html docs docs/_build/html

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build dist *.egg-info docs/_build

all: install lint type test
	$(VENV)/bin/quantforge run-config configs/multi_strategy_blend.yaml
	$(VENV)/bin/quantforge tearsheet configs/multi_strategy_blend.yaml
