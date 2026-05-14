# QuantForge Build Plan

## Overview

QuantForge is a production-grade systematic trading research platform demonstrating
the full quant research lifecycle. This file tracks build progress against the
spec in `the original build specification`.

## Phase breakdown

- [x] **Phase 0: Scaffolding.** Repo init, pyproject, pre-commit, ruff, mypy, CI, package tree.
- [ ] **Phase 1: Data layer.** loaders, universe, quality audit; tests; ADR on survivorship bias.
- [ ] **Phase 2: Features.** returns, vol, frac-diff, cross-sectional ops, triple-barrier.
- [ ] **Phase 3: Backtest engine.** event-driven core, walk-forward, no-lookahead property tests.
- [ ] **Phase 4: Signals.** TSMOM, XSMOM, pairs, OU mean reversion, ML signal.
- [ ] **Phase 5: Portfolio construction.** EW, MVO+LW, ERC, HRP, BL.
- [ ] **Phase 6: Risk and metrics.** VaR/CVaR, drawdown, sizing, stress; performance, PSR, DSR.
- [ ] **Phase 7: CPCV and validation harness.**
- [ ] **Phase 8: Reporting.** HTML + PDF tearsheet.
- [ ] **Phase 9: Multi-strategy blend.** Headline result tearsheet.
- [ ] **Phase 10: Documentation polish.** README, methodology docs, Sphinx site.
- [ ] **Phase 11: Release.** v0.1.0 tag, PDF tearsheet asset.

## Module dependency graph

```
data -> features -> signals ----+
                                |--> backtest -> metrics -> reporting
                  portfolio ----+
                  execution ----+
                  risk    ------+
```

## Architectural decisions

Tracked as ADRs in `docs/adr/`. Each non-trivial choice gets its own one-page ADR.

## Final acceptance checklist

- [ ] `make all` runs from clean clone in under 30 minutes.
- [ ] All tests pass; coverage >= 90% on `quantforge/`.
- [ ] mypy strict passes with zero errors.
- [ ] Ruff and Black clean.
- [ ] At least 5 working strategy configs produce non-trivial OOS results.
- [ ] Headline multi-strategy tearsheet committed as HTML and PDF.
- [ ] Headline DSR > 0 at a credible n_trials, or honest negative-result narrative.
- [ ] All 6 methodology docs written with math and references.
- [ ] All 9 validation protocol items pass for headline config.
- [ ] README renders correctly with embedded figure and KPI table.
- [ ] Sphinx docs build and deploy to gh-pages.
- [ ] CI green on `main`.
- [ ] `v0.1.0` tagged with PDF tearsheet attached.
- [ ] No em dashes anywhere in human-written prose.

## Manual user steps

These cannot be done by the build agent:

1. Pin the repo on the GitHub profile.
2. Enable GitHub Pages from `gh-pages` branch in repo settings.
3. Authenticate `gh` CLI to push to GitHub.
