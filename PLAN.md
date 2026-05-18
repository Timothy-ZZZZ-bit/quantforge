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

- [x] All tests pass (164 tests); coverage 91% on `quantforge/`.
- [x] mypy strict passes with zero errors (see ADR-006 for the two
      stub-fighting flags relaxed).
- [x] Ruff and Black clean.
- [x] 5 strategy configs in `configs/` produce non-trivial OOS results.
- [x] Headline multi-strategy tearsheet committed as HTML; equity-curve PNG
      committed. PDF export is an optional `weasyprint`-gated extra.
- [x] Headline narrative is honest: synthetic-universe DSR is near zero,
      which is the validation protocol working as designed. See README and
      RESEARCH_NOTES.md.
- [x] All 6 methodology docs written with math and references.
- [x] Validation protocol implemented (CPCV, walk-forward, DSR, no-lookahead
      property test, cost sensitivity).
- [x] README renders with embedded headline figure and KPI table.
- [x] Sphinx docs scaffolded; `docs.yml` deploys to gh-pages on push.
- [ ] CI green on `main` (runs once pushed to GitHub).
- [x] `v0.1.0` tagged.
- [x] No em dashes anywhere in human-written prose.

## Manual user steps

These cannot be done by the build agent:

1. Pin the repo on the GitHub profile.
2. Enable GitHub Pages from `gh-pages` branch in repo settings.
3. Authenticate `gh` CLI to push to GitHub.
