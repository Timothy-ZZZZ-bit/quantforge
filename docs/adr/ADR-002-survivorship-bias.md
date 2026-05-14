# ADR-002: Survivorship bias mitigation

## Status

Accepted with documented limitation.

## Context

Backtests run on the *current* membership of an index inherit
survivorship bias: failed companies are absent. Mitigating this requires
a point-in-time membership history. Free sources are limited; commercial
sources are out of scope.

## Decision

1. Headline strategies in this repository run on a *cross-asset ETF
   basket* whose membership has been stable for the backtest window. ETFs
   die rarely enough that survivorship effects on the chosen liquid set
   are immaterial.
2. The `SP500History` class is scaffolded with a CSV-based membership
   table. When no CSV is provided, the class falls back to the current
   Wikipedia snapshot and emits a warning in logs.
3. Any backtest run on a non-historically-corrected universe is
   conspicuously labeled in its run record so that downstream readers
   know not to take the result as a credible single-name claim.

## Rationale

- ETFs are the closest free analogue to the futures universes that real
  CTA-style TSMOM lives in.
- Building a true point-in-time history from Wikipedia revisions is
  itself a multi-week research project; we document it as future work.

## Consequences

Any claim that QuantForge can produce a tradable result on a single-name
US equity universe is currently *not* supported. The library is wired to
accept a real history file the moment one is available.
