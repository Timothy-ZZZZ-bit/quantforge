# Research Notes

These are my reflections after building QuantForge end to end. They are written
in the voice of a quant researcher reviewing their own work; the intended
audience is myself in six months and any interviewer trying to gauge
research maturity.

## What I set out to build

A modular library that demonstrates the full quantitative-research lifecycle
on free data, with engineering and statistical discipline that would not
embarrass me in front of a Two Sigma or Citadel interviewer. The single
most important constraint was *honesty*: the temptation in any retail quant
repo is to iterate on backtest knobs until the curve looks pretty, then
publish that curve as a finding. The validation protocol baked into this
codebase, described in [`docs/methodology/05_validation_and_overfitting.md`](docs/methodology/05_validation_and_overfitting.md),
exists specifically to neutralize that temptation.

## What worked

- **Event-driven engine with a strict no-lookahead invariant.** The engine
  receives only the panel masked to `date <= t` when computing the weight
  at time `t`. This is enforced by an assertion in the inner loop and a
  property test that permutes future bars and checks decisions are
  unchanged.
- **HRP versus naive MVO.** HRP allocates more sensibly under realistic
  covariance noise; the toy-example unit test reproduces the LdP 2016
  numbers, and the property test confirms positivity and unit sum across
  arbitrary positive-definite covariances.
- **Deflated Sharpe as the headline statistic.** This was the most useful
  decision. Raw Sharpe is easy to make look good with a few iterations;
  DSR pulls the bar up in proportion to how many strategies you have
  silently tried. The honest result on the synthetic universe is a near-
  zero DSR, which is exactly what one should expect from a small, noisy
  cross section.
- **Pydantic-validated configs.** Every run is a YAML file that the
  platform validates before doing any work, and the config is hashed and
  recorded in `research_log.jsonl` so each result is linked back to its
  spec.

## What did not work, or where I deliberately cut scope

- **True survivorship-bias-free S&P 500 reconstruction.** Building a
  point-in-time membership table from Wikipedia revision history is
  itself a research project. I shipped the `SP500History` scaffolding
  with a clear documentation of the residual bias when membership data
  is unavailable, but the headline backtests in this repository use
  cross-asset ETFs to sidestep the problem rather than solve it.
- **Live broker integration.** Out of scope from the start; the platform
  ends at backtest plus tearsheet. Adding a live execution layer would
  require careful design around order management, partial fills, and
  reconciliation that has nothing to do with research methodology.
- **PDF tearsheet via WeasyPrint.** WeasyPrint is an optional dependency
  rather than a core dependency because it pulls in heavy native
  libraries on macOS. The HTML tearsheet is the canonical artifact;
  print-to-PDF from a browser produces a high-quality file when needed.
- **ML signal sophistication.** The shipped ML signal uses gradient
  boosting on a small feature set with triple-barrier labels. A serious
  extension would add monotonic constraints, model stacking, and proper
  feature-importance attribution via SHAP. I prefer to ship a working,
  well-tested skeleton than a sprawling, half-finished model.

## What I would do next

1. **Real survivorship-bias-free universe.** Either Wikipedia history
   reconstruction or a static snapshot from a research vendor would push
   the platform to a level where US large-cap claims could be made
   responsibly.
2. **Fractional-Kelly position sizing wired into the engine.** The
   primitives are present in `quantforge.risk.sizing` but not yet
   integrated into the rebalance loop.
3. **Intraday TCA.** Daily-bar transaction costs are a coarse
   approximation; intraday TCA on real liquidity data is the natural
   next step for any strategy that turns over fast enough to care.
4. **Continuous-contract futures.** The ETF universe is a stand-in for
   the futures universe that real CTA-style TSMOM lives in. Adding a
   continuous-contract roll layer would let the existing TSMOM
   implementation generate more credible numbers.
5. **Bayesian factor model.** The Black-Litterman implementation is
   serviceable but could be replaced with a proper Bayesian factor
   inference layer that returns posterior distributions over weights.

## What this repository is not

It is not a claim that I can outperform the market with retail data and a
laptop. The honest conclusion of running the validation protocol on the
shipped configurations is that none of them produces a Deflated Sharpe
that survives a credible multiple-testing adjustment. That is the right
answer; backtest results that look heroic on free data should be presumed
overfitted until proven otherwise.

The repository is intended as a portfolio piece demonstrating that I can
build correct, validated, reproducible quantitative research infrastructure.
The methodology is the artifact, not the curve.

## Acknowledgements

The intellectual debts are documented in the References section of the
[`README`](README.md). I owe the most to Lopez de Prado's *Advances in
Financial Machine Learning*, which is the closest thing the field has to
a unified statement of how to do this work without fooling yourself.
