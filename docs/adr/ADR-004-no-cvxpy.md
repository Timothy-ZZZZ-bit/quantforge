# ADR-004: SLSQP over cvxpy for portfolio optimization

## Status

Accepted.

## Context

The shipped MVO and ERC solvers need to handle linear constraints with
quadratic objectives. The original spec called for cvxpy. We considered:

- cvxpy: clean modeling language; pulls in heavy compiled solvers (ECOS,
  OSQP, SCS) and complicates the dependency tree.
- scipy.optimize.minimize (SLSQP): in stdlib-adjacent territory; handles
  the QP fine for the dimensionalities we operate on (<= 500 names).

## Decision

Use `scipy.optimize.minimize` with SLSQP for both MVO and ERC. Keep
cvxpy in the optional-extras-only list for users who want to plug in
exotic objectives.

## Rationale

- The target universe sizes (cross-asset ETF basket, US large caps) are
  small enough that SLSQP is fast and stable.
- Removing cvxpy from the core dependency set significantly reduces
  install time on CI and the laptop the build is required to finish on
  in 30 minutes.
- The reference ERC solver (`solve_erc_slsqp`) is used in unit tests to
  validate the cyclic-coordinate-descent implementation.

## Consequences

If a future allocator requires conic constraints or exotic objectives
(e.g., CVaR optimization with chance constraints), cvxpy can be added
back as an optional dependency without breaking the core API.
