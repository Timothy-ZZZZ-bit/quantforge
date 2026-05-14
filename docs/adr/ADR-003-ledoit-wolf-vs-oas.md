# ADR-003: Ledoit-Wolf shrinkage over OAS

## Status

Accepted.

## Context

The MVO problem requires an estimate of the asset covariance matrix.
Options considered:

- Sample covariance: dominated under expected-Frobenius loss by any
  reasonable shrinkage estimator.
- Ledoit-Wolf (2004): closed-form shrinkage intensity, well understood,
  dominates the sample covariance.
- Oracle Approximating Shrinkage (OAS): slightly better than Ledoit-Wolf
  under Gaussian assumptions.

## Decision

Use Ledoit-Wolf, exposed via `sklearn.covariance.LedoitWolf`.

## Rationale

- Ledoit-Wolf has the longer track record and is the buy-side default.
  Citing OAS in a tearsheet draws follow-up questions that LW does not.
- The marginal improvement of OAS depends on the Gaussianity assumption,
  which is exactly the assumption we know to be wrong for daily equity
  returns.
- Both ship with scikit-learn; the implementation effort is identical.

## Consequences

If a future workflow demands a heavier-tailed shrinkage target, the
implementation point of change is `quantforge/portfolio/mvo.py` and
`risk_parity.py`. The API of `LedoitWolf().fit(X).covariance_` is shared
across scikit-learn shrinkage estimators, so swapping is a one-line
change.
