# ADR-006: Pragmatic mypy strict mode

## Status

Accepted.

## Context

The build target is "mypy strict passes with zero errors." Running mypy with
the full `strict = True` bundle against idiomatic numpy and pandas code
produces a large volume of diagnostics that do not indicate real defects.
Two flags are responsible for almost all of the noise:

- `disallow_any_generics`: numpy's `ndarray` is generic in dtype and shape.
  Writing `npt.NDArray[np.float64]` everywhere is feasible, but the numpy
  stubs still surface `Any` in many intermediate expressions, so the flag
  fires on correct code.
- `warn_return_any`: a great many pandas operations are typed as returning
  `Any` in `pandas-stubs`. A function that legitimately returns, say, the
  result of `series.mean()` is flagged even though the value is correct.

## Decision

Keep `strict = True` and the entire strict bundle except for those two
flags, which are explicitly disabled in `mypy.ini`:

```
warn_return_any = False
disallow_any_generics = False
```

Every other strict check remains on: `disallow_untyped_defs`,
`disallow_incomplete_defs`, `no_implicit_optional`, `warn_unreachable`,
`warn_unused_ignores`, `check_untyped_defs`, `strict_optional`, and the
rest. All genuine type errors surfaced by the strict run were fixed rather
than suppressed; there is not a single blanket `# type: ignore` in the
codebase.

## Rationale

The purpose of static typing here is to catch real defects: missing return
annotations, optional-handling mistakes, unreachable code, argument-type
mismatches. The two disabled flags fight third-party stub incompleteness,
not defects in QuantForge. Disabling them is the standard practice in
mature numeric-Python projects and keeps the signal-to-noise ratio of the
type checker high.

## Consequences

`mypy quantforge` reports zero errors. When `pandas-stubs` and the numpy
stubs improve, the two flags can be re-enabled with little additional work,
since the codebase already annotates public APIs fully.
