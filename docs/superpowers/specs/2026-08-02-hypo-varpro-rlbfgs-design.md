# Hypo single-index VarPro with Riemannian L-BFGS

## Goal

Add a solver variant under `hypo/varpro/` that keeps the current
`hypo/single_index` data preparation, structural-adaptation loop, statistics,
benchmark scenario, and telemetry, but replaces alternating slopes/LSMR with
the variable-projection Riemannian L-BFGS algorithm from
`tex/riemannian_lbfgs_varpro.tex`.

The existing `hypo/single_index/` implementation remains unchanged and serves
as the LSMR baseline.

## Files and API

- `hypo/varpro/ADP_single_index.py` subclasses the current single-index class
  and overrides only the solver boundary.
- `hypo/varpro/tester.py` keeps the current CLI arguments, generated data,
  recovery threshold, timing output, and weight-density checks. It imports the
  VarPro class and checks solver-specific diagnostics.

`hypo/tester.py` needs no change because it discovers every
`hypo/*/tester.py`; the new variant is invoked as:

```bash
python hypo/tester.py varpro --seed 42
```

The model keeps `fit(X, Y)`, `beta_`, `beta_init_`, `slopes_`, `h0_`,
`trace_`, `timings_`, `weight_density_`, and
`mean_zero_weight_fraction_`.

## Solver

For fixed statistics `I` with shape `(J, P)` and `U` with shape `(J, P, d)`,
the solver evaluates

```text
q_j(beta) = U_j beta
a_j       = <I_j, q_j>
b_j       = ||q_j||² + local_ridge
slope_j   = a_j / b_j
F(beta)   = 1/2 sum_j (||I_j - slope_j q_j||²
                       + local_ridge * slope_j²)
```

The Euclidean gradient is

```text
-sum_j slope_j U_j.T (I_j - slope_j U_j beta),
```

and its projection onto the tangent space of the unit sphere is

```text
grad F = euclidean_gradient
         - beta * <beta, euclidean_gradient>.
```

The scalar `local_ridge` is the common `eta_j` from the TeX derivation.
`lambda_penalty` remains accepted by the inherited constructor for API
compatibility but is not part of the VarPro objective.

Each outer ADP step starts Riemannian L-BFGS from the preceding unit beta:

1. Apply the standard limited-memory two-loop recursion.
2. Fall back to `-grad F` if the computed direction is not descending.
3. Use Armijo backtracking with `c1=1e-4`, shrink factor `0.5`, and the
   normalized sphere retraction.
4. Project the stored tangent pairs to the new tangent space.
5. Add the new pair only when
   `<s, y> > 1e-10 * ||s|| * ||y||`.
6. Stop on Riemannian-gradient norm, relative objective change, or the
   inherited `inner_steps` iteration limit.
7. Recover slopes analytically and align the final beta sign with the starting
   beta.

The variant adds only `lbfgs_memory` (default `10`) and changes the inherited
default `inner_steps` to `50`. No manifold dependency is added.

## Diagnostics and failure handling

Each outer trace record contains:

- `solver_status`: `gradient`, `objective`, `max_iterations`, or
  `line_search_failed`;
- `solver_iterations`;
- final profiled `objective`;
- final `gradient_norm`;
- total `line_search_steps`;
- sign-invariant `beta_delta`.

The inherited fit timing slot used while the parent loop runs is renamed from
`lsmr` to `rlbfgs` before `fit` returns. The obsolete standalone `slopes`
timing is omitted because slope elimination is part of every VarPro
function/gradient evaluation.

Invalid L-BFGS memory raises `ValueError`. Nonfinite objective, gradient,
slopes, retraction, or direction raises `RuntimeError`. A failed Armijo search
is reported as a technical solver status without discarding the last valid
iterate.

## Verification

The tester performs:

- a central finite-difference check of the analytical Riemannian gradient
  along a tangent direction;
- the same seeded recovery benchmark as `hypo/single_index/tester.py`;
- finite unit-beta, trace, solver-diagnostic, timing, and weight-density checks.

Both variants are run with the identical explicit seed for a paired result.
The existing statistics smoke tester is rerun to catch import or shared
statistics regressions.
