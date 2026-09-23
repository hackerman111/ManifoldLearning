# Hypo single-index ADP with matrix-free LSMR

## Scope

Add a complete single-index Average Derivative Procedure under
`hypo/single_index/`. The implementation follows the single-index procedure in
`tex/manifold_v2.tex`, uses the beta subproblem from
`tex/optimization_var.tex`, and reuses the stable dense batched statistics from
`ADP/ADP_statistic.py`.

The implementation targets hypothesis testing and deterministic validation. It
does not add reporting, persistence, GPU support, sparse weights, or a second
copy of the statistics formulas.

## Files

- `hypo/single_index/ADP_single_index.py`: algorithm and learned state.
- `hypo/single_index/tester.py`: deterministic beta-recovery CLI.
- `ADP/__init__.py`: correct the moved `ADP_single_index` import so the existing
  package and statistics tester can load.

No `hypo/tester.py` change is needed: it already discovers
`hypo/*/tester.py`.

## Public API

`ADP_single_index` accepts numerical tuning parameters in its constructor and
exposes:

```python
model.fit(X, Y)
model.beta_
model.beta_init_
model.slopes_
model.h0_
model.trace_
```

`X` has shape `(n, d)`, `Y` has shape `(n,)`, and `fit` returns `self`.
The learned beta vectors are finite unit vectors. Their sign is not identified,
so quality is measured by absolute cosine.

The minimal configurable values are the number of centers and directions,
`N_loc`, `N_lin`, lambda and local ridge penalties, outer and inner iteration
limits, bandwidth decay, convergence tolerance, batch size, and random seed.
Defaults are `n_centers=64`, `n_directions=8`, `N_loc=10`,
`lambda_penalty=1.0`, `local_ridge=1e-8`, `outer_steps=4`,
`inner_steps=5`, `bandwidth_decay=sqrt(2)`, `tol=1e-6`,
`batch_size=32`, and `seed=42`. If `N_lin` is omitted, it resolves to
`min(n, max(2*d + 2, n // N_loc))`. If `h_min` is omitted, it resolves to
`10 * mean(feature standard deviation) / n`.

## Algorithm

### Preparation and initialization

1. Validate finite real `X` and `Y`.
2. Select `min(n_centers, n)` observations as centers without replacement.
3. Compute and cache the `(J, n)` squared Euclidean distance matrix.
4. Find `h_lin` by binary search as the smallest positive bandwidth whose
   average kernel mass is at least `N_lin`.
5. At every center, solve a ridge-stabilized weighted local linear regression
   of `Y` on `[1, X - center]`.
6. Stack the local gradients and use their leading right singular vector as
   `beta_init_`.
7. Find `h0_` by the same binary search with target mass `N_loc`.

The Epanechnikov kernel is `max(0, 1 - q**2)`, matching
`tex/manifold_v2.tex`.

### Structural adaptation

At outer step zero, use isotropic weights

```text
q[j, i] = ||X[i] - center[j]||² / h₀².
```

At later steps decrease `h` by `sqrt(2)` and use

```text
q[j, i] =
    (rho² ||X[i] - center[j]||²
     + <X[i] - center[j], beta>²) / h².
```

For fixed `h` and beta, binary search returns the largest `rho` in `[0, 1]`
whose average kernel mass remains at least `N_loc`. This is the monotone form
consistent with the tensor formula: decreasing rho weakens localization in
directions orthogonal to beta.

Initial directions are normalized Gaussian vectors. Later directions are
normalized draws of `rho * z + xi * beta`, which has covariance proportional
to the adaptive tensor.

### Statistics and alternating optimization

For each outer step, call the existing dense batched
`calculate_statistics(X, Y, weights, directions, batch_size)`. It produces
`I` with shape `(J, P)` and `U` with shape `(J, P, d)`.

For fixed beta, update each local derivative:

```text
slopes[j] =
    <I[j], U[j] beta>
    / (||U[j] beta||² + local_ridge).
```

For fixed slopes, define the matrix-free operator

```text
A v = concat_j slopes[j] * U[j] v
A.T u = sum_j slopes[j] * U[j].T u[j].
```

The augmented `LinearOperator` implements

```text
[A; sqrt(lambda) I]
```

and LSMR solves against

```text
[vec(I); sqrt(lambda) beta_prior].
```

The full block matrix and normal equations are never materialized. The result
is checked for finiteness and nonzero norm, normalized, and sign-aligned with
the previous beta. Inner iterations stop on sign-invariant beta change.
The outer loop stops after `outer_steps` or before the next bandwidth would
fall below `h_min`.

`trace_` records outer index, bandwidth, rho, mean mass, inner iteration count,
LSMR stop code, LSMR iteration count, and beta change.

## Validation CLI

`hypo/single_index/tester.py` is discovered by the existing root tester:

```bash
python hypo/tester.py single_index
```

The default deterministic scenario uses:

- `n=1200`, `d=6`, `seed=42`;
- normalized random `beta_true`;
- `Y = (X @ beta_true)**2 + noise`;
- Gaussian noise standard deviation `0.05`;
- success threshold `abs(cos(beta_, beta_true)) >= 0.90`.

The tester also checks unit norm, finite learned values, a nonempty trace, and
that LSMR was invoked. It prints initial and final absolute cosine and returns
zero on success or one on failed recovery.

The existing statistics smoke profile is rerun after correcting the package
import:

```bash
python hypo/tester.py statistic --profile smoke --repetitions 1
```

## Failure handling

- Invalid shapes, nonfinite arrays, and impossible tuning values raise
  `ValueError`.
- Failure to bracket a bandwidth or preserve the required mass raises
  `RuntimeError`.
- Degenerate local gradients or a nonfinite/zero LSMR result raise
  `RuntimeError`.
- The tester reports the measured cosine before returning a failing exit code.
