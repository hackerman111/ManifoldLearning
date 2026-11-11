# Conjugate guard S1 checks

Read-only validation of the isolated PR+ prototype against PLAN.md S1. No pilot or R experiment was run.

## Verification

- `UV_CACHE_DIR=/tmp/adp-uv-cache rtk proxy uv run --no-sync pytest -q tests/test_grassman_conjugate_guard.py tests/test_grassman_spectral_guard.py tests/test_grassman_intrinsic.py tests/test_grassman_armijo.py tests/test_grassman_quadratic.py` — **31 passed in 0.79s**.
- `rtk ruff check experiments/grassman_accuracy_2026_10_18/backtracking.py experiments/grassman_accuracy_2026_10_18/conjugate_guard.py tests/test_grassman_conjugate_guard.py` — clean.
- `UV_CACHE_DIR=/tmp/adp-uv-cache rtk proxy timeout 30s uv run --no-sync pyright experiments/grassman_accuracy_2026_10_18/backtracking.py experiments/grassman_accuracy_2026_10_18/conjugate_guard.py tests/test_grassman_conjugate_guard.py` — timed out with exit 124 and no diagnostics; type check is unconfirmed.

## Contract review

- Projection transport is `H - Y(Y.T @ H)` for both prior gradient and direction. PR+ uses `<G, G - T(G_old)> / ||G_old||_F^2`, clamps beta at zero, and falls back to steepest descent for invalid history or if the actual proposed direction misses the strict `1e-3` cosine descent margin.
- The selected search vector is SVD-truncated to the current rank-q subspace. The Armijo slope is computed from the current full gradient and the actual rank-q geodesic velocity, divided by its leading singular value; trials are skipped unless this slope is negative. The centered finite-difference test independently agrees with this slope.
- History is retained only after an accepted nonsmooth step and is compared against that step's local-rank pattern. A changed pattern passes `None` history to the next direction; a first step also matches the spectral baseline.
- Every trial profiles all centers. The final profile is recomputed from live `U @ Y`; final coefficients, objective, local-rank loss, stationarity applicability, and gradient certificate use that refit. Direct least-squares checks agree with the returned result.
- Smooth-path output matches production over the tested five-step budget. No source-level correctness blocker found in the reviewed contracts; this is focused small-case evidence, not proof of convergence or a pilot result.

## Exploratory 90800 pilot

Ran once with seed 90800, shared method seed 1173550236, float64, single-thread BLAS; per-case ratios use `projector_distance` relative to `grassman5`. Orthogonality is the maximum of returned basis and inner-solver errors.

| Budget | Case | EDR error ratio | Fit time ratio | Max orthogonality |
|---:|:---|---:|---:|---:|
| 5 | d-5 | 1.48294 | 0.44166 | 3.32e-16 |
| 5 | d-20 | 0.93962 | 0.55456 | 3.95e-16 |
| 5 | d-50 | 1.89113 | 1.03640 | 7.85e-16 |
| 10 | d-5 | 1.06898 | 0.86549 | 4.74e-16 |
| 10 | d-20 | 0.91697 | 0.92183 | 1.05e-15 |
| 10 | d-50 | 0.84856 | 1.21472 | 1.24e-15 |
| 15 | d-5 | 1.06577 | 1.25522 | 7.49e-16 |
| 15 | d-20 | 0.93972 | 1.28427 | 8.86e-16 |
| 15 | d-50 | 0.90907 | 1.61857 | 1.02e-15 |

| Budget | EDR geometric mean | Median time ratio | Max EDR ratio | Accepted inner steps | Profile evaluations |
|---:|---:|---:|---:|---:|---:|
| 5 | 1.38123 | 0.55456 | 1.89113 | 70 | 235 |
| 10 | 0.94045 | 0.92183 | 1.06898 | 130 | 609 |
| 15 | 0.96922 | 1.28427 | 1.06577 | 198 | 1117 |

Pilot data and hashes are in `pilot_conjugate_guard.json`. Its production `ADP/solver/grassman.py` hash is `6d5aa753da9d9c8532545fa7519e63ba479c0f4082a8a6d52358cd4c66af72a0`, matching the expected frozen production source.
