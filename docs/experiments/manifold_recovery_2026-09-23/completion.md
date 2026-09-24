# M5: compatibility, cost and completion

The final `local_quadratic` estimator is opt-in and requires `index_dim=1`.
The original `estimator="manifold"` remains the default because the selected
correction assumes a locally quadratic scalar response and the self-only graph
removes manifold-neighbor pooling. The radial and noiseless checks validate
these two designs, not a general replacement for all manifold problems.

The full suite passes: `309 passed, 28 skipped` (CUDA device unavailable).
Ruff passes on the affected code and on `ADP experiments tests benchmarks`;
repository-wide `ruff check .` reports ten pre-existing issues only in the
untouched `test/` directory. Pyright reports zero errors and one existing
warning in `ADP/solver/legacy_lsmr.py`; `uv lock --check`, format checks and
`git diff --check` pass. The direct reference, rank-failure, public result,
`transform` and `predict` checks are in `tests/test_manifold.py`.

Fixed seed 0, `(n,d,J,P,m)=(240,4,24,10,1)`, `float64`, one BLAS thread,
five timed fits after warm-up in separate processes per estimator:

| Estimator | Median fit, s | `tracemalloc` fit peak, MiB | Process RSS high-water, MiB | RMS sine |
|---|---:|---:|---:|---:|
| Original | 0.0708 | 0.6456 | 73.57 | 0.4810 |
| Local quadratic (packed) | 0.0981 | 0.6462 | 73.70 | 0.0687 |

The variant is about 39% slower on this small task. `tracemalloc` excludes
some native allocations, and RSS high-water includes imports and warm-up;
neither measurement proves identical memory in every environment. Commands
and raw repeats are in `benchmarks/manifold_recovery_cost.py` and `cost_*.json`.
The active variant stores packed quadratic coefficients `(J,d(d+1)/2)` and
one `(n,d)` center block at a time, with a 60-observation polynomial design;
it never builds `(J,n,d)` or a dense `d×d` projector. Full quadratic rank
requires at most nine features with the fixed 60-neighbor pilot; larger
dimensions fail explicitly rather than silently changing estimators.

Usage:

```python
model = ADP_Manifold(1, estimator="local_quadratic", N_loc=20)
model.fit(X, Y)
```
