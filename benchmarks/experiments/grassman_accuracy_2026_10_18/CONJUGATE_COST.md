# Conjugate exploratory profile (seed 90800)

Diagnostic-only cProfile runs through `pilot_intrinsic.fit`: `d-20` and `d-50`, `n=500`, method seed `1173550236`, one BLAS thread. Each cell is one fit. Original uses `grassman.solve` with budget 5; candidate uses `conjugate_guard.solve` with budget 10. Settings remain unchanged. Profiler timing is perturbed and is not performance evidence.

| Case / method | Fits | fit wall s | full fit cProfile s | solver cumulative s (% full fit) | solver self s |
|---|---:|---:|---:|---:|---:|
| d-20 original5 | 1 | 0.869 | 0.869 | 0.588 (67.6%) | 0.013 |
| d-20 conjugate10 | 1 | 0.808 | 0.808 | 0.528 (65.4%) | 0.131 |
| d-50 original5 | 1 | 0.772 | 0.772 | 0.261 (33.9%) | 0.016 |
| d-50 conjugate10 | 1 | 0.964 | 0.964 | 0.454 (47.0%) | 0.116 |

Fit wall time is the timer returned by `pilot_intrinsic.fit`; cProfile total is the `ADP_multi_index.fit` cumulative time. Solver cumulative is cumulative time in the actual solver entry point (`grassman.solve` or `backtracking.solve`); it is nested inside full-fit time. Self time for `backtracking.solve` is a useful locator for its Python loop/control work, but is not itself a removable-time estimate.

## Function timing

Numbers are cumulative seconds (% of full-fit cProfile time). `_profile` includes its descendants, so rows overlap with SVD and least-squares rows. `numpy.linalg.svd/lstsq` are aggregated by function name. `_core_jacobian` was not called for d-20.

| Case / method | `_gradient` | `_project` | `_profile` | `_core_jacobian` | NumPy SVD | NumPy lstsq |
|---|---:|---:|---:|---:|---:|---:|
| d-20 original5 | .00237 (.27%) | .00130 (.15%) | .488 (56.2%) | 0 | .183 (21.1%) | .148 (17.1%) |
| d-20 conjugate10 | .00419 (.52%) | .00349 (.43%) | .365 (45.2%) | 0 | .135 (16.7%) | .111 (13.7%) |
| d-50 original5 | .00424 (.55%) | .00393 (.51%) | .159 (20.6%) | .00287 (.37%) | .0753 (9.8%) | .173 (22.5%) |
| d-50 conjugate10 | .00716 (.74%) | .00870 (.90%) | .285 (29.6%) | .00298 (.31%) | .110 (11.4%) | .190 (19.7%) |

The largest exclusive self-time hotspots were NumPy SVD/lstsq, `_dense_moments`, and the solver loop. The conjugate solver entry spent 0.131 s self time for d-20 and 0.116 s for d-50 (16.2% and 12.0% of full-fit cProfile time); this includes loop/control and direct solver work. Gradient and projection self times were each below 0.8% of full-fit time in all four fits. These shares locate code for inspection; they do not establish an optimization opportunity or predict a speedup.

## Reproducibility artifacts

Profiles and sorted cumulative/tottime listings: `/tmp/grassman-cg-profile-2026-10-22/`. The directory contains four `.pstats` files, eight text listings, and `summary.json`. The temporary runner is `/tmp/grassman_cg_profile.py`.
