# Grassmann solve stage profile (2026-10-06)

## Protocol

Existing Spokoiny full-fit generator; n=800, d=50, m=2, J=800, P=10, three seeds (0,1,2), float64, max_steps=5. Each subprocess uses one BLAS/OMP/MKL thread. Baseline and cProfiled rows are paired sequentially; a separate lightweight timer-only pass reused the exact baselines. Production solver source was not changed.

Raw JSONL, protocol metadata, and per-seed `.prof` files are in `results/` and `lightweight_results/`. The initial failed smoke serialization rows are retained in `failures/`; the first timer-only pass with out-of-solve linalg counted is retained in `failures/lightweight_scope_miss/` and excluded from analysis.

## Paired results

| Seed | Full fit s (baseline) | Solver s (baseline) | Solver s (timer only) | Timer delta | Solver / fit | Quality | Stop | RSS MiB | Rank fallbacks | Profile evals | GN solves |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 24.64 | 3.009 | 3.106 | +3.2% | 12.2% | 0.05417 | h_min | 105.1 | 190 | 3191 | 620 |
| 1 | 25.24 | 2.589 | 2.438 | -5.8% | 10.3% | 0.04728 | h_min | 108.4 | 0 | 810 | 810 |
| 2 | 28.16 | 6.255 | 6.344 | +1.4% | 22.2% | 0.04494 | h_min | 105.2 | 605 | 8373 | 205 |

The basis and complete solver-diagnostics hashes match exactly for baseline, cProfiled, and timer-only runs at every seed. Timer-only solver-time deltas have median +1.4% (per seed shown above); cProfile overhead has median +26.6%, so stage ranking uses timer-only rows.

## Lightweight disjoint stage accounting

Pooled shares use the sum of stage seconds divided by total timer-only solver seconds across all three seeds. The per-seed local-profile share is also shown because fallback-heavy seeds vary substantially.

| Disjoint bucket | Pooled seconds | Pooled solver share |
|---|---:|---:|
| local_profile | 5.626 | 47.3% |
| project_gradient | 1.183 | 10.0% |
| core_jacobian | 0.807 | 6.8% |
| gn_lstsq | 0.300 | 2.5% |
| polar | 0.310 | 2.6% |
| geometry_qr | 0.207 | 1.7% |
| geometry_svd | 0.621 | 5.2% |
| remainder | 2.833 | 23.8% |

Local-profile share by seed: 0: 36.9%, 1: 33.3%, 2: 57.8%; pooled share 47.3%. `project_gradient` combines `_project` and `_gradient`; the geometry QR/SVD buckets are separate. `remainder` covers Python/controller, line-search, and instrumentation work not assigned to named child stages; it is not a pure line-search measurement.

## Interpretation and limits

Local profiling/refitting is the largest pooled stage at 47.3%, but its per-seed share ranges from about one third to 57.8%. Seed 2 has 605 rank-guard fallbacks, 8,373 profile evaluations, and only 205 GN solves; its cProfile artifact shows repeated local profile SVD and bounded scalar minimization as the dominant fallback path. Seeds 0/1 have 190/0 rank fallbacks. This variation is solver-path dependent, so the pooled share is descriptive of these three seeds rather than universal.

Across these full fits, solver time is 10–23% of fit time; most runtime remains outside `solve`. All fits stop at `h_min` after 162 outer iterations; the final inner call reports `max_steps` with `converged=false`. Quality and stop state were unchanged across paired instrumentation modes. RSS is process peak RSS and includes Python/runtime overhead.

CProfile timings are retained only as a diagnostic artifact because instrumentation increased solver wall time materially (median 26.6%). No optimization or estimator change was made.

## Provenance

Solver source SHA-256: `f30c1d3596f5092a1e96d7275790384f9a3206b149f5fb8fa395e0344b3ab333`. Timer-only harness SHA-256: `e1795edf8ae973faf9a1bc858a32552d73be05546761c451b1bb50339bad8edc`. cProfile harness SHA-256: `de51ba7ccd2a5c45d3574df7ee96c8720803390a9f51d55121f04d7acdb2fa12`. Runtime, command, thread settings, commit, and dirty paths are recorded in both protocol JSON files.
