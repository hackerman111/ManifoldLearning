# SVD vs grassman_optim on heavy Spokoiny

We ran six cold full fits on the paired `mi-spokoini-m2-dhigh` workload:
`n=800, d=50, m=2, N_J=800, N_phi=10, N_lin=100`, float64, seeds 0–2.
Each run used a fresh process, one thread for OpenBLAS/OMP/MKL, and the order
alternated by seed. Input hashes match within each seed. The raw records and
machine-readable protocol are in `fullfit_runs.jsonl` and
`fullfit_protocol.json`.

| Seed | SVD time (s) | grassman_optim time (s) | grassman_optim / SVD | SVD projector error | grassman_optim projector error |
|---:|---:|---:|---:|---:|---:|
| 0 | 24.282 | 23.484 | 0.967 | 0.1183 | 0.0542 |
| 1 | 24.903 | 22.608 | 0.908 | 0.0899 | 0.0473 |
| 2 | 25.067 | 26.782 | 1.068 | 0.0987 | 0.0449 |
| Median | 24.903 | 23.484 | 0.967 (paired ratios) | 0.0987 | 0.0473 |

The ratio of the two median times is 0.943: `grassman_optim` took 5.7% less
time by that summary. The median of the three paired ratios is 0.967 (3.3% less
time); seed 2 was slower for `grassman_optim`, so the small timing advantage is
not uniform. Peak RSS medians were 104,916 KiB for SVD and 108,704 KiB for
`grassman_optim`.

Both methods made 162 outer iterations and stopped at `h_min`. The optimized
Grassmann inner call at the final step reached its five-iteration cap without
converging on all three seeds. SVD exposes different solver diagnostics, so
the inner convergence status is not directly comparable. Projector error to
the known basis was smaller for `grassman_optim` on all three seeds, but the
two solver methods have different algorithms and objectives; this timing
experiment does not establish estimator equivalence. Solver settings and
source hashes are recorded in the protocol.

Reproduce from the repository root with:

```bash
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_CACHE_DIR=/tmp/adp-uv-cache PYTHONPATH=. uv run --no-sync python -m benchmarks.svd_grassman_optim_benchmark --output experiments/svd_vs_grassman_optim_spokoiny_2026-10-02-rerun
```
