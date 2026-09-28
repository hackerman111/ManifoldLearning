# Fixed-g rank-constraint comparison

Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m benchmarks.svd_correction \
  --output experiments/svd_correction_2026-09-29/results.json
```

`results.json` records the commit/dirty state, Python/NumPy/SciPy versions,
thread variables, seeds, shape `(J,p,d,m)=(40,8,20,3)`, float64, ridge 0.3,
the full fixed-g ridge correction spectrum and energy `E_r`, and per-method
objective, time, traced peak allocation, U passes and projector distances.
The dense reference solves the same fixed-g objective without a rank bound;
it is not a full outer HPAO fit. Memory is `tracemalloc` Python allocation
peak, excluding native BLAS allocations. No estimator or initialization
parameters enter this fixed-statistics experiment.

| seed | full objective | rank-2 matrix objective | rank-2 correction objective | E_2 |
| ---: | ---: | ---: | ---: | ---: |
| 11 | 3.488 | 213.893 | 120.718 | 0.884 |
| 23 | 4.431 | 198.687 | 117.213 | 0.929 |
| 37 | 4.827 | 202.140 | 231.584 | 0.892 |

The rank-2 correction improves this objective on two of three seeds and
regresses on one. Both greedy solvers remain far above the unrestricted
reference; high `E_2` of the reference correction does not certify greedy
recovery of its best rank-2 approximation. Median measured times were 2.69 ms
for dense full, 9.56 ms for rank-2 matrix and 8.91 ms for rank-2 correction.
These small-shape timings do not establish a general speed ranking.
