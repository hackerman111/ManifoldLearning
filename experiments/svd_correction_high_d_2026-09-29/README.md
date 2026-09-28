# SVD correction timing at larger dimensions

## Setup

This is a fixed-`g` microbenchmark of `low_rank_target="matrix"` (the
historical rank-`B` solve) and `low_rank_target="correction"` (rank-`(B-P)`),
using the same generated arrays inside each seed/dimension pair. The task uses
`(J,p,m)=(40,8,3)`, correction rank 2, ridge `0.3`, `float64`, seeds 11, 23,
and 37, and three repetitions per method. Method order alternates across
repetitions.

These options impose different rank constraints and can return different
objectives. This measures their runtime on the same fixed task data; it does
not show that they are interchangeable estimators or compare recovery quality.

Reproduce with one BLAS thread:

```sh
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync \
  python -m benchmarks.svd_correction \
  --dimensions 100 300 600 --repeats 3 \
  --output experiments/svd_correction_high_d_2026-09-29/results.json
```

Environment: Python 3.13.12, NumPy 2.5.2, SciPy 1.18.1; run from source
revision `e072e97d140d6e6e650853c66bb6f28929633ed2` with working-tree changes
present. Raw rows, diagnostics, and environment metadata are in `results.json`.
For `d=100`, the certified direct solve is used; `d=300` and `d=600` use
LSMR. All 54 solver calls completed successfully. Maximum reported normal
residual was `3.68e-9` for the LSMR cases and `1.20e-15` for direct solves.

## Timing

Each cell below is the median of three repetitions for one seed. The final
column is the median across the three **per-seed paired speed ratios**, where
speed ratio is matrix time / correction time. Values above 1 mean correction
was faster. The ranges show those three seed-level ratios.

| d | Matrix medians (ms), seeds 11 / 23 / 37 | Correction medians (ms), seeds 11 / 23 / 37 | Paired speed ratio median (range) |
|---:|---:|---:|---:|
| 100 | 19.3 / 21.2 / 19.8 | 17.9 / 21.5 / 19.2 | 1.030 (0.988–1.078) |
| 300 | 5014 / 5370 / 4211 | 4953 / 4399 / 4244 | 1.012 (0.992–1.221) |
| 600 | 3558 / 2781 / 5256 | 2960 / 2598 / 6008 | 1.071 (0.875–1.202) |

The result is effectively tied at `d=100/300`; `d=600` has a modest median
paired gain, but one seed is slower with the correction constraint. This
three-seed fixed-`g` run is not enough to claim a general speedup. LSMR work
also varies substantially by seed: the correction mode used fewer median
`U`-vector passes at `d=300` (29,944 vs 33,798) and `d=600` (17,541 vs
21,448), consistent with its runtime pattern.

`tracemalloc` reports only Python-traced allocations and excludes NumPy/BLAS
native memory; its sub-megabyte figures are not a peak-RSS comparison. Median
traced peaks (matrix / correction) were 0.48 / 0.48 MiB at `d=100`,
0.32 / 0.33 MiB at `d=300`, and 0.54 / 0.55 MiB at `d=600`, so this metric
shows no meaningful memory difference. The large-d runs skip the dense
full-rank reference to bound work.
