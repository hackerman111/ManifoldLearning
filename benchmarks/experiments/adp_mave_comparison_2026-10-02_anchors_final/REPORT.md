# ADP vs MAVE-family benchmark

Synthetic paired benchmark; each method receives identical saved inputs for a given seed.
Higher projector overlap is better; lower projector distance, principal angle, fit time, and RSS are better.
Fit time excludes process startup, imports, and package loading. It includes method initialization and response-kernel construction.

## Results

| Method | Success | Median fit (s) | Median peak RSS (MiB) | Median overlap | Median projector distance | Median angle (deg) | Paired time ratio vs ADP |
|---|---:|---:|---:|---:|---:|---:|---:|
| ADP | 3/3 | 0.154 | 73.9 | 0.9976 | 0.0493 | 2.73 | 1.000x |
| MAVE | 3/3 | 1.301 | 90.5 | 0.9985 | 0.0390 | 2.12 | 8.334x |
| dMAVE | 3/3 | 6.331 | 98.8 | 0.9987 | 0.0367 | 1.85 | 41.066x |
| rMAVE | 3/3 | 0.893 | 114.4 | 0.9974 | 0.0514 | 2.45 | 5.835x |

## Protocol and interpretation

- Problem: `n=300, d=8, m=2`; seeds `0, 1, 2`; R MAVE `1.3.12`. Raw per-fit rows, including failures, are in `runs.jsonl`.
- Data: `X ~ N(0,I)`, `z = X B_true`, `y = sin(z1) + 0.5 z2^2 + N(0,0.15^2)` for `m=2` (additional active `tanh` terms for `m>2`).
- `MAVE` is the package's `CSMAVE` method ([MAVE manual](https://stat.ethz.ch/CRAN/web/packages/MAVE/MAVE.pdf)). The installed MAVE package does not expose selectors named `dMAVE` or `rMAVE`.
- `dMAVE` follows the kernelized-response construction for density MAVE ([review](https://pmc.ncbi.nlm.nih.gov/articles/PMC3685755/)): at five quantile anchors `t`, fit package `CSMAVE` to `K_h(y-t)` and aggregate projectors. `h=1.06 sd(y) n^(-1/5)`; all anchor fits are included in fit time.
- `rMAVE` is a benchmark-only projected-kernel refinement initialized with package `CSOPG`; its local/global weighted least-squares iterations and bandwidth schedule are implemented in `benchmarks/adp_mave_methods.R`.
- These are distinct estimators/implementations. This small synthetic suite measures recovery for the stated model and does not establish general superiority.
- Peak RSS is process high-water mark from `/proc/self/status` when available. R/Python startup memory is included; startup time is reported separately as `process_seconds` in raw rows.

## Reproduction

```sh
UV_CACHE_DIR=/tmp/adp-mave-uv-cache uv run --no-sync python benchmarks/adp_mave_benchmark.py --output experiments/adp_mave_comparison_reproduction
```

The output directory must be new; the driver refuses to overwrite previous measurements.
