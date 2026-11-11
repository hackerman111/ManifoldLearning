# ADP vs MAVE-family benchmark

Synthetic paired benchmark; each method receives identical saved inputs for a given seed.
Higher projector overlap is better; lower projector distance, principal angle, fit time, and RSS are better.
Fit time excludes process startup, imports, and package loading. It includes method initialization and response-kernel construction.

## Results

| Method | Success | Median fit (s) | Median peak RSS (MiB) | Median overlap | Median projector distance | Median angle (deg) | Paired time ratio vs ADP |
|---|---:|---:|---:|---:|---:|---:|---:|
| ADP | 3/3 | 0.151 | 73.9 | 0.9976 | 0.0493 | 2.73 | 1.000× |
| MAVE | 3/3 | 1.220 | 90.4 | 0.9985 | 0.0390 | 2.12 | 8.093× |
| dMAVE | 3/3 | 2.248 | 89.5 | 0.3968 | 0.7767 | 55.43 | 14.899× |
| rMAVE | 3/3 | 0.866 | 115.3 | 0.9974 | 0.0514 | 2.45 | 5.745× |

## Protocol and interpretation

- Seeds: `0, 1, 2`. Raw per-fit rows, including failures, are in `runs.jsonl`.
- `MAVE` is the package's `CSMAVE` method. The installed MAVE package does not expose selectors named `dMAVE` or `rMAVE`.
- `dMAVE` is a density-response wrapper: a Gaussian kernel response matrix is passed to the package's `MEANMAVE` multi-response interface. Its response bandwidth is included in fit time.
- `rMAVE` is a benchmark-only projected-kernel refinement initialized with package `CSOPG`; its local/global weighted least-squares iterations and bandwidth schedule are implemented in `benchmarks/adp_mave_methods.R`.
- These are distinct estimators/implementations. This small synthetic suite measures recovery for the stated model and does not establish general superiority.
- Peak RSS is process high-water mark from `/proc/self/status` when available. R/Python startup memory is included; startup time is reported separately as `process_seconds` in raw rows.

## Reproduction

```sh
uv run --no-sync python benchmarks/adp_mave_benchmark.py
```
