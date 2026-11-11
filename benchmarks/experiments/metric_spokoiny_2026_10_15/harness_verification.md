# Harness verification and reproduction

## Checks

- Tiny synthetic end-to-end smoke: captured a two-outer-call fit, mapped call 0 as `early` and the final available call as `late`, loaded NPZ fields `U`, `I`, `mass`, `index_init`, and `truth_basis`, then completed a frozen metric2 solve. Passed; temporary data/artifacts were removed.
- Focused regression: `rtk proxy env UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync pytest -q tests/test_metric_spokoiny.py` -> 10 passed in 0.36s. It verifies all 90 unique frozen job specifications and mocks the launcher to assert `--snapshot` receives the selected NPZ path.
- `ruff format` and `ruff check` passed on run.py, workers.py, analysis.py, diagnostics.py, and the focused test.
- Scoped Pyright passed on the four harness modules: 0 errors, warnings, or information diagnostics.
- `python -m experiments.metric_spokoiny_2026_10_15.run --help` and the analysis CLI `--help` passed.
- `git diff --check` passed.

## Timed stage commands

Runs were serial; each solve used a fresh subprocess with `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`. No tests overlapped any timed stage. The exact stage commands were:

```text
rtk proxy env UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.metric_spokoiny_2026_10_15.run --stage fullfit5
rtk proxy env UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.metric_spokoiny_2026_10_15.run --stage capture
rtk proxy env UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.metric_spokoiny_2026_10_15.run --stage frozen
rtk proxy env UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.metric_spokoiny_2026_10_15.run --stage frozen --output experiments/metric_spokoiny_2026_10_15/results_repaired
rtk proxy env UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.metric_spokoiny_2026_10_15.run --stage budget20 --output experiments/metric_spokoiny_2026_10_15/results_repaired
rtk proxy env UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.metric_spokoiny_2026_10_15.run --stage validation --output experiments/metric_spokoiny_2026_10_15/results_repaired
rtk proxy env UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.metric_spokoiny_2026_10_15.analysis --output experiments/metric_spokoiny_2026_10_15/results_repaired
```

`fullfit5` retained 9 successful fits; `capture` retained 4 successful diagnostic fits and 12 NPZ snapshots (9 core snapshots plus 3 seed-0 metric2 snapshots); the first frozen launch retained 90 CLI `process_error` rows because the runner omitted `--snapshot`. No solver ran for those rows. The one-line frozen-job argument fix is isolated to `run.py`. Old runner bytes and SHA-256 are preserved in `results/launch_original.py.txt`; original full-fit/capture JSONL and NPZ inputs were copied byte-for-byte into `results_repaired/`, with source distinction and hashes recorded in `results_repaired/repair_note.md`. The repaired frozen stage then retained 90 successful solves; budget20 retained 6 successful fits. Validation wrote a skipped marker because the predeclared development gate failed; seeds 3–7 were untouched.

The analysis command writes `analysis_summary.json`, `paired_results.csv`, and `metric_spokoiny_summary.png`. Repaired analysis includes the 90 imported original launch errors as failures so they are not lost in the aggregate. It never counts them as numerical solver failures.

Final counts: original fullfit5 9/9 success, capture 4/4 success, frozen launch 90/90 process errors; repaired fullfit5/capture inputs are copied unchanged, repaired frozen 90/90 success, budget20 6/6 success, validation 1 skipped. Analysis records exactly 90 retained launch errors and a failed development gate. The independent lead audit confirmed unique frozen inputs, matching hashes, one thread per BLAS pool, monotone objectives, and maximum orthogonality error `6.22e-15`.

## Measurement boundaries

Full-fit worker wall time starts immediately before `.fit()` and excludes imports and problem generation; no whole-fit warmup is used. The worker records cold first-solve calls, summed solver wall time, and the model's built-in whole-fit stage profile after `.fit()`. Fixed-statistic wall time wraps only the solver call; certificate/objective/projector recomputation happens afterward. Snapshot capture is diagnostic-only and excluded from performance comparisons. `ru_maxrss` is reported as Linux KiB. The tested settings, input hashes, source hashes, effective config, threadpools, thread environment, Git commit/dirty fingerprint, and protocol hash are retained in rows and manifests.

The supplementary cProfile driver `results_repaired/profiles/profile_solver.py` profiles only the solve call and stores top-25 cumulative/self-time text, `.pstats`, and a final independently recomputed certificate. It is not part of benchmark timings. Invoke it in separate fresh processes with one BLAS thread, as shown by the exact core and metric2 commands in `REPORT.md`. The profile inputs and implementation hashes are covered by `results_repaired/run_manifest.json` for the numerical code; the driver itself is an untimed result-side recipe.

