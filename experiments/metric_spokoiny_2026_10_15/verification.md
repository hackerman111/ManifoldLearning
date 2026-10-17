# Independent correctness checks

Protocol/source read: `PROTOCOL.md`, `MATH.md`,
`ADP/solver/_grassman_schur.py::metric_direction`,
`ADP/solver/grassman_research.py::solve`, and the true-profile/polar helpers
used to define the tested objective. All tests use small deterministic arrays;
there are no timing assertions or benchmark-sized fits.

Command:

```text
rtk proxy env UV_CACHE_DIR=/tmp/edr-structured-2026-10-15-uv-cache uv run python -m pytest -q tests/test_metric_spokoiny.py
```

Result after the diagnostics refactor, matched-control patch-path smoke, and
frozen-launcher regression: **10 passed in 0.36s**. Covered an independent eig-floor plus SVD
oracle, unit norm under the floored metric, rank and negative slope; the
explicit floor-one rank-two formula with large coefficients and an inactive
`tan(0.5)` cap; right-gauge covariance; polar-profile finite differences;
8x8 Gauss-Legendre integration of the population OPG against
`(5557/1155) I`; and metric solver ranks 1 and 2 with monotone profile,
orthogonality, and an independent augmented least-squares final full-gradient
certificate. The last two cases verify the q=2 spectral-norm matched
isotropic formula and run both `isotropic2` and `isotropic2_matched` through
the worker's `_solver` plus diagnostics `metric_patch`/`restore_metric` route
on small arrays, restoring the patched function in `finally`. The launcher
test builds all nine required seed/phase inputs, verifies `_jobs("frozen")`
creates 90 unique arm/budget jobs, then mocks one worker process and checks
that `_run_job` passes its existing NPZ path as `--snapshot`.

The preserved pre-fix `results/frozen.jsonl` has 90 `process_error` rows; the
worker parser reported `--snapshot is required for frozen mode`. The launcher
fix was already present when this regression test was added, so this test
agent captured a GREEN run rather than an additional RED run. The saved rows
are the pre-fix failure evidence.

The first local run exposed two errors in the test oracle itself: it measured
the norm with unfloored `R` instead of `R_f`, and omitted the residual outer
product in the independent gradient. Those test-only expressions were
corrected; the final recorded run above passed. No production files changed.

Scoped style checks passed for the test and experiment files:

```text
rtk proxy env UV_CACHE_DIR=/tmp/edr-structured-2026-10-15-uv-cache uv run ruff check tests/test_metric_spokoiny.py experiments/metric_spokoiny_2026_10_15/diagnostics.py experiments/metric_spokoiny_2026_10_15/workers.py
All checks passed!
rtk proxy env UV_CACHE_DIR=/tmp/edr-structured-2026-10-15-uv-cache uv run ruff format --check tests/test_metric_spokoiny.py experiments/metric_spokoiny_2026_10_15/diagnostics.py experiments/metric_spokoiny_2026_10_15/workers.py
3 files already formatted
```

Pyright passed on the test file (`0 errors, 0 warnings`). An intermediate
harness check found two type diagnostics at `int(PARAMS["m"])`; the harness
owner corrected the dimension source to `truth.shape[1]`. Refer to
`harness_verification.md` for the final harness-wide verification. This test
agent did not rerun that broader check after the correction.

No production, worker, protocol or benchmark files were changed by this test
task.

## Final scoped verification after launcher regression

Focused tests, run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
MKL_NUM_THREADS=1`:

```text
rtk proxy env UV_CACHE_DIR=/tmp/edr-structured-2026-10-15-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m pytest -q tests/test_metric_spokoiny.py
10 passed in 0.31s
```

Pyright on `run.py`, `workers.py`, `analysis.py`, `diagnostics.py`,
`audit_proposals.py`, and this test file: **0 errors, 0 warnings**. Ruff
format-check passed for the same six files. Ruff check passed for the five
files other than the lead-owned `audit_proposals.py`; including that file
reports one `E501` at line 73 (99 characters) on the JSON `scope` string. This
was reported without editing the lead-owned file.

Lead final correction: split the single long `scope` string in
`audit_proposals.py` into adjacent literals; no semantic change. Ruff check
and format-check now pass for that file too. The exact executed audit source
is preserved as `results_repaired/proposal_audit_source.py.txt`; its SHA-256
matches `proposal_audit.json`. No numerical results were rewritten.
