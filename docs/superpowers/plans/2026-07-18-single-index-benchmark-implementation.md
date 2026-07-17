# Single-index Benchmark Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy C/S/T/R/M single-index catalog with the deterministic 27,400-run benchmark from `new_benchmark.md`, exposed through a resumable process-parallel CLI whose individual ADP fits are pinned to one core.

**Architecture:** Keep the existing `adp.evaluation.single_index` package and rewrite its experiment catalog, data generator, executor, storage, runner, and reporting contracts in place. Add opt-in, benchmark-neutral telemetry to the ADP engine and random-projection implementation; persist normalized per-run shards atomically and derive all combined CSV files and plots exclusively from committed shards.

**Tech Stack:** Python 3.14, NumPy, SciPy, pandas, matplotlib, `concurrent.futures.ProcessPoolExecutor`, `threadpoolctl`, pytest.

---

The user explicitly requested implementation in the current checkout, so this plan does not create a git worktree. Existing unrelated deletions and the untracked `new_benchmark.md` must remain untouched.

## File map

- Modify `adp/evaluation/single_index/types.py`: immutable experiment parameters, seed bundles, jobs, series config, and run outcomes.
- Modify `adp/evaluation/single_index/scenarios.py`: exact full/smoke grids, selectors, seed parsing, and job-count metadata.
- Modify `adp/evaluation/single_index/datasets.py`: deterministic AR(1), distributions, links, noise, outliers, and misspecification.
- Create `adp/evaluation/single_index/telemetry.py`: ESS, local 2x2 diagnostics, row aggregates, beta encoding, and timing remainder.
- Modify `adp/common/types.py`, `adp/engine/algorithm.py`, `adp/variants/random_projection.py`, and `adp/backends/numpy_backend.py`: opt-in generic telemetry hooks and real CG diagnostics without changing numerical updates.
- Modify `adp/evaluation/single_index/executors.py`: one ADP fit, status classification, and normalized row construction.
- Modify `adp/evaluation/single_index/schema.py`: stable headers for seven public tables.
- Modify `adp/evaluation/single_index/storage.py`: atomic per-run shards, commit-last semantics, resume compatibility, retry replacement, and bounded merge.
- Modify `adp/evaluation/single_index/runner.py`: deterministic job expansion, dry-run, process pool, thread caps, reports-only, and progress.
- Create `adp/evaluation/single_index/plots.py`: small CSV-frame plot renderers.
- Modify `adp/evaluation/single_index/reports.py`: CSV-only report orchestration and artifact isolation.
- Modify `adp/evaluation/cli.py` and `adp/evaluation/single_index/__init__.py`: public CLI/API contract.
- Modify `README.md`: launch, resume, reports-only, outputs, and one-core guarantees.
- Rewrite `tests/test_single_index_benchmark_*.py` and update `tests/test_cli.py`; add focused ADP telemetry coverage to `tests/test_performance_optimizations.py`.

### Task 1: Exact experiment catalog and deterministic identities

**Files:**
- Modify: `adp/evaluation/single_index/types.py`
- Modify: `adp/evaluation/single_index/scenarios.py`
- Modify: `adp/evaluation/single_index/runner.py`
- Test: `tests/test_single_index_benchmark_scenarios.py`
- Test: `tests/test_single_index_benchmark_runner.py`

- [ ] **Step 1: Replace legacy scenario tests with failing full-matrix tests**

```python
EXPECTED_COUNTS = {
    "1": 800, "2": 2_000, "3": 4_200, "4": 3_600, "5": 3_000,
    "6": 3_600, "7.1": 1_200, "7.2": 1_200, "8.1": 800,
    "8.2": 2_000, "8.3": 1_600,
}

def test_full_profile_has_exact_independent_experiment_counts():
    jobs = build_single_index_jobs(SingleIndexSeriesConfig(profile="full"))
    counts = Counter(job.experiment for job in jobs)
    assert counts == EXPECTED_COUNTS
    assert len(jobs) == 27_400
    assert all(job.parameters.n == math.ceil(job.parameters.d * job.parameters.n_over_d) for job in jobs)
    assert all(job.parameters.center_fraction == 1.0 for job in jobs)

def test_selector_parser_normalizes_and_rejects_unknown_values():
    assert parse_experiment_selectors("all") == tuple(EXPECTED_COUNTS)
    assert parse_experiment_selectors("8.3,1,1") == ("1", "8.3")
    with pytest.raises(ValueError, match="unknown experiment selector"):
        parse_experiment_selectors("9.1")
```

- [ ] **Step 2: Run the catalog tests and verify the expected RED failure**

Run: `python -m pytest tests/test_single_index_benchmark_scenarios.py tests/test_single_index_benchmark_runner.py -q`

Expected: FAIL because `ExperimentParameters`, the new selectors, and the 27,400-run expansion do not exist.

- [ ] **Step 3: Implement immutable experiment and series types**

```python
EXPERIMENT_SELECTORS = ("1", "2", "3", "4", "5", "6", "7.1", "7.2", "8.1", "8.2", "8.3")

@dataclass(frozen=True, slots=True)
class ExperimentParameters:
    d: int
    n_over_d: float
    sigma_x: float = 1.0
    rho_corr: float = 0.0
    sigma_eps: float = 0.5
    link: str = "quadratic"
    x_distribution: str = "gaussian"
    noise_distribution: str = "gaussian"
    heteroscedastic: bool = False
    outlier_fraction: float = 0.0
    outlier_scale: float = 1.0
    delta: float = 0.0
    center_fraction: float = 1.0

    @property
    def n(self) -> int:
        return math.ceil(self.d * self.n_over_d)

    @property
    def n_centers(self) -> int:
        return min(self.n, math.ceil(self.center_fraction * self.n))

@dataclass(frozen=True, slots=True)
class SeedBundle:
    beta: int
    features: int
    noise: int
    centers: int
    directions: int
    init: int
    outliers: int
    outlier_noise: int
    gamma: int
    misspecification: int

@dataclass(frozen=True, slots=True)
class SingleIndexSeriesConfig:
    profile: Literal["smoke", "full"] = "smoke"
    experiments: tuple[str, ...] = EXPERIMENT_SELECTORS
    jobs: int | Literal["auto"] = "auto"
    seeds: tuple[int, ...] | None = None
    diagnostic_seeds: tuple[int, ...] = (0, 1, 2)
    center_fraction: float = 1.0
    retry_failed: bool = False
    max_runs: int | None = None
```

Validate finite positive dimensions/scales, `0 <= rho_corr < 1`, nonnegative noise/contamination, `0 < center_fraction <= 1`, unique nonnegative seeds, and positive integer jobs.

- [ ] **Step 4: Implement literal independent grids and stable job expansion**

Use one `_full_parameter_grid(selector)` branch per selector, with values copied literally from the spec. Full defaults to seeds `range(100)`; smoke uses one small representative configuration per selected family and seed `0`. Split every user seed with `np.random.SeedSequence([seed, selector_index, parameter_index]).generate_state(10)`. Build `run_id` from selector, canonical parameter dict, user seed, and sub-seeds; exclude jobs/process count from run identity. Apply `max_runs` only after deterministic expansion.

- [ ] **Step 5: Verify GREEN and deterministic invariants**

Run: `python -m pytest tests/test_single_index_benchmark_scenarios.py tests/test_single_index_benchmark_runner.py -q`

Expected: PASS, including identical IDs after changing job count and exact total 27,400.

- [ ] **Step 6: Commit the catalog slice**

```bash
git add adp/evaluation/single_index/types.py adp/evaluation/single_index/scenarios.py adp/evaluation/single_index/runner.py tests/test_single_index_benchmark_scenarios.py tests/test_single_index_benchmark_runner.py
git commit -m "feat: define new single-index benchmark matrix"
```

### Task 2: Deterministic standardized synthetic data

**Files:**
- Modify: `adp/evaluation/single_index/datasets.py`
- Test: `tests/test_single_index_benchmark_executors.py`

- [ ] **Step 1: Write failing generator tests**

```python
def test_gaussian_features_use_ar1_covariance():
    generated = generate_synthetic_data(make_job(rho_corr=0.75, d=6, n_over_d=2000))
    empirical = np.corrcoef(generated.data.X, rowvar=False)
    assert empirical[0, 1] == pytest.approx(0.75, abs=0.03)
    assert empirical[0, 2] == pytest.approx(0.75**2, abs=0.03)

@pytest.mark.parametrize("distribution", ["uniform", "student_t5"])
def test_feature_distributions_are_standardized(distribution):
    generated = generate_synthetic_data(make_job(x_distribution=distribution, d=4, n_over_d=5000))
    assert generated.data.X.var(axis=0) == pytest.approx(np.ones(4), abs=0.06)

def test_signal_noise_outliers_and_misspecification_are_reproducible():
    first = generate_synthetic_data(make_job(sigma_eps=0.5, outlier_fraction=0.05, delta=0.5))
    second = generate_synthetic_data(make_job(sigma_eps=0.5, outlier_fraction=0.05, delta=0.5))
    np.testing.assert_array_equal(first.data.y, second.data.y)
    assert np.var(first.signal) == pytest.approx(1.0, abs=1e-12)
    assert abs(first.data.beta @ first.gamma) < 1e-12
```

Also test standardized `t5`/`t3` noise, exact outlier replacement indices, all six links, `sigma_eps=0` infinite SNR, isolated sub-seed changes, dense unit beta, and degenerate link rejection.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_single_index_benchmark_executors.py -q`

Expected: FAIL because the old generator uses equicorrelation and one shared data RNG.

- [ ] **Step 3: Implement generation metadata and deterministic transforms**

```python
@dataclass(frozen=True, slots=True)
class GeneratedSingleIndexData:
    data: ADPData
    signal: np.ndarray
    ordinary_noise: np.ndarray
    gamma: np.ndarray | None
    metadata: dict[str, Scalar]

def _ar1_factor(d: int, rho: float) -> np.ndarray:
    covariance = rho ** np.abs(np.subtract.outer(np.arange(d), np.arange(d)))
    return np.linalg.cholesky(covariance)

def _standardize_sample(values: np.ndarray, name: str) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(values))
    scale = float(np.std(values, ddof=0))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise ValueError(f"{name} has degenerate sample variance")
    return (values - mean) / scale, mean, scale
```

Draw Gaussian features as `rng.normal(size=(n, d)) @ factor.T`; uniform from `[-sqrt(3), sqrt(3)]`; `t5 * sqrt(3/5)`. Normalize every link sample to variance one. Standardize `t5` and `t3` noises by `sqrt(3/5)` and `sqrt(1/3)`. Implement heteroscedastic scale exactly. Replace ordinary errors at selected outlier indices. Orthogonalize and normalize `gamma`, then normalize `g(gamma.T @ X)` independently. Record normalization scalars, outlier count, SNR, and effective parameters.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_single_index_benchmark_executors.py -q`

Expected: PASS.

- [ ] **Step 5: Commit deterministic generation**

```bash
git add adp/evaluation/single_index/datasets.py tests/test_single_index_benchmark_executors.py
git commit -m "feat: generate benchmark data deterministically"
```

### Task 3: Generic ADP telemetry and diagnostic math

**Files:**
- Create: `adp/evaluation/single_index/telemetry.py`
- Modify: `adp/common/types.py`
- Modify: `adp/engine/algorithm.py`
- Modify: `adp/variants/random_projection.py`
- Modify: `adp/backends/numpy_backend.py`
- Modify: `adp/backends/cupy_backend.py`
- Test: `tests/test_performance_optimizations.py`
- Test: `tests/test_single_index_benchmark_executors.py`

- [ ] **Step 1: Add failing pure telemetry tests**

```python
def test_weight_and_local_diagnostics_match_fixed_arrays():
    weights = np.array([[1.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
    summary = summarize_weights(weights)
    np.testing.assert_allclose(summary.ess, [1.8, 0.0])
    np.testing.assert_array_equal(summary.nonzero, [2, 0])

    diagnostics = diagnose_local_systems(
        S=np.array([[1.0, 2.0]]),
        U=np.array([[[2.0], [1.0]]]),
        imav=np.array([[2.0, 3.0]]),
        beta=np.array([1.0]),
        intercepts=np.array([0.0]),
        slopes=np.array([1.0]),
        regularization=1e-10,
    )
    assert diagnostics[0].rank == 2
    assert diagnostics[0].condition >= 1.0
    assert diagnostics[0].singular is False
```

Add tests for zero matrices, the exact `2*eps*max(lambda_max, 1)` singular threshold, beta encoding precision, nonnegative service overhead, and real CG callback residuals.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_performance_optimizations.py tests/test_single_index_benchmark_executors.py -q`

Expected: FAIL because telemetry helpers and ADP trace fields are absent.

- [ ] **Step 3: Implement pure telemetry helpers**

```python
@dataclass(slots=True)
class WeightTelemetry:
    sum_w: np.ndarray
    sum_w2: np.ndarray
    nonzero: np.ndarray
    min_weight: np.ndarray
    max_weight: np.ndarray

    @property
    def ess(self) -> np.ndarray:
        return np.divide(self.sum_w**2, self.sum_w2, out=np.zeros_like(self.sum_w), where=self.sum_w2 > 0)

@dataclass(frozen=True, slots=True)
class LocalSystemDiagnostic:
    determinant: float
    lambda_min: float
    lambda_max: float
    condition: float
    rank: int
    residual: float
    regularization: float
    singular: bool
```

Build `A_j` and `b_j` exactly from `S`, `U @ beta`, and `imav`; use `np.linalg.eigvalsh`, the specified threshold, and infinity for a zero/singular condition number. Never replace actual solver coefficients with the diagnostic solution.

- [ ] **Step 4: Add opt-in result fields and real solver instrumentation**

Extend `ADPConfig` with `record_telemetry=False` and `record_solver_trace=False`. Extend `LocalStatistics` with optional weight vectors and per-stage durations. Extend `TrainingStep` with objective-before/after, pre-normalization norm, gradient norm, residuals, linear solver iterations/status, coefficient changes, runtime, transient beta, and residual trace. Extend `ADPResult` with `outer_telemetry` and `local_telemetry` lists.

In the NumPy and CuPy statistics paths, accumulate `sum_w2`, support, and extrema from weights already computed; do not retain weights. In `_solve_beta_default`, attach a CG callback that counts iterations and, when requested, computes `||A x_k-b||/max(||b||, eps)`. Always compute final absolute/relative residual and status from SciPy `info`. In `ADPAlgorithm`, snapshot stage-call timing at each outer iteration, compute inner metrics, aggregate local diagnostics, and retain full center rows only when `record_telemetry` is enabled.

- [ ] **Step 5: Verify GREEN and no numerical regression**

Run: `python -m pytest tests/test_performance_optimizations.py tests/test_adp.py tests/test_stage_factories.py tests/test_single_index_benchmark_executors.py -q`

Expected: PASS; existing beta/objective tests remain unchanged within their original tolerances.

- [ ] **Step 6: Commit telemetry**

```bash
git add adp/common/types.py adp/engine/algorithm.py adp/variants/random_projection.py adp/backends/numpy_backend.py adp/backends/cupy_backend.py adp/evaluation/single_index/telemetry.py tests/test_performance_optimizations.py tests/test_single_index_benchmark_executors.py
git commit -m "feat: expose ADP benchmark telemetry"
```

### Task 4: Executor statuses and normalized rows

**Files:**
- Modify: `adp/evaluation/single_index/executors.py`
- Modify: `adp/evaluation/single_index/types.py`
- Test: `tests/test_single_index_benchmark_executors.py`

- [ ] **Step 1: Write failing executor contract tests**

```python
def test_execute_job_returns_all_normalized_row_groups():
    outcome = execute_job(make_smoke_job(seed=0), make_config(diagnostic_seeds=(0,)))
    assert outcome.run_row["status"] in {"success", "nonconverged"}
    assert outcome.outer_rows
    assert outcome.inner_rows
    assert outcome.local_rows
    assert outcome.solver_rows
    assert all(row["run_id"] == outcome.run_row["run_id"] for row in outcome.outer_rows)
    assert outcome.run_row["statistics_workers"] == 1

def test_nonfinite_result_is_numerical_failure_and_keeps_partial_rows(monkeypatch):
    monkeypatch.setattr(executors, "_fit_adp", fake_nonfinite_fit)
    outcome = execute_job(make_smoke_job(), make_config())
    assert outcome.run_row["status"] == "numerical_failure"
    assert outcome.outer_rows
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_single_index_benchmark_executors.py -q`

Expected: FAIL because `RunOutcome` still exposes legacy metrics/iterations.

- [ ] **Step 3: Implement one-core ADP execution and row conversion**

Define `RunOutcome(run_row, outer_rows, inner_rows, local_rows, solver_rows)`. Build `ADPConfig(n_centers=J, n_directions=max(4, min(d, 32)), statistics_workers=1, show_progress=False, record_telemetry=True, record_solver_trace=diagnostic_seed, random_state=init_seed)`. Wrap only `model.fit(...)` in `threadpool_limits(limits=1)`. Derive truth metrics with absolute cosine and projector Frobenius error, encode outer betas, count invalid values and singular systems, and classify statuses exactly as specified. Catch numerical exceptions inside the executor so partial telemetry can be converted when available; store type/message/traceback fields without swallowing process failures unrelated to the fit contract.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_single_index_benchmark_executors.py -q`

Expected: PASS.

- [ ] **Step 5: Commit executor rewrite**

```bash
git add adp/evaluation/single_index/executors.py adp/evaluation/single_index/types.py tests/test_single_index_benchmark_executors.py
git commit -m "feat: normalize single-index run diagnostics"
```

### Task 5: Stable schemas and crash-safe storage

**Files:**
- Modify: `adp/evaluation/single_index/schema.py`
- Modify: `adp/evaluation/single_index/storage.py`
- Test: `tests/test_single_index_benchmark_schema.py`
- Test: `tests/test_single_index_benchmark_storage.py`

- [ ] **Step 1: Write failing schema and commit-last tests**

```python
PUBLIC_TABLES = {
    "run_summary": RUN_SUMMARY_COLUMNS,
    "outer_iterations": OUTER_ITERATION_COLUMNS,
    "inner_iterations": INNER_ITERATION_COLUMNS,
    "local_diagnostics": LOCAL_DIAGNOSTIC_COLUMNS,
    "solver_iterations": SOLVER_ITERATION_COLUMNS,
    "series": SERIES_COLUMNS,
    "artifacts": ARTIFACT_COLUMNS,
}

def test_every_public_schema_has_series_and_run_identity():
    assert RUN_SUMMARY_COLUMNS[:3] == ("schema_version", "series_id", "run_id")
    for name in ("outer_iterations", "inner_iterations", "local_diagnostics", "solver_iterations"):
        assert PUBLIC_TABLES[name][:3] == ("schema_version", "series_id", "run_id")

def test_run_summary_is_written_last_and_is_the_only_commit_marker(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    monkeypatch.setattr(store, "_atomic_write_rows", fail_on_run_summary)
    with pytest.raises(OSError):
        store.commit(make_outcome())
    assert store.completed_run_ids() == set()
```

Also test retry replacement, failure skip by default, incompatible resume fingerprint, stable empty headers, and merge ordering by planned run order.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_single_index_benchmark_schema.py tests/test_single_index_benchmark_storage.py -q`

Expected: FAIL because filenames and shard contracts are legacy.

- [ ] **Step 3: Define exact stable columns**

Declare complete tuples matching the four required tables and auxiliary tables. Include all effective generation parameters/sub-seeds in `run_summary.csv`; all requested aggregate, quality, and timing fields in `outer_iterations.csv`; all solver fields in `inner_iterations.csv`; all local spectral fields in `local_diagnostics.csv`; and artifact status/error fields in `artifacts.csv`.

- [ ] **Step 4: Implement atomic per-run directories**

Use `.shards/<run_id>/pending-*` temporary files and `os.replace`. Write outer, inner, local, and solver fragments first; write `run_summary.csv` last. Treat the final run row as the commit marker. On retry, write a sibling replacement directory and atomically replace individual fragments before the marker. Resume validates a canonical fingerprint that excludes `jobs` and `retry_failed` but includes profile, selectors, seeds, diagnostics, center fraction, and schema version. Merge by streaming shard CSV readers in planned job order so full-profile aggregation does not load all detailed rows at once.

- [ ] **Step 5: Verify GREEN**

Run: `python -m pytest tests/test_single_index_benchmark_schema.py tests/test_single_index_benchmark_storage.py -q`

Expected: PASS.

- [ ] **Step 6: Commit storage contract**

```bash
git add adp/evaluation/single_index/schema.py adp/evaluation/single_index/storage.py tests/test_single_index_benchmark_schema.py tests/test_single_index_benchmark_storage.py
git commit -m "feat: persist normalized benchmark shards"
```

### Task 6: CLI, process parallelism, resume, and dry-run

**Files:**
- Modify: `adp/evaluation/cli.py`
- Modify: `adp/evaluation/single_index/runner.py`
- Modify: `adp/evaluation/single_index/__init__.py`
- Test: `tests/test_single_index_benchmark_runner.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI and process-isolation tests**

```python
def test_full_dry_run_reports_27400_without_fitting(tmp_path):
    result = subprocess.run([sys.executable, "run_benchmarks.py", "single-index", "--profile", "full", "--dry-run", "--output", str(tmp_path)], capture_output=True, text=True, check=True)
    assert "total: 27400" in result.stdout
    assert not list(tmp_path.iterdir())

def test_worker_initializer_caps_every_supported_runtime(monkeypatch):
    _initialize_worker()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert os.environ[name] == "1"

def test_parallel_runner_uses_one_process_future_per_fit(tmp_path):
    saved = run_single_index_benchmark(make_config(jobs=2, max_runs=2), tmp_path)
    runs = pd.read_csv(saved["run_summary"])
    assert len(runs) == 2
    assert set(runs["statistics_workers"]) == {1}
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_single_index_benchmark_runner.py tests/test_cli.py -q`

Expected: FAIL because the new options and filenames do not exist.

- [ ] **Step 3: Implement the new CLI parser**

Expose `--profile smoke|full`, `--experiments`, `--jobs auto|N`, `--seeds`, `--diagnostic-seeds`, `--center-fraction`, `--output`, `--resume`, `--retry-failed`, `--dry-run`, `--reports-only`, and `--max-runs`. Remove `--statistics-workers`, `--base-seed`, and `--max-scenarios` from the single-index command. Require `--resume` with `--reports-only`; reject incompatible combinations through `parser.error`.

- [ ] **Step 4: Implement process dispatch and one-core enforcement**

Resolve `auto` with `max(1, os.cpu_count() or 1)`. Before constructing the pool and inside `initializer=_initialize_worker`, set all four thread environment variables to `1`. Submit one `_execute_and_commit(series_dir, job, config)` future per pending job. Each worker reopens the store by path; do not pickle a mutable store. Advance tqdm and flush `completed/total experiment=<selector> seed=<seed> status=<status>` only after the run marker exists. Preserve safe serial operation for `jobs=1`; do not silently change requested multi-process semantics except for a documented process-creation `OSError` fallback.

- [ ] **Step 5: Implement dry-run and reports-only paths**

Dry-run prints counts in selector order and returns before store creation. Reports-only opens a compatible completed/partial series and calls the CSV report builder without executing jobs. Resume skips every committed status except failed statuses when `--retry-failed` is set.

- [ ] **Step 6: Verify GREEN**

Run: `python -m pytest tests/test_single_index_benchmark_runner.py tests/test_cli.py -q`

Expected: PASS, including a real two-process smoke fixture where the host exposes at least two CPUs.

- [ ] **Step 7: Commit the CLI runner**

```bash
git add adp/evaluation/cli.py adp/evaluation/single_index/runner.py adp/evaluation/single_index/__init__.py tests/test_single_index_benchmark_runner.py tests/test_cli.py
git commit -m "feat: run benchmark fits across single-core processes"
```

### Task 7: CSV-only reports and every required PNG

**Files:**
- Create: `adp/evaluation/single_index/plots.py`
- Modify: `adp/evaluation/single_index/reports.py`
- Test: `tests/test_single_index_benchmark_reports.py`

- [ ] **Step 1: Write failing plot-manifest tests**

```python
REQUIRED_PLOTS = {
    "quality_vs_outer_iteration.png", "bandwidth_vs_outer_iteration.png",
    "rho_vs_outer_iteration.png", "beta_step_vs_outer_iteration.png",
    "objective_vs_outer_iteration.png", "objective_vs_inner_iteration.png",
    "beta_step_vs_inner_iteration.png", "solver_residual_vs_iteration.png",
    "local_mass_by_outer_iteration.png", "effective_neighbors_by_outer_iteration.png",
    "local_condition_by_outer_iteration.png", "mass_vs_condition.png",
    "local_slopes_by_outer_iteration.png", "quality_heatmap_d_nd_ratio.png",
    "success_rate_heatmap.png", "runtime_vs_dimension.png", "memory_vs_dimension.png",
    "iterations_heatmap_d_nd_ratio.png", "quality_vs_sigma_eps.png",
    "success_rate_vs_sigma_eps.png", "runtime_vs_sigma_eps.png",
    "outer_iterations_vs_sigma_eps.png", "final_objective_vs_sigma_eps.png",
    "quality_vs_correlation.png", "success_rate_vs_correlation.png",
    "local_condition_vs_correlation.png", "solver_iterations_vs_correlation.png",
    "runtime_vs_correlation.png", "quality_vs_sigma_x.png", "h0_vs_sigma_x.png",
    "final_bandwidth_vs_sigma_x.png", "local_mass_vs_sigma_x.png",
    "runtime_vs_sigma_x.png", "quality_by_link_function.png",
    "success_rate_by_link_function.png", "outer_iterations_by_link_function.png",
    "objective_by_link_function.png", "local_slopes_by_link_function.png",
    "quality_by_x_distribution.png", "quality_by_noise_distribution.png",
    "failure_rate_by_distribution.png", "runtime_by_distribution.png",
    "quality_by_heteroscedasticity.png", "quality_vs_outlier_fraction.png",
    "failure_rate_vs_outliers.png", "quality_vs_model_misspecification.png",
    "objective_vs_model_misspecification.png", "runtime_breakdown.png",
}

def test_fixture_csvs_render_every_applicable_plot(tmp_path):
    write_fixture_tables(tmp_path)
    artifacts = write_single_index_reports(tmp_path)
    created = {Path(path).name for path in artifacts.loc[artifacts.status == "created", "path"]}
    assert REQUIRED_PLOTS <= created
```

Add tests that experiment families are never pooled, bands use 5/50/95 percentiles, numerical failures count against success rate, experiment 1 uses 0.99, rerendering never invokes the executor, and one failing renderer is recorded while later renderers still run.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_single_index_benchmark_reports.py -q`

Expected: FAIL because legacy reports read legacy filenames and omit required plots.

- [ ] **Step 3: Implement focused plotting primitives**

In `plots.py`, provide `line_with_quantile_band`, `grouped_line`, `boxplot`, `scatter`, `heatmap`, and `stacked_runtime` helpers. Each accepts prepared pandas frames, explicit x/y/group columns, output path, labels, and optional log axes. Close every figure in `finally`. Use finite values only and render a labeled “no finite data” panel when a selected experiment has committed rows but no plottable points.

- [ ] **Step 4: Implement report orchestration from persisted CSV only**

Load `run_summary.csv`, `outer_iterations.csv`, `inner_iterations.csv`, `local_diagnostics.csv`, and `solver_iterations.csv`. Define a manifest mapping every required filename to selectors, input table, preparation function, and renderer. Put collision-prone diagnostic filenames below `plots/experiment_<selector>/`; summary filenames below `plots/summary/`. Attempt each applicable plot independently and rewrite `artifacts.csv` with every CSV/PNG path, size, status, and error. Never import or call `execute_job` from report code.

- [ ] **Step 5: Verify GREEN**

Run: `python -m pytest tests/test_single_index_benchmark_reports.py -q`

Expected: PASS.

- [ ] **Step 6: Commit reports**

```bash
git add adp/evaluation/single_index/plots.py adp/evaluation/single_index/reports.py tests/test_single_index_benchmark_reports.py
git commit -m "feat: render benchmark reports from CSV artifacts"
```

### Task 8: Documentation and full acceptance

**Files:**
- Modify: `README.md`
- Modify: `tests/test_cli.py`
- Modify: `docs/superpowers/plans/2026-07-18-single-index-benchmark-implementation.md`

- [ ] **Step 1: Add the final end-to-end smoke assertion**

Run the CLI with `--profile smoke --jobs 2 --max-runs 2`; assert the seven public CSVs exist, no JSON files exist, required smoke-applicable PNGs exist, every run records `statistics_workers=1`, and `reports-only` regenerates a removed PNG without changing `run_summary.csv`.

- [ ] **Step 2: Verify the smoke test fails for any remaining integration gap**

Run: `python -m pytest tests/test_cli.py::test_cli_runs_new_single_index_smoke_with_two_processes -q`

Expected: FAIL only if an integration contract remains incomplete; fix through a focused RED/GREEN cycle in the owning module.

- [ ] **Step 3: Document exact commands and artifacts**

Add README examples for full, selected experiments, custom seeds, resume/retry, reports-only, dry-run, and smoke. State clearly: `--jobs` controls independent worker processes; `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` are capped at one; `threadpoolctl` wraps each fit; `ADPConfig.statistics_workers` is always one. List `run_summary.csv`, `outer_iterations.csv`, `inner_iterations.csv`, `local_diagnostics.csv`, `solver_iterations.csv`, `series.csv`, and `artifacts.csv`.

- [ ] **Step 4: Run focused and full verification**

```bash
python run_benchmarks.py single-index --profile full --dry-run
python run_benchmarks.py single-index --profile smoke --jobs 2 --max-runs 2 --output /tmp/adp_new_benchmark_smoke
python -m pytest -q
git diff --check
```

Expected: dry-run total `27400`; real smoke exits zero and writes normalized CSV/PNG artifacts; all tests pass; diff check emits no output.

- [ ] **Step 5: Inspect repository scope and commit the final slice**

Run `git status --short` and confirm the pre-existing deleted logs, deleted `single_index_adp_benchmark.md`, and untracked `new_benchmark.md` were not staged or modified by this implementation.

```bash
git add README.md tests/test_cli.py docs/superpowers/plans/2026-07-18-single-index-benchmark-implementation.md
git commit -m "docs: describe new single-index benchmark CLI"
```
