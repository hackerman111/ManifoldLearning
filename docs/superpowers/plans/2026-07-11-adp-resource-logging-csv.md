# ADP Resource Logging and CSV Experiment Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure algorithm and end-to-end RSS/time and migrate maintained experiment-series metadata from JSON to normalized CSV tables.

**Architecture:** A lightweight sampling context manager aggregates process RSS online and exposes a flat metric dictionary. Core `fit()` stores algorithm metrics in `ADPResult`; experiment runners add full-run metrics and persist normalized run/iteration/series/artifact tables through a shared CSV utility.

**Tech Stack:** Python standard library (`csv`, `threading`, `time`, `/proc` and `resource` fallbacks), optional `psutil`, NumPy, pandas, pytest.

---

### Task 1: Process RSS sampler

**Files:**
- Create: `adp/common/resource_monitor.py`
- Modify: `adp/common/__init__.py`
- Create: `tests/test_resource_logging.py`

- [ ] **Step 1: Write failing sampler tests**

```python
def test_resource_monitor_reports_ordered_rss_and_elapsed_time():
    with ResourceMonitor(sample_interval_sec=0.001) as monitor:
        time.sleep(0.004)
    usage = monitor.usage
    assert usage.elapsed_sec > 0.0
    assert usage.samples >= 2
    assert 0.0 < usage.rss_min_mib <= usage.rss_mean_mib <= usage.rss_max_mib
    assert usage.rss_peak_delta_mib >= 0.0


def test_resource_usage_flattens_with_prefix():
    usage = ResourceUsage(...)
    assert usage.to_dict("algorithm")["algorithm_time_sec"] == usage.elapsed_sec
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_resource_logging.py -q`

Expected: FAIL because `adp.common.resource_monitor` does not exist.

- [ ] **Step 3: Implement the monitor**

Implement:

```python
@dataclass(frozen=True, slots=True)
class ResourceUsage:
    elapsed_sec: float
    rss_start_mib: float
    rss_min_mib: float
    rss_mean_mib: float
    rss_max_mib: float
    rss_peak_delta_mib: float
    samples: int
    source: str

    def to_dict(self, prefix: str) -> dict[str, float | int | str]: ...


class ResourceMonitor:
    def __enter__(self) -> ResourceMonitor: ...
    def __exit__(self, exc_type, exc, traceback) -> None: ...
```

Use an online accumulator guarded by a lock. Read RSS from `psutil.Process(os.getpid()).memory_info().rss`, then `/proc/self/statm`, then `resource.getrusage`. Always sample synchronously at entry and exit.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_resource_logging.py -q`

Expected: all Task 1 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add adp/common/resource_monitor.py adp/common/__init__.py tests/test_resource_logging.py
git commit -m "feat: add process resource monitor"
```

### Task 2: Core algorithm resource metrics

**Files:**
- Modify: `adp/common/types.py:186-209`
- Modify: `adp/common/result_store.py:11-54`
- Modify: `adp/engine/algorithm.py:65-298`
- Modify: `adp/engine/diagnostics.py:42-75`
- Modify: `tests/test_resource_logging.py`

- [ ] **Step 1: Write failing core integration tests**

```python
def test_fit_exposes_algorithm_resource_usage():
    model = ADP.create("new", ADPConfig(..., show_progress=False))
    data = model.generate_data(...)
    result = model.fit(data.X, data.y, centers=data.centers, directions=data.directions)
    usage = result.resource_usage
    assert usage["algorithm_time_sec"] > 0.0
    assert usage["algorithm_rss_min_mib"] <= usage["algorithm_rss_mean_mib"]
    assert usage["algorithm_rss_mean_mib"] <= usage["algorithm_rss_max_mib"]
    assert model.summary()["resource_usage"] == usage


def test_failed_fit_retains_last_algorithm_resource_usage():
    with pytest.raises(ValueError):
        model.fit(np.ones((5, 2)), np.ones(4))
    assert model.last_resource_usage_["algorithm_time_sec"] > 0.0
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_resource_logging.py -q`

Expected: FAIL because `resource_usage` and `last_resource_usage_` are absent.

- [ ] **Step 3: Add result storage and wrap the whole fit**

Add to `ADPResult`:

```python
resource_usage: dict[str, float | int | str] = field(default_factory=dict)
```

Wrap the complete `ADPAlgorithm.fit()` body in a resource monitor using a small private `_fit_impl()` extraction so every success and failure reaches one `finally`. Store the flattened result on `model.last_resource_usage_`; copy it into the successful `ADPResult`. Keep `timings["total"]` unchanged for compatibility.

- [ ] **Step 4: Expose metrics through summary**

Add `"resource_usage": dict(result.resource_usage)` to `DiagnosticsMixin.summary()`.

- [ ] **Step 5: Verify GREEN and core regressions**

Run: `python -m pytest tests/test_resource_logging.py tests/test_adp.py tests/test_stage_factories.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add adp/common/types.py adp/common/result_store.py adp/engine/algorithm.py adp/engine/diagnostics.py tests/test_resource_logging.py
git commit -m "feat: log ADP algorithm resources"
```

### Task 3: Shared normalized CSV writer

**Files:**
- Create: `adp/common/experiment_log.py`
- Modify: `adp/common/__init__.py`
- Create: `tests/test_experiment_csv_log.py`

- [ ] **Step 1: Write failing CSV schema tests**

```python
def test_csv_table_appends_rows_with_stable_header(tmp_path):
    table = CSVTable(tmp_path / "runs.csv", ("schema_version", "run_id", "failed"))
    table.append({"run_id": "r1", "failed": False})
    table.append({"run_id": "r2", "failed": True})
    frame = pd.read_csv(table.path)
    assert list(frame["run_id"]) == ["r1", "r2"]
    assert set(frame["schema_version"]) == {1}


def test_flatten_mapping_never_writes_json_cells():
    flat = flatten_mapping({"config": {"outer_steps": 2}, "methods": ("a", "b")})
    assert flat == {"config_outer_steps": 2, "methods": "a|b"}
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_experiment_csv_log.py -q`

Expected: FAIL because `experiment_log` does not exist.

- [ ] **Step 3: Implement flat CSV helpers**

Implement `flatten_mapping`, `stable_run_id`, `CSVTable.append/append_many`, `write_single_row_csv`, and `write_artifacts_csv`. Reject dict/list values that remain after flattening; encode scalar sequences with `|`, not JSON.

- [ ] **Step 4: Add shard merge behavior**

Implement `merge_csv_shards(paths, destination, fieldnames)` by streaming through `csv.DictReader`/`DictWriter`. Validate every shard header before appending and delete shards only after a successful destination replace.

- [ ] **Step 5: Verify GREEN**

Run: `python -m pytest tests/test_experiment_csv_log.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add adp/common/experiment_log.py adp/common/__init__.py tests/test_experiment_csv_log.py
git commit -m "feat: add normalized CSV experiment log"
```

### Task 4: Confirmatory series full-run logging and CSV migration

**Files:**
- Modify: `experiments/adp_confirmatory_common.py:293-494,721-876`
- Modify: `tests/test_confirmatory_experiments.py`

- [ ] **Step 1: Replace JSON expectations with failing CSV contract tests**

Update the runner test to require keys and files:

```python
assert {"runs", "iterations", "initial_parameters", "summary", "final_success", "series", "artifacts"} <= saved.keys()
assert saved["runs"].suffix == ".csv"
assert saved["initial_parameters"].suffix == ".csv"
assert saved["series"].suffix == ".csv"
assert not list(tmp_path.glob("*.json"))
```

Check `runs.csv` contains algorithm/full-run min/mean/max RSS, both times, persistence time, samples, status, seeds and `run_id`. Check `iterations.csv` contains `run_id`.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_confirmatory_experiments.py -q`

Expected: FAIL on missing CSV files and resource columns.

- [ ] **Step 3: Flatten initial parameters**

Replace `save_initial_parameters()` JSON payload with one CSV row per `RunJob`. Flatten scenario, `ADPConfig`, seeds and beta initializer metadata into scalar columns.

- [ ] **Step 4: Return a structured job outcome**

Add a picklable `JobOutcome(iterations, run)` dataclass. `run_job()` starts `ResourceMonitor` before data generation, captures core `result.resource_usage`, computes rows, persists iteration rows to its PID shard when a shard directory is passed, then stops the full monitor and builds the run row. Failure follows the same path.

- [ ] **Step 5: Stream sequential and parallel outcomes**

The parent stores only run rows needed for summary metadata, merges iteration shards to `<prefix>_iterations.csv`, and reads the merged file only for existing summary/plot generation. Preserve the current progress lines and process-pool fallback.

- [ ] **Step 6: Write series and artifact tables**

Replace manifest JSON with one `<prefix>_series.csv` row and `<prefix>_artifacts.csv`. Do not create JSON. Keep PNG plots.

- [ ] **Step 7: Add resource aggregates to summary**

Join final iteration rows with `runs.csv` by `run_id` and add median algorithm/full time plus RSS min/mean/max columns to each scenario/method summary.

- [ ] **Step 8: Verify GREEN including CLI smoke tests**

Run: `python -m pytest tests/test_confirmatory_experiments.py -q`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add experiments/adp_confirmatory_common.py tests/test_confirmatory_experiments.py
git commit -m "feat: log confirmatory runs to CSV"
```

### Task 5: Benchmark and stress resource fields

**Files:**
- Modify: `adp/evaluation/runner.py:106-174`
- Modify: `adp/evaluation/reports.py:75-124`
- Modify: `adp/evaluation/stress.py:609-840`
- Modify: `tests/test_benchmarks.py`
- Modify: `tests/test_stress_adp_single_index.py`

- [ ] **Step 1: Write failing benchmark/stress field tests**

Require algorithm and full-run time/RSS fields in benchmark and stress records. Replace stress `manifest.json` expectations with `series.csv` and `artifacts.csv`, and assert no JSON is emitted.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_benchmarks.py tests/test_stress_adp_single_index.py -q`

Expected: FAIL on the new columns/files.

- [ ] **Step 3: Replace `tracemalloc` windows**

Use `ResourceMonitor` around each complete benchmark/stress case. For ADP copy the algorithm metrics from `ADPResult`; for baselines set algorithm metrics equal to their method-call window. Preserve `fit_time_sec` and `peak_memory_kib` as aliases for compatibility.

- [ ] **Step 4: Extend summaries**

Aggregate `algorithm_time_sec`, `full_run_time_sec`, and min/mean/max RSS fields. Retain old plots and use the new maximum full-run RSS as the memory plot source.

- [ ] **Step 5: Replace stress manifest JSON**

Write `adp_single_index_stress_series.csv` and `adp_single_index_stress_artifacts.csv`; update returned keys and CLI output.

- [ ] **Step 6: Verify GREEN**

Run: `python -m pytest tests/test_benchmarks.py tests/test_stress_adp_single_index.py tests/test_cli.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add adp/evaluation/runner.py adp/evaluation/reports.py adp/evaluation/stress.py tests/test_benchmarks.py tests/test_stress_adp_single_index.py tests/test_cli.py
git commit -m "feat: add resource metrics to benchmark runs"
```

### Task 6: NumPy statistics benchmark CSV output

**Files:**
- Modify: `experiments/benchmark_numpy_statistics.py`
- Modify: `tests/test_statistics_benchmark.py`

- [ ] **Step 1: Write failing flat CSV benchmark test**

Require one row per case/repetition, scalar shape columns, timing and RSS min/mean/max columns. CLI `--output` must create CSV and reject a `.json` suffix.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_statistics_benchmark.py -q`

Expected: FAIL because output is nested JSON.

- [ ] **Step 3: Replace nested result and `tracemalloc`**

Return flat records with `n`, `d`, `n_centers`, `n_directions`, `repetition`, `elapsed_sec`, RSS statistics and output shapes. Measure every warmed statistics call with `ResourceMonitor`.

- [ ] **Step 4: Write CSV from CLI**

Use `csv.DictWriter`; print the output path and compact aggregate statistics rather than dumping JSON to stdout.

- [ ] **Step 5: Verify GREEN**

Run: `python -m pytest tests/test_statistics_benchmark.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add experiments/benchmark_numpy_statistics.py tests/test_statistics_benchmark.py
git commit -m "feat: save statistics benchmarks as CSV"
```

### Task 7: Documentation and full verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the measurement contract and CSV tables**

Describe absolute process RSS in MiB, the algorithm/full-run boundaries, the `statistics_workers=2` opt-in, CSV output names, and the intentional removal of maintained JSON series artifacts.

- [ ] **Step 2: Run focused verification**

Run:

```bash
python -m pytest tests/test_resource_logging.py tests/test_experiment_csv_log.py tests/test_confirmatory_experiments.py tests/test_benchmarks.py tests/test_stress_adp_single_index.py tests/test_statistics_benchmark.py tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 3: Run full verification**

Run:

```bash
python -m pytest -q
git diff --check
git status --short
```

Expected: full suite PASS; no whitespace errors; only intended source/test/README changes plus the user's pre-existing generator changes.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md
git commit -m "docs: describe ADP resource CSV logs"
```
