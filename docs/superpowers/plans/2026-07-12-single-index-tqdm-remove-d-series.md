# Single-index tqdm and D-series Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove D01-D04 from the executable single-index benchmark catalog and report persisted job completion through one tqdm progress bar.

**Architecture:** Keep the real-data loader and executor as reusable infrastructure, but remove D scenarios and their CLI/documentation surface. Centralize progress advancement in the single-index runner so serial, process-pool, resume, and fallback paths update the same bar while retaining flushed newline logs.

**Tech Stack:** Python, pytest, tqdm, pandas, argparse.

---

### Task 1: Remove D-series scenarios from the benchmark catalog

**Files:**
- Modify: `tests/test_single_index_benchmark_scenarios.py`
- Modify: `tests/test_single_index_benchmark_executors.py`
- Modify: `adp/evaluation/single_index/scenarios.py`

- [ ] **Step 1: Write the failing registry test**

Remove D01-D04 from the expected registry set and add explicit profile assertions:

```python
def test_registry_and_profiles_exclude_d_series():
    assert all(not scenario.scenario_id.startswith("D") for scenario in scenario_registry())
    assert all(
        not scenario_id.startswith("D")
        for scenario_ids in PROFILE_IDS.values()
        for scenario_id in scenario_ids
    )
```

Change the real-data executor test to construct its reusable infrastructure fixture directly instead of retrieving an executable D scenario:

```python
scenario = SingleIndexScenario(
    scenario_id="D01",
    family="D",
    executor="real_data",
    hypothesis="real-data infrastructure fixture",
    data={"dataset": "D01", "folds": 5},
    algorithm={"n_centers": 2, "n_directions": 2, "min_neighbors": 1.0},
    solver={"outer_steps": 1, "inner_steps": 1},
    repeats=1,
    methods=("full_adp",),
)
```

- [ ] **Step 2: Run the registry test and verify RED**

Run: `python -m pytest tests/test_single_index_benchmark_scenarios.py::test_registry_and_profiles_exclude_d_series -q`

Expected: FAIL because D01-D04 are still returned by `scenario_registry()` and included in `full` and `publication`.

- [ ] **Step 3: Remove executable D definitions**

In `scenarios.py`, remove D01-D04 from `_TITLES`, remove the `family == "D"` executor branch, and remove the D-specific data/repeats/methods branch. Leave generic real-data types, loader, and executor unchanged.

- [ ] **Step 4: Run focused scenario and executor tests**

Run: `python -m pytest tests/test_single_index_benchmark_scenarios.py tests/test_single_index_benchmark_executors.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the catalog change**

```bash
git add adp/evaluation/single_index/scenarios.py tests/test_single_index_benchmark_scenarios.py tests/test_single_index_benchmark_executors.py
git commit -m "refactor: remove D-series benchmark scenarios"
```

### Task 2: Add one tqdm bar for persisted single-index jobs

**Files:**
- Modify: `tests/test_single_index_benchmark_runner.py`
- Modify: `adp/evaluation/single_index/runner.py`

- [ ] **Step 1: Write the failing serial progress test**

Add a fake progress implementation and assert the public runner contract:

```python
def test_runner_reports_persisted_jobs_with_tqdm(tmp_path, monkeypatch):
    calls = []

    class FakeProgress:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.n = 0
            self.postfixes = []
            calls.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, amount=1):
            self.n += amount

        def set_postfix(self, values, refresh=True):
            self.postfixes.append((values, refresh))

    monkeypatch.setattr(single_runner, "tqdm", FakeProgress)
    run_single_index_benchmark(make_config(max_scenarios=1), tmp_path)

    progress = calls[0]
    assert progress.kwargs == {
        "total": 1,
        "desc": "single-index",
        "unit": "job",
        "dynamic_ncols": True,
    }
    assert progress.n == 1
    assert progress.postfixes[-1][0] == {"scenario": "C01", "method": "full_adp"}
```

- [ ] **Step 2: Run the serial progress test and verify RED**

Run: `python -m pytest tests/test_single_index_benchmark_runner.py::test_runner_reports_persisted_jobs_with_tqdm -q`

Expected: FAIL because `adp.evaluation.single_index.runner` does not expose or create `tqdm`.

- [ ] **Step 3: Implement the minimal shared progress path**

Import `tqdm` from `tqdm.auto`. Create one context-managed bar in `run_single_index_benchmark` after pending jobs are known:

```python
with tqdm(
    total=total,
    desc="single-index",
    unit="job",
    dynamic_ncols=True,
) as progress:
    # serial or process-pool dispatch
```

Pass the bar to `_run_serial`. Replace duplicated completion logic with:

```python
def _mark_job_done(progress, completed, total, job):
    completed += 1
    progress.set_postfix(
        {"scenario": job.scenario.scenario_id, "method": job.method},
        refresh=True,
    )
    progress.update(1)
    _log_progress(completed, total, job)
    return completed
```

On process-pool fallback, recalculate committed completion from pending jobs and synchronize the bar before continuing:

```python
remaining = list(store.pending_jobs(jobs))
completed = total - len(remaining)
if progress.n < completed:
    progress.update(completed - progress.n)
_run_serial(store, remaining, config, completed, total, progress)
```

- [ ] **Step 4: Add fallback accounting assertions**

Extend `test_process_pool_oserror_falls_back_and_logs_progress` with the fake bar and assert `progress.n == 1`, one C01/full_adp postfix, and the existing flushed `1/1` progress line.

- [ ] **Step 5: Add a process-pool completion test**

Use an immediate in-process pool so the real parallel dispatch branch is tested
without starting child processes:

```python
class ImmediateFuture:
    def result(self):
        return None

class ImmediatePool:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, function, *args):
        function(*args)
        return ImmediateFuture()

monkeypatch.setattr(single_runner, "ProcessPoolExecutor", ImmediatePool)
monkeypatch.setattr(single_runner, "as_completed", lambda futures: iter(futures))
monkeypatch.setattr(single_runner, "tqdm", FakeProgress)
run_single_index_benchmark(make_config(jobs=2, max_scenarios=1), tmp_path)
assert calls[0].n == 1
assert calls[0].postfixes[-1][0] == {"scenario": "C01", "method": "full_adp"}
```

- [ ] **Step 6: Run runner tests**

Run: `python -m pytest tests/test_single_index_benchmark_runner.py -q`

Expected: PASS with the serial, resume, failure, and process-pool fallback paths green.

- [ ] **Step 7: Commit tqdm progress**

```bash
git add adp/evaluation/single_index/runner.py tests/test_single_index_benchmark_runner.py
git commit -m "feat: log single-index benchmark progress with tqdm"
```

### Task 3: Remove D-series CLI and documentation surface

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `adp/evaluation/cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing CLI contract**

Rename the help test and assert D-series controls are absent:

```python
def test_single_index_help_exposes_series_controls_without_d_series():
    result = subprocess.run(
        [sys.executable, "run_benchmarks.py", "single-index", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for option in (
        "--profile",
        "--jobs",
        "--statistics-workers",
        "--resume",
        "--retry-failed",
        "--max-scenarios",
    ):
        assert option in result.stdout
    assert "--data-dir" not in result.stdout
    assert "--allow-download" not in result.stdout
    assert "D01" not in result.stdout
    assert "adp_D1_data" not in result.stdout
```

- [ ] **Step 2: Run the CLI test and verify RED**

Run: `python -m pytest tests/test_cli.py::test_single_index_help_exposes_series_controls_without_d_series -q`

Expected: FAIL because the two D-series options and `adp_D1_data` are still advertised.

- [ ] **Step 3: Remove obsolete CLI arguments and README text**

Delete `--data-dir` and `--allow-download` from `build_single_index_parser`, and stop passing them in `run_single_index_command`; the config defaults remain available for direct infrastructure tests. Remove `--data-dir adp_D1_data` and the D01-D04 paragraph from the README example.

- [ ] **Step 4: Run CLI tests**

Run: `python -m pytest tests/test_cli.py -q`

Expected: PASS.

- [ ] **Step 5: Commit CLI and documentation cleanup**

```bash
git add adp/evaluation/cli.py tests/test_cli.py README.md
git commit -m "docs: remove D-series benchmark interface"
```

### Task 4: Verify the complete change

**Files:**
- Verify only; do not modify unrelated user files.

- [ ] **Step 1: Run the focused benchmark suite**

Run: `python -m pytest tests/test_single_index_benchmark_scenarios.py tests/test_single_index_benchmark_executors.py tests/test_single_index_benchmark_runner.py tests/test_cli.py -q`

Expected: PASS.

- [ ] **Step 2: Run the full test suite**

Run: `python -m pytest -q`

Expected: PASS.

- [ ] **Step 3: Run a real CLI smoke check**

Run: `python run_benchmarks.py single-index --profile smoke --max-scenarios 1 --output /tmp/adp_single_index_tqdm_smoke`

Expected: stderr displays a `single-index` tqdm bar reaching `1/1` plus the flushed `1/1 scenario=C01 method=full_adp` line; output CSV and reports are created.

- [ ] **Step 4: Check formatting and scope**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; the pre-existing deleted log files and untracked `adp_D1_data/` remain untouched.
