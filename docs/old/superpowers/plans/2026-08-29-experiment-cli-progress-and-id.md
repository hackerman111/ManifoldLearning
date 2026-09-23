# Experiment CLI Progress and ID Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fit-level `tqdm` progress to the experiment CLI and group every selector from one CLI invocation under one generated experiment ID.

**Architecture:** `main()` generates one timestamp ID and passes it through the existing `Experiment.run()`/`run_experiment()` path. The runner writes each selector directly under `<output>/<id>/<selector>`, records the ID in `series.json`, and enables a disabled-by-default `tqdm` counter around the existing fit loops.

**Tech Stack:** Python 3.11+, argparse, pathlib, tqdm, pytest, uv, Ruff, Pyright

---

### Task 1: Record the new CLI and artifact contract

**Files:**
- Modify: `tests/test_experiment.py`

- [x] **Step 1: Extend the artifact test with ID/path assertions**

Replace the existing manifest seed assertion in
`test_paired_ab_writes_log_and_plots` with:

```python
manifest = json.loads((series / "series.json").read_text())
assert series == tmp_path / manifest["experiment_id"] / "custom"
assert manifest["seed"] == 11
```

- [x] **Step 2: Verify that CLI selectors share one ID and enable progress**

Add `from pathlib import Path` and this focused test:

```python
def test_cli_groups_experiments_under_one_id(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, str, bool]] = []

    def fake_run(self: Experiment, *args: object, **kwargs: object) -> Path:
        experiment_id = kwargs["experiment_id"]
        progress = kwargs["progress"]
        assert isinstance(experiment_id, str)
        assert isinstance(progress, bool)
        calls.append((self.selector, experiment_id, progress))
        series = tmp_path / experiment_id / self.selector.replace(".", "_")
        series.mkdir(parents=True)
        (series / "runs.csv").write_text("status\nsuccess\n", encoding="utf-8")
        return series

    monkeypatch.setattr(Experiment, "run", fake_run)

    assert (
        experiment_main(
            [
                "--experiment",
                "1,2",
                "--output-dir",
                str(tmp_path),
                "--no-plots",
            ]
        )
        == 0
    )
    assert [selector for selector, _, _ in calls] == ["1", "2"]
    assert len({experiment_id for _, experiment_id, _ in calls}) == 1
    assert all(progress for _, _, progress in calls)
```

- [x] **Step 3: Verify the tests fail before implementation**

Run:

```bash
UV_CACHE_DIR=/tmp/adp-uv-cache rtk uv run --no-sync pytest -q \
  tests/test_experiment.py::test_paired_ab_writes_log_and_plots \
  tests/test_experiment.py::test_cli_groups_experiments_under_one_id
```

Expected: FAIL because the manifest has no `experiment_id` and
`Experiment.run()` does not receive shared `experiment_id`/`progress` values.

### Task 2: Implement ID grouping and fit progress

**Files:**
- Modify: `ADP/experiment.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Test: `tests/test_experiment.py`

- [x] **Step 1: Declare tqdm as a runtime dependency**

Add `"tqdm>=4.67"` to `[project].dependencies` in `pyproject.toml`, then update
the lock and environment:

```bash
UV_CACHE_DIR=/tmp/adp-uv-cache rtk uv lock
UV_CACHE_DIR=/tmp/adp-uv-cache rtk uv sync
```

Expected: `uv.lock` records `tqdm` and sync completes.

- [x] **Step 2: Add the ID and progress parameters**

Import tqdm:

```python
from tqdm.auto import tqdm
```

Add the following keyword-only parameters to both `Experiment.run()` and
`run_experiment()`, and forward them unchanged from the method:

```python
def run(
    self,
    *,
    experiment_id: str | None = None,
    progress: bool = False,
) -> Path: ...
```

Resolve the ID once at the start of `run_experiment()`:

```python
experiment_id = _experiment_id(experiment_id)
series_dir = _series_directory(Path(output_dir), experiment.selector, experiment_id)
_write_manifest(series_dir, experiment, builds, profile, runs, seed, experiment_id)
```

- [x] **Step 3: Create and validate readable IDs**

Replace the old timestamp-per-selector helper with:

```python
def _experiment_id(value: str | None = None) -> str:
    if value is None:
        return datetime.now().strftime("%Y%m%dT%H%M%S%f")
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError("experiment_id must be a path-safe name")
    return value


def _series_directory(output_dir: Path, selector: str, experiment_id: str) -> Path:
    path = output_dir / experiment_id / selector.replace(".", "_")
    path.mkdir(parents=True)
    return path
```

Add `experiment_id: str` to `_write_manifest()` and add this manifest field:

```python
"experiment_id": experiment_id,
```

- [x] **Step 4: Count every persisted fit**

Wrap the existing nested point/run/build loops in:

```python
with tqdm(
    total=len(points) * runs * len(builds),
    desc=experiment.selector,
    unit="fit",
    disable=not progress,
) as progress_bar:
```

Call `progress_bar.update()` immediately after each `_append_row()` in both
the data-generation failure branch and the ordinary fit branch. Do not change
the existing status/error behavior.

- [x] **Step 5: Share one ID from the CLI**

In `main()`, create the ID once after selector validation:

```python
experiment_id = _experiment_id()
```

Pass these arguments to every `experiment.run()` call:

```python
path = experiment.run(
    build,
    experiment_id=experiment_id,
    progress=True,
)
```

- [x] **Step 6: Verify focused behavior**

Run:

```bash
UV_CACHE_DIR=/tmp/adp-uv-cache rtk uv run --no-sync pytest -q tests/test_experiment.py
```

Expected: all experiment tests PASS.

### Task 3: Verify integration and commit the narrow diff

**Files:**
- Modify: `ADP/experiment.py`
- Modify: `tests/test_experiment.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [x] **Step 1: Run configured checks**

```bash
UV_CACHE_DIR=/tmp/adp-uv-cache rtk uv run --no-sync ruff format --check .
UV_CACHE_DIR=/tmp/adp-uv-cache rtk uv run --no-sync ruff check .
UV_CACHE_DIR=/tmp/adp-uv-cache rtk uv run --no-sync pyright
UV_CACHE_DIR=/tmp/adp-uv-cache rtk uv run --no-sync pytest
```

Expected: all commands exit zero. If an unrelated dirty-tree file fails, also
run and report the focused checks for the four task files without modifying
or reverting that unrelated work.

- [x] **Step 2: Smoke the real CLI without numerical work**

```bash
UV_CACHE_DIR=/tmp/adp-uv-cache rtk uv run --no-sync python -m ADP.experiment --list
```

Expected: the catalog is printed and no output directory is created.

- [x] **Step 3: Commit only task-owned files**

```bash
rtk git add ADP/experiment.py tests/test_experiment.py pyproject.toml uv.lock \
  docs/superpowers/plans/2026-08-29-experiment-cli-progress-and-id.md
rtk git commit -m "feat: track experiment progress by run id"
```

Expected: unrelated dirty files remain unstaged and unchanged.
