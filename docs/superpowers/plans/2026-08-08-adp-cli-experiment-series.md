# ADP CLI Experiment Series Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add repeated-seed experiment execution, incremental CSV persistence, and automatic convergence plots to the uppercase ADP CLI.

**Architecture:** Keep argument parsing in `ADP/cli.py`. Put one public `Scenario` dataclass and its private CSV/plot helpers in `ADP/single_index/scenario.py`; the scenario constructs fresh data and models for each seed, persists each completed run, then aggregates `result.trace` by outer iteration. Exercise the full path with a small integration test in `test_ADP_cli_experiments.py`.

**Tech Stack:** Python dataclasses, `csv`, `pathlib`, NumPy, Matplotlib, pytest.

---

### Task 1: Define the Scenario integration contract

**Files:**
- Create: `test_ADP_cli_experiments.py`
- Read: `ADP/ADP_Config.py`
- Read: `ADP/single_index/ADP_single_index.py`

- [ ] **Step 1: Write the failing integration test**

```python
import csv
import json

import numpy as np

from ADP import ADP_Config
from ADP.single_index.scenario import Scenario


def test_scenario_saves_runs_trace_and_plots(tmp_path):
    scenario = Scenario(
        n=40,
        d=3,
        noise=0.05,
        runs=2,
        seed=7,
        config=ADP_Config(
            seed=7,
            N_loc=5,
            N_lin=8,
            N_J=20,
            N_phi=3,
            h_min=1.0,
        ),
        output_dir=tmp_path,
    )

    series_dir, failed_runs = scenario.run()

    assert failed_runs == 0
    with (series_dir / "runs.csv").open(newline="", encoding="utf-8") as stream:
        runs = list(csv.DictReader(stream))
    with (series_dir / "outer_iterations.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        outer = list(csv.DictReader(stream))

    assert [int(row["seed"]) for row in runs] == [7, 8]
    assert all(row["status"] == "success" for row in runs)
    assert outer
    assert {"outer_k", "h_k", "rho_k", "cosine_abs", "beta"} <= set(outer[0])

    for row in outer:
        rng = np.random.default_rng(int(row["seed"]))
        rng.normal(size=(40, 3))
        beta_true = rng.normal(size=3)
        beta_true /= np.linalg.norm(beta_true)
        beta = np.asarray(json.loads(row["beta"]), dtype=float)
        expected = abs(float(beta @ beta_true))
        assert np.isclose(float(row["cosine_abs"]), expected)

    for name in (
        "quality_vs_outer_iteration.png",
        "bandwidth_vs_outer_iteration.png",
        "rho_vs_outer_iteration.png",
    ):
        assert (series_dir / name).stat().st_size > 0
```

- [ ] **Step 2: Run the test and verify the missing module failure**

Run:

```bash
rtk python -m pytest test_ADP_cli_experiments.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'ADP.single_index.scenario'`.

### Task 2: Implement Scenario execution and durable CSV rows

**Files:**
- Create: `ADP/single_index/scenario.py`
- Test: `test_ADP_cli_experiments.py`

- [ ] **Step 1: Add the Scenario type, validation, and per-seed execution**

Implement this public boundary:

```python
@dataclass(frozen=True, slots=True)
class Scenario:
    n: int
    d: int
    noise: float
    runs: int
    seed: int
    config: ADP_Config
    output_dir: Path

    def run(self) -> tuple[Path, int]:
        series_dir = _create_series_dir(self.output_dir)
        _write_header(series_dir / "runs.csv", RUN_COLUMNS)
        _write_header(series_dir / "outer_iterations.csv", OUTER_COLUMNS)
        failed_runs = 0
        for run_index in range(self.runs):
            run_row, outer_rows = self._run_once(run_index, self.seed + run_index)
            _append_rows(series_dir / "runs.csv", RUN_COLUMNS, (run_row,))
            _append_rows(
                series_dir / "outer_iterations.csv", OUTER_COLUMNS, outer_rows
            )
            failed_runs += run_row["status"] != "success"
        _create_plots(series_dir)
        return series_dir, int(failed_runs)
```

`_run_once` must use a fresh `np.random.default_rng(seed)`, generate `X`, normalized `beta_true`, and `Y` in the same order as the existing CLI, and call a fresh `ADP_single_index(replace(self.config, seed=seed))`. After fitting it must call `result.Set_beta_true(beta_true)` and `result.Calculate_cosine()`.

For every trace item, write:

```python
{
    "run_index": run_index,
    "seed": seed,
    "outer_k": int(item["k"]),
    "h_k": float(item["h"]),
    "rho_k": float(item["rho"]),
    "cosine_abs": float(abs(np.asarray(item["beta"]) @ beta_true)),
    "mean_mass": float(item["mean_mass"]),
    "beta": json.dumps(np.asarray(item["beta"]).tolist(), separators=(",", ":")),
    "inner_iterations": item.get("inner_iterations", ""),
    "lsmr_stop": item.get("lsmr_stop", ""),
    "lsmr_iterations": item.get("lsmr_iterations", ""),
    "beta_delta": item.get("beta_delta", ""),
    "stop_reason": item.get("stop_reason", ""),
}
```

Catch `Exception` per run, write `status="error"`, the exception type/message, and no outer rows. Initialize each CSV before the first fit and reopen it in append mode after each run so completed rows survive later failures.

- [ ] **Step 2: Add fixed schemas and serialization helpers**

Define stable `RUN_COLUMNS` and `OUTER_COLUMNS`. `RUN_COLUMNS` must include run identity, every current `ADP_Config` input, status/stop reason, iteration count, cosine metrics, fit time, peak traced memory, and error text. Serialize the kernel as `kernel.__name__` and nullable values as empty CSV cells.

Use only `csv.DictWriter`:

```python
def _write_header(path: Path, columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=columns).writeheader()


def _append_rows(path: Path, columns: tuple[str, ...], rows) -> None:
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writerows(rows)
```

- [ ] **Step 3: Run the test to reach the plotting failure**

Run:

```bash
rtk python -m pytest test_ADP_cli_experiments.py -q
```

Expected: the scenario produces both CSV files and fails only because the plot helper is not implemented or PNG files are absent.

### Task 3: Generate the three main-compatible convergence plots

**Files:**
- Modify: `ADP/single_index/scenario.py`
- Test: `test_ADP_cli_experiments.py`
- Reference: `main:adp/evaluation/single_index/reports.py`
- Reference: `main:adp/evaluation/single_index/plots.py`
- Reference: `main:adp/common/plotting.py`

- [ ] **Step 1: Read outer rows and aggregate each metric**

Implement `_metric_quantiles(rows, metric)` by grouping finite values by integer `outer_k`, sorting iterations, and returning NumPy arrays for `outer_k`, median, 5% quantile, and 95% quantile. Do not fabricate values for iterations with no observations.

- [ ] **Step 2: Implement one reusable quantile-band plotter**

Use these exact report parameters:

```python
PLOTS = (
    ("quality_vs_outer_iteration.png", "cosine_abs",
     "Качество направления по внешним итерациям",
     "Абсолютный косинус направления", "linear", (0.0, 1.0), 0.99),
    ("bandwidth_vs_outer_iteration.png", "h_k",
     "Ширина окна по внешним итерациям",
     "Ширина окна h (логарифмическая шкала)", "log", None, None),
    ("rho_vs_outer_iteration.png", "rho_k",
     "Анизотропия по внешним итерациям",
     "Параметр анизотропии ρ", "linear", (0.0, 1.0), None),
)
```

Create a `10.5 × 6.2` figure, draw the median as a blue line with circle markers, fill the 5–95% interval at alpha `0.18`, use the subtitle `медиана и интервал 5–95% по запускам`, style the background/spines/grid with the constants from `main:adp/common/plotting.py`, and save at `dpi=300`. Set `MPLCONFIGDIR=/tmp/adp_matplotlib` before importing pyplot. If no finite observations exist, write `нет наблюдений` in the axes and still save the PNG.

- [ ] **Step 3: Run the integration test**

Run:

```bash
MPLCONFIGDIR=/tmp/adp_matplotlib rtk python -m pytest test_ADP_cli_experiments.py -q
```

Expected: `1 passed` and three non-empty PNG files.

### Task 4: Expose experiment series through the CLI

**Files:**
- Modify: `ADP/cli.py`
- Test: `test_ADP_cli_experiments.py`

- [ ] **Step 1: Add series arguments**

Add an `экспериментальная серия` argument group:

```python
series = parser.add_argument_group("экспериментальная серия")
series.add_argument("--runs", type=int, default=1)
series.add_argument(
    "--output-dir",
    type=Path,
    default=Path("ADP/experiment_outputs"),
)
```

Reject `args.runs < 1` with `parser.error("runs должен быть положительным")`.

- [ ] **Step 2: Replace the one-off fit with Scenario**

After constructing `ADP_Config`, run:

```python
scenario = Scenario(
    n=args.n,
    d=args.d,
    noise=args.noise,
    runs=args.runs,
    seed=args.seed,
    config=config,
    output_dir=args.output_dir,
)
series_dir, failed_runs = scenario.run()
print(f"серия сохранена: {series_dir}")
print(f"успешных запусков: {args.runs - failed_runs}")
print(f"ошибок: {failed_runs}")
if failed_runs:
    raise SystemExit(1)
```

Remove the superseded one-run data generation, result printing, and profile formatting import from `ADP/cli.py`.

- [ ] **Step 3: Verify a real two-run CLI series**

Run:

```bash
MPLCONFIGDIR=/tmp/adp_matplotlib rtk python ADP/cli.py \
  --runs 2 --output-dir /tmp/adp-cli-series \
  --n 40 --d 3 --N_loc 5 --N_lin 8 --N_J 20 --N_phi 3 --h_min 1.0
```

Expected: exit code `0`, two successful runs, a printed series directory, two CSV files, and three PNG files.

- [ ] **Step 4: Run focused regression verification**

Run:

```bash
rtk python -m py_compile ADP/cli.py ADP/single_index/scenario.py
MPLCONFIGDIR=/tmp/adp_matplotlib rtk python -m pytest \
  test_ADP_cli_experiments.py test_ADP_statistic.py -q
```

Expected: compilation succeeds and all focused tests pass.

Implementation commits are intentionally omitted: `ADP/cli.py` already contains overlapping uncommitted user changes, so committing implementation would capture work outside this plan.

