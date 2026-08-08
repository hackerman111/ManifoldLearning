# ADP Single-Index Experiment Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the `main` single-index experiment catalog and CSV-driven plotting surface into modular uppercase ADP Scenario classes selectable from `ADP/cli.py`.

**Architecture:** Each selector lives in its own `scenario_*.py` and exposes `Scenario.grid(profile)`. A registry builds deterministic jobs; a shared runner generates data, adapts requested ADP sizes to each dataset, persists schema-complete CSV rows, and invokes reports that are ported from `main`. The existing manual experiment remains available through `scenario_custom.py` and a compatibility import.

**Tech Stack:** Python dataclasses and stdlib CSV/hashlib, NumPy, pandas, Matplotlib, pytest.

---

### Task 1: Establish the selector, grid, and registry contract

**Files:**
- Create: `ADP/single_index/experiments/__init__.py`
- Create: `ADP/single_index/experiments/types.py`
- Create: `ADP/single_index/experiments/registry.py`
- Create: `ADP/single_index/experiments/scenario_custom.py`
- Create: `ADP/single_index/experiments/scenario_1.py`
- Create: `ADP/single_index/experiments/scenario_2.py`
- Create: `ADP/single_index/experiments/scenario_3.py`
- Create: `ADP/single_index/experiments/scenario_4.py`
- Create: `ADP/single_index/experiments/scenario_5.py`
- Create: `ADP/single_index/experiments/scenario_6.py`
- Create: `ADP/single_index/experiments/scenario_7_1.py`
- Create: `ADP/single_index/experiments/scenario_7_2.py`
- Create: `ADP/single_index/experiments/scenario_8_1.py`
- Create: `ADP/single_index/experiments/scenario_8_2.py`
- Create: `ADP/single_index/experiments/scenario_8_3.py`
- Test: `test_ADP_experiment_catalog.py`

- [ ] **Step 1: Write failing registry and grid tests**

```python
from ADP.single_index.experiments.registry import (
    EXPERIMENT_SELECTORS,
    get_scenario,
    parse_experiment_selectors,
)


def test_registry_matches_main_catalog():
    assert EXPERIMENT_SELECTORS == (
        "1", "2", "3", "4", "5", "6",
        "7.1", "7.2", "8.1", "8.2", "8.3",
    )
    assert parse_experiment_selectors("1,4,8.2") == ("1", "4", "8.2")
    assert parse_experiment_selectors("all") == EXPERIMENT_SELECTORS
    assert [len(get_scenario(item).grid("full")) for item in EXPERIMENT_SELECTORS] == [
        8, 20, 42, 36, 30, 36, 12, 12, 8, 20, 16,
    ]
    assert all(len(get_scenario(item).grid("smoke")) == 1 for item in EXPERIMENT_SELECTORS)
```

- [ ] **Step 2: Run the test and verify import failure**

Run: `rtk python -m pytest test_ADP_experiment_catalog.py -q`

Expected: collection fails because `ADP.single_index.experiments` does not exist.

- [ ] **Step 3: Port validated parameter types**

Create frozen `ExperimentParameters` with fields `d`, `n_over_d`, `sigma_x`,
`rho_corr`, `sigma_eps`, `link`, feature/noise distributions,
`heteroscedastic`, outlier fields, `delta`, and `center_fraction`. Implement
`n = ceil(d * n_over_d)` and the same validation sets as
`main:adp/evaluation/single_index/types.py`.

- [ ] **Step 4: Add one Scenario class per selector**

Each module exports:

```python
class Scenario:
    selector = "3"
    title = "Устойчивость к шуму"
    plots = (
        "quality_vs_sigma_eps.png",
        "success_rate_vs_sigma_eps.png",
        "runtime_vs_sigma_eps.png",
        "outer_iterations_vs_sigma_eps.png",
        "final_objective_vs_sigma_eps.png",
    )

    @staticmethod
    def grid(profile: str) -> tuple[ExperimentParameters, ...]:
        return smoke_grid() if profile == "smoke" else full_grid()
```

Use the literal products from the approved design and one representative smoke
point from `main` in every file. `scenario_custom.py` constructs one
`ExperimentParameters` from CLI `n`, `d`, and `noise` without entering the
built-in registry.

- [ ] **Step 5: Implement canonical registry parsing**

`parse_experiment_selectors` must accept `all`, reject unknown/empty values, remove duplicates, and return selectors in canonical order. `get_scenario(selector)` returns the class from the matching module.

- [ ] **Step 6: Run registry tests**

Run: `rtk python -m pytest test_ADP_experiment_catalog.py -q`

Expected: registry and all full/smoke grid counts pass.

### Task 2: Port deterministic synthetic experiment data

**Files:**
- Create: `ADP/single_index/experiments/data.py`
- Modify: `ADP/single_index/experiments/types.py`
- Test: `test_ADP_experiment_catalog.py`
- Reference: `main:adp/evaluation/single_index/datasets.py`

- [ ] **Step 1: Write failing data tests**

```python
import numpy as np

from ADP.single_index.experiments.data import generate_data, make_seed_bundle
from ADP.single_index.experiments.types import ExperimentParameters


def test_generated_data_is_deterministic_and_gamma_is_orthogonal():
    params = ExperimentParameters(d=5, n_over_d=5, delta=0.25, rho_corr=0.5)
    seeds = make_seed_bundle("8.3", params, 7)
    first = generate_data("8.3", params, seeds)
    second = generate_data("8.3", params, seeds)
    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.Y, second.Y)
    assert np.isclose(np.linalg.norm(first.beta_true), 1.0)
    assert np.isclose(first.gamma @ first.beta_true, 0.0, atol=1e-12)
```

- [ ] **Step 2: Port the synthetic generator**

Implement independent seeds for beta/features/noise/centers/directions/init/
outliers/outlier-noise/gamma/misspecification. Port Gaussian AR(1), uniform and
Student-t features; all link functions; standardized signal; Gaussian and
Student-t noise; heteroscedastic scale; outlier replacement; orthogonal gamma;
and experiment-5 link scaling. Return `GeneratedData(X, Y, beta_true, gamma,
metadata)` without centers or directions because uppercase `fit(X, Y)` owns
those choices.

- [ ] **Step 3: Run deterministic data tests**

Run: `rtk python -m pytest test_ADP_experiment_catalog.py -q`

Expected: deterministic arrays, normalized beta, and orthogonal gamma pass.

### Task 3: Port plotting, schema, and report manifest

**Files:**
- Create: `ADP/single_index/experiments/plotting.py`
- Create: `ADP/single_index/experiments/plots.py`
- Create: `ADP/single_index/experiments/schema.py`
- Create: `ADP/single_index/experiments/reports.py`
- Test: `test_ADP_experiment_reports.py`
- Reference: `main:adp/common/plotting.py`
- Reference: `main:adp/evaluation/single_index/plots.py`
- Reference: `main:adp/evaluation/single_index/reports.py`
- Reference: `main:adp/evaluation/single_index/schema.py`

- [ ] **Step 1: Write a failing no-data and quantile report test**

Create schema-complete temporary CSVs with two experiment-3 runs and outer rows.
Call `write_single_index_reports(tmp_path)`. Assert that
`quality_vs_outer_iteration.png`, `quality_vs_sigma_eps.png`, and the unavailable
`final_objective_vs_sigma_eps.png` all exist and are non-empty.

- [ ] **Step 2: Port common plotting and plots.py**

Copy ADP colors, axis styling, headless `MPLCONFIGDIR`, figure saving,
quantile/Wilson/grouped/box/scatter/heatmap/stacked functions from `main`.
Change only package-relative imports to point at `.plotting`.

- [ ] **Step 3: Port schema.py for uppercase ADP_Config**

Keep public table names and columns from `main`; derive `ADP_CONFIG_COLUMNS`
from `fields(ADP_Config)`. Add `PUBLIC_TABLE_COLUMNS["runs"] =
RUN_SUMMARY_COLUMNS` so report loading targets `runs.csv`.

- [ ] **Step 4: Port reports.py and adapt file names**

Keep the complete `PlotSpec` manifest, Russian subject/axis labels, scales,
facets, grouping and no-data behavior. Change common plotting imports to
`.plotting` and `_PRIMARY_TABLES["runs"]` from `run_summary` to `runs`.

- [ ] **Step 5: Run report tests**

Run: `MPLCONFIGDIR=/tmp/adp_matplotlib rtk python -m pytest test_ADP_experiment_reports.py -q`

Expected: available plots contain aggregated data and unavailable metrics still produce explicit no-observation PNGs.

### Task 4: Implement the shared series runner and CSV persistence

**Files:**
- Create: `ADP/single_index/experiments/base.py`
- Modify: `ADP/single_index/scenario.py`
- Modify: `test_ADP_cli_experiments.py`
- Test: `test_ADP_experiment_catalog.py`

- [ ] **Step 1: Write failing multi-selector series test**

Run selectors `1` and `3` with smoke profile and one seed against `tmp_path`.
Assert two run rows, canonical selectors, schema-complete outer rows,
`series.csv`, `artifacts.csv`, all five table CSVs and selector-specific plots.

- [ ] **Step 2: Build deterministic jobs**

Expand selectors in canonical order, each scenario grid in literal order, and
seeds `base_seed + offset`. Derive each seed bundle from selector, canonical
parameter mapping and seed so selection order cannot change generated data.

- [ ] **Step 3: Resolve requested and effective ADP config**

For every job constrain `N_loc <= n`, `N_lin <= n`, and valid `N_J`. When
`n <= d + 1`, select random initialization and set a valid unused `N_lin`.
Write requested and effective values to runs CSV; do not silently mutate the
user's base config.

- [ ] **Step 4: Persist each completed job**

Create all CSV headers before fitting. After each job, append one runs row and
its available outer rows, then flush by closing the files. Classify successful
fits separately from numerical failures. Finalize series counts and invoke
`write_single_index_reports` exclusively from public CSVs.

- [ ] **Step 5: Add custom compatibility**

Move current custom behavior to `experiments/scenario_custom.py` and make
`ADP/single_index/scenario.py` export its `Scenario` so existing imports remain
valid.

- [ ] **Step 6: Run catalog and legacy integration tests**

Run: `MPLCONFIGDIR=/tmp/adp_matplotlib rtk python -m pytest test_ADP_experiment_catalog.py test_ADP_cli_experiments.py -q`

Expected: custom and built-in series both persist and report successfully.

### Task 5: Add experiment selection to the CLI

**Files:**
- Modify: `ADP/cli.py`
- Modify: `test_ADP_cli_experiments.py`

- [ ] **Step 1: Add CLI contract tests**

Verify `--list-experiments`, `--experiment 3`, `--experiment 1,4,8.2`,
`--experiment all`, invalid selectors, and `custom` argument compatibility.

- [ ] **Step 2: Add selection arguments and dispatch**

Add `--experiment` defaulting to `custom`, `--profile smoke|full`, and
`--list-experiments`. Parse built-in selectors through `registry.py`; dispatch
custom to `scenario_custom.Scenario` and built-ins to `run_experiments`.

- [ ] **Step 3: Run CLI tests and help inspection**

Run: `MPLCONFIGDIR=/tmp/adp_matplotlib rtk python -m pytest test_ADP_cli_experiments.py -q`

Run: `rtk python ADP/cli.py --help`

Expected: selector/profile controls are visible and all CLI tests pass.

### Task 6: Verify the catalog end to end

**Files:**
- Verify: all files above

- [ ] **Step 1: Compile all uppercase experiment modules**

Run: `rtk python -m compileall -q ADP/single_index/experiments ADP/cli.py`

Expected: exit code 0.

- [ ] **Step 2: Run focused and full tests**

Run: `MPLCONFIGDIR=/tmp/adp_matplotlib rtk python -m pytest test_ADP_experiment_catalog.py test_ADP_experiment_reports.py test_ADP_cli_experiments.py test_ADP_statistic.py -q`

Run: `MPLCONFIGDIR=/tmp/adp_matplotlib rtk python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 3: Run a real all-selector smoke series**

Run: `MPLCONFIGDIR=/tmp/adp_matplotlib rtk python ADP/cli.py --experiment all --profile smoke --runs 1 --output-dir /tmp/adp-catalog-smoke`

Expected: eleven completed jobs, public CSVs, experiment plot directories,
summary plots, and no fabricated values in unsupported metric columns.

- [ ] **Step 4: Inspect persisted artifacts**

Check run statuses separately from `cosine_abs`, inspect requested/effective
config columns, verify selectors and seeds, and visually inspect at least one
quality, bandwidth, heatmap and no-data plot.

Implementation commits are omitted because the current CLI and Scenario files
contain overlapping uncommitted work from the approved preceding feature.

