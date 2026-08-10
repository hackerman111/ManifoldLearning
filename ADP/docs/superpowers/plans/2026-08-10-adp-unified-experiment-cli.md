# Unified ADP Experiment CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one CLI and experiment runner for single-index and multi-index ADP, paired one/two-variant runs, durable archives/resume, and applicable main-derived reports.

**Architecture:** Keep public experiment contracts and Python-file loading in `ADP/experiment.py`, execution plus durable storage/CSV export in `ADP/experiment_runner.py`, and plotting in `ADP/experiment_reports.py`. Both fitted models expose one optional outer-iteration callback; the runner remains the only component that knows about `tqdm`, RSS sampling, archives, or reports.

**Tech Stack:** Python 3.14, dataclasses, argparse, NumPy, SciPy, pandas, Matplotlib, tqdm, psutil, threadpoolctl, pytest.

---

## Working-tree constraints

The live checkout already contains user-owned edits in:

- `ADP/single_index/ADP_single_index.py`: single-index default changed to LSMR with `tol=1e-8`;
- `ADP/single_index/solvers/VarPro.py`: line-search, stall detection, and flattened-matrix optimizations.

Preserve both. Do not restore or rewrite `VarPro.py`, and do not absorb either pre-existing diff into feature commits. When adding the progress callback to `ADP_single_index.py`, edit around the current LSMR constructor and keep it as the working-tree `auto` default. Stage only the new callback hunks from that overlapping file. Before every commit, inspect `git diff --cached --name-only`; never stage `VarPro.py` as part of this feature.

## File map

- Create `ADP/experiment.py`: public immutable experiment types, validation, Python-file loader, built-in data generation, and data validation.
- Create `ADP/experiment_runner.py`: job expansion, solver/model construction, progress, telemetry, atomic NPZ/JSON commits, resume, CSV exports, and paired summaries.
- Create `ADP/experiment_reports.py`: compact main-derived plot manifest, aggregation/rendering, and `artifacts.csv`.
- Modify `ADP/ADP_Data.py`: permit missing ground truth.
- Modify `ADP/__init__.py`: export experiment types and runner entry point.
- Modify `ADP/single_index/ADP_single_index.py`: optional outer-iteration callback; preserve the current LSMR default.
- Modify `ADP/multi_index/ADP_multi_index.py`: the same callback contract.
- Rewrite `ADP/cli.py`: unified manual/file/resume/reports-only interface.
- Create `ADP/examples/experiment_compare.py`: runnable two-point LSMR/VarPro example.
- Create `ADP/test_experiment.py`: focused unit and integration coverage.

### Task 1: Public experiment contracts, loader, and data generation

**Files:**
- Create: `ADP/experiment.py`
- Modify: `ADP/ADP_Data.py:6-10`
- Modify: `ADP/__init__.py:5-22`
- Create: `ADP/test_experiment.py`

- [ ] **Step 1: Write failing contract and loader tests**

Create `ADP/test_experiment.py` with these first tests:

```python
from pathlib import Path

import numpy as np
import pytest

from ADP import ADP_Config, ADP_Data


def test_python_file_exports_valid_experiment(tmp_path: Path):
    from ADP.experiment import load_experiment

    source = tmp_path / "experiment_file.py"
    source.write_text(
        """
from ADP import ADP_Config
from ADP.experiment import ADP_Experiment, ADP_ExperimentPoint, ADP_ExperimentVariant

experiment = ADP_Experiment(
    name="pair",
    mode="single",
    runs=2,
    seed=11,
    points=(ADP_ExperimentPoint("noise", 24, 3, 0.1, {"sigma_eps": 0.1}),),
    variants={"base": ADP_ExperimentVariant(ADP_Config(), "auto")},
)
""",
        encoding="utf-8",
    )

    experiment = load_experiment(source)

    assert experiment.name == "pair"
    assert experiment.points[0].metadata == {"sigma_eps": 0.1}
    assert tuple(experiment.variants) == ("base",)


def test_multi_varpro_is_rejected():
    from ADP.experiment import (
        ADP_Experiment,
        ADP_ExperimentPoint,
        ADP_ExperimentVariant,
        validate_experiment,
    )

    experiment = ADP_Experiment(
        name="bad",
        mode="multi",
        index_dim=2,
        points=(ADP_ExperimentPoint("p", 24, 4),),
        variants={"v": ADP_ExperimentVariant(ADP_Config(), "varpro")},
    )

    with pytest.raises(ValueError, match="varpro.*multi"):
        validate_experiment(experiment)


def test_default_multi_data_and_optional_truth():
    from ADP.experiment import (
        ADP_Experiment,
        ADP_ExperimentPoint,
        ADP_ExperimentVariant,
        make_data,
    )

    point = ADP_ExperimentPoint("p", 30, 5, 0.0)
    experiment = ADP_Experiment(
        name="multi",
        mode="multi",
        index_dim=2,
        points=(point,),
        variants={"v": ADP_ExperimentVariant(ADP_Config(), "lsmr")},
    )
    first = make_data(experiment, point, 13)
    second = make_data(experiment, point, 13)

    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.Y, second.Y)
    assert first.true_index.shape == (5, 2)
    np.testing.assert_allclose(first.true_index.T @ first.true_index, np.eye(2))
    assert ADP_Data(first.X, first.Y, None).true_index is None
```

- [ ] **Step 2: Run the tests and confirm the missing-module failure**

Run from the repository root:

```bash
rtk proxy python -m pytest ADP/test_experiment.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'ADP.experiment'`.

- [ ] **Step 3: Make `ADP_Data.true_index` optional**

Change only its annotation:

```python
@dataclass(frozen=True, slots=True)
class ADP_Data:
    X: np.ndarray
    Y: np.ndarray
    true_index: np.ndarray | None
```

- [ ] **Step 4: Implement the public experiment types and validation**

Add `ADP/experiment.py` with this public boundary:

```python
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .ADP_Config import ADP_Config
from .ADP_Data import ADP_Data
from .engine.utils import _prepare_xy
from .single_index.solvers.LSMR import solve as solve_lsmr
from .single_index.solvers.VarPro import solve as solve_varpro

MetadataValue = str | int | float | bool | None
SolverName = Literal["auto", "lsmr", "varpro"]
DataFactory = Callable[["ADP_ExperimentPoint", np.random.Generator], ADP_Data]


@dataclass(frozen=True, slots=True)
class ADP_ExperimentPoint:
    name: str
    n: int
    d: int
    noise: float = 0.05
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ADP_ExperimentVariant:
    config: ADP_Config
    solver: SolverName = "auto"
    solver_settings: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ADP_Experiment:
    name: str
    mode: Literal["single", "multi"]
    points: tuple[ADP_ExperimentPoint, ...]
    variants: Mapping[str, ADP_ExperimentVariant]
    runs: int = 1
    seed: int = 7
    index_dim: int = 1
    data_factory: DataFactory | None = None


def _safe_name(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
    ):
        raise ValueError(f"{field_name} must be a non-empty path-safe string")
    return value


def _solver_method(mode: str, name: str):
    if name == "varpro":
        if mode == "multi":
            raise ValueError("varpro is not available in multi-index mode")
        return solve_varpro
    if name in {"auto", "lsmr"}:
        return solve_lsmr
    raise ValueError("solver must be 'auto', 'lsmr', or 'varpro'")


def _validate_solver_settings(mode: str, variant: ADP_ExperimentVariant) -> None:
    if not isinstance(variant.solver_settings, Mapping):
        raise TypeError("solver_settings must be a mapping")
    method = _solver_method(mode, variant.solver)
    forbidden = {"statistics", "beta", "lambda_penalty", "local_ridge"}
    unknown = set(variant.solver_settings) - set(inspect.signature(method).parameters)
    overlap = set(variant.solver_settings) & forbidden
    if unknown:
        raise ValueError(f"unknown solver settings: {', '.join(sorted(unknown))}")
    if overlap:
        raise ValueError(f"problem parameters cannot be solver settings: {', '.join(sorted(overlap))}")
    try:
        json.dumps(dict(variant.solver_settings), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("solver_settings must be finite JSON values") from error


def validate_experiment(experiment: ADP_Experiment) -> ADP_Experiment:
    if not isinstance(experiment, ADP_Experiment):
        raise TypeError("experiment file must export ADP_Experiment as 'experiment'")
    _safe_name(experiment.name, "experiment.name")
    if experiment.mode not in {"single", "multi"}:
        raise ValueError("mode must be 'single' or 'multi'")
    if isinstance(experiment.runs, bool) or not isinstance(experiment.runs, int) or experiment.runs < 1:
        raise ValueError("runs must be a positive integer")
    if isinstance(experiment.seed, bool) or not isinstance(experiment.seed, int) or experiment.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not experiment.points:
        raise ValueError("points must not be empty")
    if isinstance(experiment.index_dim, bool) or not isinstance(experiment.index_dim, int):
        raise TypeError("index_dim must be an integer")
    if not isinstance(experiment.variants, Mapping):
        raise TypeError("variants must be a mapping")
    if not 1 <= len(experiment.variants) <= 2:
        raise ValueError("variants must contain one or two entries")

    point_names = []
    for point in experiment.points:
        if not isinstance(point, ADP_ExperimentPoint):
            raise TypeError("points must contain ADP_ExperimentPoint values")
        point_names.append(_safe_name(point.name, "point.name"))
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (point.n, point.d)):
            raise TypeError(f"point {point.name}: n and d must be integers")
        if point.d < 1 or point.n <= point.d + 1:
            raise ValueError(f"point {point.name}: require d >= 1 and n > d + 1")
        if isinstance(point.noise, bool) or not isinstance(point.noise, (int, float)):
            raise TypeError(f"point {point.name}: noise must be numeric")
        if not np.isfinite(point.noise) or point.noise < 0:
            raise ValueError(f"point {point.name}: noise must be finite and nonnegative")
        if not isinstance(point.metadata, Mapping):
            raise TypeError(f"point {point.name}: metadata must be a mapping")
        for key, value in point.metadata.items():
            _safe_name(key, "metadata key")
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError(f"metadata {key} must be a JSON scalar")
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(f"metadata {key} must be finite")
        if experiment.mode == "multi" and not 1 <= experiment.index_dim < point.d:
            raise ValueError(f"point {point.name}: require 1 <= index_dim < d")
    if len(point_names) != len(set(point_names)):
        raise ValueError("point names must be unique")
    if experiment.mode == "single" and experiment.index_dim != 1:
        raise ValueError("single-index experiments require index_dim=1")

    variant_names = []
    for name, variant in experiment.variants.items():
        variant_names.append(_safe_name(name, "variant name"))
        if not isinstance(variant, ADP_ExperimentVariant):
            raise TypeError("variant values must be ADP_ExperimentVariant")
        if not isinstance(variant.config, ADP_Config):
            raise TypeError("variant.config must be ADP_Config")
        _validate_solver_settings(experiment.mode, variant)
    if len(variant_names) != len(set(variant_names)):
        raise ValueError("variant names must be unique")
    if experiment.data_factory is not None and not callable(experiment.data_factory):
        raise TypeError("data_factory must be callable")
    return experiment
```

Add the loader and data functions to the same module:

```python
def load_experiment(path: str | Path) -> ADP_Experiment:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location("_adp_experiment_file", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load experiment file: {source}")
    module = importlib.util.module_from_spec(spec)
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(source.parent))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    if not hasattr(module, "experiment"):
        raise ValueError("experiment file must export 'experiment'")
    return validate_experiment(module.experiment)


def _orient_basis(basis: np.ndarray) -> np.ndarray:
    columns = np.arange(basis.shape[1])
    signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), columns])
    basis *= np.where(signs == 0, 1.0, signs)
    return basis


def make_data(
    experiment: ADP_Experiment,
    point: ADP_ExperimentPoint,
    seed: int,
) -> ADP_Data:
    rng = np.random.default_rng(seed)
    if experiment.data_factory is not None:
        data = experiment.data_factory(point, rng)
    else:
        X = rng.normal(size=(point.n, point.d))
        if experiment.mode == "single":
            true_index = rng.normal(size=point.d)
            true_index /= np.linalg.norm(true_index)
            signal = np.sin(X @ true_index)
        else:
            true_index, _ = np.linalg.qr(
                rng.normal(size=(point.d, experiment.index_dim)),
                mode="reduced",
            )
            true_index = _orient_basis(true_index)
            signal = np.sin(X @ true_index).sum(axis=1)
        Y = signal + point.noise * rng.normal(size=point.n)
        data = ADP_Data(X, Y, true_index)
    return validate_data(data, experiment, point)


def validate_data(
    data: ADP_Data,
    experiment: ADP_Experiment,
    point: ADP_ExperimentPoint,
) -> ADP_Data:
    if not isinstance(data, ADP_Data):
        raise TypeError("data_factory must return ADP_Data")
    X, Y = _prepare_xy(data.X, data.Y)
    if X.shape != (point.n, point.d):
        raise ValueError(f"point {point.name}: data must have shape {(point.n, point.d)}")
    if data.true_index is None:
        return ADP_Data(X, Y, None)
    truth = np.asarray(data.true_index, dtype=float)
    expected = (point.d,) if experiment.mode == "single" else (point.d, experiment.index_dim)
    if truth.shape != expected or not np.all(np.isfinite(truth)):
        raise ValueError(f"true_index must have shape {expected} and finite values")
    if experiment.mode == "single" and np.linalg.norm(truth) == 0:
        raise ValueError("true_index must be nonzero")
    if experiment.mode == "multi" and np.linalg.matrix_rank(truth) != experiment.index_dim:
        raise ValueError("true_index must have full column rank")
    return ADP_Data(X, Y, truth)
```

- [ ] **Step 5: Export the public types**

Add these imports and `__all__` entries in `ADP/__init__.py`:

```python
from .experiment import (
    ADP_Experiment,
    ADP_ExperimentPoint,
    ADP_ExperimentVariant,
    load_experiment,
)
```

Export all four names without changing existing exports.

- [ ] **Step 6: Run the focused tests**

```bash
rtk proxy python -m pytest ADP/test_experiment.py -q
```

Expected: `3 passed`.

- [ ] **Step 7: Commit the contract slice**

```bash
rtk git add ADP/ADP_Data.py ADP/__init__.py ADP/experiment.py ADP/test_experiment.py
rtk git commit -m "feat: define ADP experiment contracts"
```

### Task 2: Optional model progress and current-default solver construction

**Files:**
- Modify: `ADP/single_index/ADP_single_index.py:64-71,180-193`
- Modify: `ADP/multi_index/ADP_multi_index.py:79-86,197-210`
- Modify: `ADP/experiment.py`
- Modify: `ADP/test_experiment.py`

- [ ] **Step 1: Add failing progress and solver tests**

Append:

```python
from ADP import ADP_SolverResult, ADP_solver


def _unchanged_single(statistics, initial_index, **params):
    return ADP_SolverResult(
        index=np.asarray(initial_index),
        coefficients=np.ones(len(statistics.I)),
        diagnostics={"inner_iterations": 1, "beta_delta": 0.0},
    )


def _unchanged_multi(statistics, initial_index, **params):
    width = np.asarray(initial_index).shape[1]
    return ADP_SolverResult(
        index=np.asarray(initial_index),
        coefficients=np.ones((len(statistics.I), width)),
        diagnostics={
            "inner_iterations": 1,
            "beta_delta": 0.0,
            "eigenvalues": np.ones(width),
        },
    )


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_fit_reports_each_outer_iteration(mode):
    from ADP import ADP_Config, ADP_multi_index, ADP_single_index

    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 4))
    Y = np.sin(X[:, 0])
    config = ADP_Config(
        seed=5,
        N_loc=6,
        N_lin=8,
        N_J=4,
        N_phi=3,
        h_min=1e6,
        index_init="random",
    )
    events = []
    if mode == "single":
        model = ADP_single_index(config, ADP_solver(_unchanged_single))
    else:
        model = ADP_multi_index(2, config, ADP_solver(_unchanged_multi))

    model.fit(X, Y, progress=events.append)

    assert len(events) == len(model.result_.trace) == 1
    assert events[0]["k"] == 0
    assert (("rho" in events[0]) if mode == "single" else ("alpha" in events[0]))
    assert model.coefficients_ is not None


def test_auto_solver_uses_current_single_model_default():
    from ADP.experiment import ADP_Experiment, ADP_ExperimentPoint, ADP_ExperimentVariant
    from ADP.experiment_runner import _build_model
    from ADP.single_index.solvers.LSMR import solve as solve_lsmr

    experiment = ADP_Experiment(
        name="auto",
        mode="single",
        points=(ADP_ExperimentPoint("p", 24, 3),),
        variants={
            "v": ADP_ExperimentVariant(
                ADP_Config(), "auto", {"max_steps": 7}
            )
        },
    )

    model = _build_model(experiment, experiment.variants["v"], seed=9)

    assert model.solver.method is solve_lsmr
    assert model.solver.settings == {"tol": 1e-8, "max_steps": 7}
    assert model.config.seed == 9


def test_auto_solver_uses_current_multi_model_default():
    from ADP.experiment import ADP_Experiment, ADP_ExperimentPoint, ADP_ExperimentVariant
    from ADP.experiment_runner import _build_model
    from ADP.single_index.solvers.LSMR import solve as solve_lsmr

    experiment = ADP_Experiment(
        name="auto_multi",
        mode="multi",
        index_dim=2,
        points=(ADP_ExperimentPoint("p", 24, 4),),
        variants={"v": ADP_ExperimentVariant(ADP_Config(), "auto", {"max_steps": 7})},
    )

    model = _build_model(experiment, experiment.variants["v"], seed=9)

    assert model.solver.method is solve_lsmr
    assert model.solver.settings == {"tol": 1e-7 / 2, "max_steps": 7}
    assert model.config.seed == 9
```

The last test initially fails because `experiment_runner.py` does not exist; create the minimal module in Step 4.

- [ ] **Step 2: Run the progress tests and verify failure**

```bash
rtk proxy python -m pytest ADP/test_experiment.py -q
```

Expected: failures for the unexpected `progress` keyword and missing `ADP.experiment_runner`.

- [ ] **Step 3: Add the callback without changing solver defaults**

Apply the same narrow change to both models:

```python
def fit(self, X, Y, *, progress=None):
    tracker = start_tracking()
    try:
        return self._fit(X, Y, tracker, progress)
    finally:
        self.profile_ = finish_tracking(tracker)


def _fit(self, X, Y, tracker, progress):
    # existing body stays unchanged
```

Immediately after each existing `result.trace.append(...)`, add:

```python
if progress is not None:
    progress(dict(result.trace[-1]))
```

At the successful end of the single-index fit, retain the already-computed final coefficients just as the multi-index model does:

```python
self.coefficients_ = solver_result.coefficients
```

Do not import `tqdm` in either model. In the single-index file preserve these live lines exactly in behavior:

```python
self.solver = solver or ADP_solver(solve_lsmr, tol=1e-8)
```

Do not touch `ADP/single_index/solvers/VarPro.py`.

- [ ] **Step 4: Add minimal model construction**

Create `ADP/experiment_runner.py` with `_build_model`:

```python
from dataclasses import replace

from .ADP_Solver import ADP_solver
from .experiment import ADP_Experiment, ADP_ExperimentVariant, _solver_method
from .multi_index.ADP_multi_index import ADP_multi_index
from .single_index.ADP_single_index import ADP_single_index


def _build_model(
    experiment: ADP_Experiment,
    variant: ADP_ExperimentVariant,
    seed: int,
):
    config = replace(variant.config, seed=seed)
    model = (
        ADP_single_index(config)
        if experiment.mode == "single"
        else ADP_multi_index(experiment.index_dim, config)
    )
    if variant.solver == "auto":
        if variant.solver_settings:
            settings = {**model.solver.settings, **dict(variant.solver_settings)}
            model.solver = ADP_solver(model.solver.method, **settings)
        return model
    model.solver = ADP_solver(
        _solver_method(experiment.mode, variant.solver),
        **dict(variant.solver_settings),
    )
    return model
```

In the runner, use one tiny identity helper for persisted effective solver names:

```python
def _solver_name(method) -> str:
    if method is _solver_method("single", "lsmr"):
        return "lsmr"
    if method is _solver_method("single", "varpro"):
        return "varpro"
    return f"{method.__module__}:{method.__qualname__}"
```

- [ ] **Step 5: Run tests and current multi-index regression**

```bash
rtk proxy python -m pytest ADP/test_experiment.py test_ADP_multi_index.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit only the progress/model slice**

Review the overlap first:

```bash
rtk proxy git diff -- ADP/single_index/ADP_single_index.py
rtk git status --short
```

Stage the non-overlapping files normally. Stage `ADP_single_index.py` interactively, accepting only the new `fit/_fit` callback hunks and rejecting the pre-existing import/default-solver hunks:

```bash
rtk git add ADP/multi_index/ADP_multi_index.py ADP/experiment_runner.py ADP/test_experiment.py
rtk git add -p ADP/single_index/ADP_single_index.py
rtk git diff --cached --name-only
rtk proxy git diff --cached -- ADP/single_index/ADP_single_index.py
rtk git commit -m "feat: expose ADP outer iteration progress"
```

The cached single-index diff must contain only the progress callback and final `coefficients_` assignment. If it contains the LSMR import/default change, abort the commit and correct the staged selection; the working-tree LSMR change itself remains untouched.

### Task 3: Deterministic job plan and atomic series store

**Files:**
- Modify: `ADP/experiment_runner.py`
- Modify: `ADP/test_experiment.py`

- [ ] **Step 1: Add failing ordering and atomic-data tests**

Append:

```python
def _paired_experiment(data_factory=None):
    from ADP.experiment import ADP_Experiment, ADP_ExperimentPoint, ADP_ExperimentVariant

    return ADP_Experiment(
        name="paired",
        mode="single",
        runs=2,
        seed=20,
        points=(
            ADP_ExperimentPoint("p1", 24, 3, 0.0, {"sigma_eps": 0.0}),
            ADP_ExperimentPoint("p2", 24, 3, 0.1, {"sigma_eps": 0.1}),
        ),
        variants={
            "A": ADP_ExperimentVariant(ADP_Config(N_J=4, N_phi=3, h_min=1e6)),
            "B": ADP_ExperimentVariant(ADP_Config(N_J=4, N_phi=3, h_min=1e6)),
        },
        data_factory=data_factory,
    )


def test_jobs_share_seed_sequence_and_alternate_variants():
    from ADP.experiment_runner import _build_jobs

    jobs = _build_jobs(_paired_experiment())

    assert [(j.point.name, j.seed, j.variant_name) for j in jobs] == [
        ("p1", 20, "A"), ("p1", 20, "B"),
        ("p1", 21, "B"), ("p1", 21, "A"),
        ("p2", 20, "A"), ("p2", 20, "B"),
        ("p2", 21, "B"), ("p2", 21, "A"),
    ]


def test_series_store_round_trips_truthless_data(tmp_path: Path):
    from ADP.experiment_runner import _SeriesStore

    experiment = _paired_experiment()
    store = _SeriesStore.create(tmp_path, experiment)
    point = experiment.points[0]
    data = ADP_Data(np.ones((24, 3)), np.zeros(24), None)

    store.save_data(point, 20, data)
    restored = store.load_data(point, 20)

    np.testing.assert_array_equal(restored.X, data.X)
    np.testing.assert_array_equal(restored.Y, data.Y)
    assert restored.true_index is None
```

- [ ] **Step 2: Run and verify missing symbols**

```bash
rtk proxy python -m pytest ADP/test_experiment.py -q
```

Expected: import failures for `_build_jobs` and `_SeriesStore`.

- [ ] **Step 3: Implement ordered jobs**

Add:

```python
from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path

import numpy as np

from .ADP_Config import ADP_Config
from .ADP_Data import ADP_Data
from .experiment import ADP_Experiment, ADP_ExperimentPoint, ADP_ExperimentVariant


@dataclass(frozen=True, slots=True)
class _Job:
    point: ADP_ExperimentPoint
    run_index: int
    seed: int
    variant_index: int
    variant_name: str
    variant: ADP_ExperimentVariant

    @property
    def run_id(self) -> str:
        return f"{self.point.name}__seed-{self.seed}__{self.variant_name}"


def _build_jobs(experiment: ADP_Experiment) -> tuple[_Job, ...]:
    variants = tuple(experiment.variants.items())
    index_by_name = {name: index for index, (name, _) in enumerate(variants)}
    jobs = []
    for point in experiment.points:
        for run_index in range(experiment.runs):
            ordered = variants if run_index % 2 == 0 else tuple(reversed(variants))
            for name, variant in ordered:
                jobs.append(
                    _Job(
                        point=point,
                        run_index=run_index,
                        seed=experiment.seed + run_index,
                        variant_index=index_by_name[name],
                        variant_name=name,
                        variant=variant,
                    )
                )
    return tuple(jobs)
```

- [ ] **Step 4: Implement atomic JSON/NPZ and store lifecycle**

Add these helpers and store methods:

```python
def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


class _SeriesStore:
    def __init__(self, series_dir: Path):
        self.series_dir = series_dir
        self.data_dir = series_dir / "data"
        self.commit_dir = series_dir / "commits"
        self.model_dir = series_dir / "models"

    @classmethod
    def create(cls, output_dir: Path, experiment: ADP_Experiment) -> "_SeriesStore":
        parent = Path(output_dir) / experiment.name
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        series_dir = parent / stamp
        suffix = 1
        while series_dir.exists():
            series_dir = parent / f"{stamp}-{suffix}"
            suffix += 1
        series_dir.mkdir()
        store = cls(series_dir)
        store.data_dir.mkdir()
        store.commit_dir.mkdir()
        store.model_dir.mkdir()
        return store

    @classmethod
    def resume(cls, series_dir: Path) -> "_SeriesStore":
        path = Path(series_dir)
        if not path.is_dir() or not (path / "commits").is_dir():
            raise ValueError(f"not an ADP series directory: {path}")
        return cls(path)

    def data_path(self, point: ADP_ExperimentPoint, seed: int) -> Path:
        return self.data_dir / point.name / f"seed_{seed}.npz"

    def save_data(self, point: ADP_ExperimentPoint, seed: int, data: ADP_Data) -> Path:
        values = {
            "X": data.X,
            "Y": data.Y,
            "has_true_index": np.asarray(data.true_index is not None),
        }
        if data.true_index is not None:
            values["true_index"] = data.true_index
        path = self.data_path(point, seed)
        _atomic_npz(path, **values)
        return path

    def load_data(self, point: ADP_ExperimentPoint, seed: int) -> ADP_Data:
        with np.load(self.data_path(point, seed), allow_pickle=False) as archive:
            truth = archive["true_index"] if bool(archive["has_true_index"]) else None
            return ADP_Data(archive["X"], archive["Y"], truth)

    def commit_path(self, job: _Job) -> Path:
        return self.commit_dir / f"{job.run_id}.json"

    def read_commits(self) -> list[dict]:
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.commit_dir.glob("*.json"))]
```

Add canonical job specs and the two commit methods. This is the only resume comparison path:

```python
def _qualified_name(value) -> str:
    return f"{value.__module__}:{value.__qualname__}"


def _spec_value(value):
    if callable(value):
        return _qualified_name(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _spec_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_spec_value(item) for item in value]
    return value


def _job_spec(job: _Job, experiment: ADP_Experiment) -> dict:
    requested_config = {
        item.name: _spec_value(getattr(job.variant.config, item.name))
        for item in fields(ADP_Config)
    }
    effective_config = {**requested_config, "seed": job.seed}
    return {
        "schema_version": 1,
        "experiment": experiment.name,
        "mode": experiment.mode,
        "index_dim": experiment.index_dim,
        "point": {
            "name": job.point.name,
            "n": job.point.n,
            "d": job.point.d,
            "noise": job.point.noise,
            "metadata": _spec_value(dict(job.point.metadata)),
        },
        "run_index": job.run_index,
        "seed": job.seed,
        "variant_index": job.variant_index,
        "variant": job.variant_name,
        "requested_config": requested_config,
        "effective_config": effective_config,
        "solver": job.variant.solver,
        "solver_settings": _spec_value(dict(job.variant.solver_settings)),
    }
```

Add to `_SeriesStore`:

```python
    def is_complete(self, job: _Job, experiment: ADP_Experiment) -> bool:
        path = self.commit_path(job)
        if not path.exists():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("spec") != _job_spec(job, experiment):
            raise ValueError(f"resume specification differs for {job.run_id}")
        return True

    def commit(self, job: _Job, experiment: ADP_Experiment, outcome: dict) -> None:
        path = self.commit_path(job)
        payload = {"spec": _job_spec(job, experiment), **outcome}
        if path.exists():
            current = json.loads(path.read_text(encoding="utf-8"))
            if current.get("spec") != payload["spec"]:
                raise ValueError(f"resume specification differs for {job.run_id}")
            return
        _atomic_json(path, payload)
```

Before writing a commit, ensure the finished outcome contains only finite JSON scalars, lists, dictionaries, and strings. Matching commits are the only jobs considered complete on resume.

- [ ] **Step 5: Run the store tests**

```bash
rtk proxy python -m pytest ADP/test_experiment.py -q
```

Expected: all current tests pass.

- [ ] **Step 6: Commit**

```bash
rtk git add ADP/experiment_runner.py ADP/test_experiment.py
rtk git commit -m "feat: add durable ADP experiment store"
```

### Task 4: Sequential execution, telemetry, snapshots, and resume

**Files:**
- Modify: `ADP/experiment_runner.py`
- Modify: `ADP/__init__.py`
- Modify: `ADP/test_experiment.py`

- [ ] **Step 1: Add a failing paired execution/resume test**

Append a fake model whose `fit` records exact arrays and seeds, then invoke the public runner twice:

```python
def test_runner_pairs_inputs_saves_failures_and_resumes(tmp_path: Path, monkeypatch):
    import ADP.experiment_runner as runner

    calls = []
    factory_calls = []

    def data_factory(point, rng):
        factory_calls.append(point.name)
        X = rng.normal(size=(point.n, point.d))
        truth = np.eye(point.d)[0]
        return ADP_Data(X, X @ truth, truth)

    class FakeResult:
        def __init__(self, beta):
            self.beta_init = beta.copy()
            self.beta_final = beta.copy()
            self.trace = [{"k": 0, "h": 1.0, "rho": 1.0, "mean_mass": 5.0, "beta": beta.copy()}]
            self.stop_reason = "h_min"

    class FakeModel:
        def __init__(self, seed, variant):
            self.config = ADP_Config(seed=seed)
            self.variant = variant
            self.solver = ADP_solver(_unchanged_single)

        def fit(self, X, Y, *, progress=None):
            calls.append((self.variant, self.config.seed, X.copy(), Y.copy()))
            if progress is not None:
                progress({"k": 0, "h": 1.0, "rho": 1.0})
            beta = np.eye(X.shape[1])[0]
            self.result_ = FakeResult(beta)
            self.beta_ = beta
            self.coefficients_ = np.ones(len(X))
            self.effective_parameters_ = {"N_J": 4, "N_phi": 3, "N_loc": 5, "N_lin": 6, "h_min": 1.0}
            self.profile_ = {"total_time_seconds": 0.01, "peak_memory_bytes": 32, "stages": {}}
            return self

    experiment = _paired_experiment(data_factory)
    monkeypatch.setattr(
        runner,
        "_build_model",
        lambda experiment, variant, seed: FakeModel(
            seed,
            next(name for name, value in experiment.variants.items() if value is variant),
        ),
    )

    series_dir, failures = runner.run_experiment(
        experiment,
        tmp_path,
        save_models=False,
        show_progress=False,
    )
    first_call_count = len(calls)
    resumed_dir, resumed_failures = runner.run_experiment(
        experiment,
        tmp_path,
        resume=series_dir,
        save_models=False,
        show_progress=False,
    )

    assert failures == resumed_failures == 0
    assert resumed_dir == series_dir
    assert len(factory_calls) == 4
    assert len(calls) == first_call_count == 8
    for offset in range(0, len(calls), 2):
        assert calls[offset][1] == calls[offset + 1][1]
        np.testing.assert_array_equal(calls[offset][2], calls[offset + 1][2])
        np.testing.assert_array_equal(calls[offset][3], calls[offset + 1][3])
    assert not list((series_dir / "models").glob("*.npz"))
```

Add one continuation check using the same fake result shape:

```python
def test_runner_commits_fit_failures_and_continues(tmp_path: Path, monkeypatch):
    import ADP.experiment_runner as runner

    calls = []

    class SometimesFails:
        def __init__(self, seed, variant):
            self.config = ADP_Config(seed=seed)
            self.variant = variant
            self.solver = ADP_solver(_unchanged_single)

        def fit(self, X, Y, *, progress=None):
            calls.append((self.variant, self.config.seed))
            if self.variant == "A":
                raise FloatingPointError("synthetic failure")
            beta = np.eye(X.shape[1])[0]
            self.result_ = type(
                "Result",
                (),
                {
                    "trace": [{"k": 0, "h": 1.0, "rho": 1.0, "beta": beta}],
                    "stop_reason": "h_min",
                },
            )()
            self.beta_ = beta
            self.coefficients_ = np.ones(len(X))
            self.effective_parameters_ = {}
            self.profile_ = {"total_time_seconds": 0.01, "peak_memory_bytes": 32, "stages": {}}
            return self

    experiment = _paired_experiment()
    monkeypatch.setattr(
        runner,
        "_build_model",
        lambda experiment, variant, seed: SometimesFails(
            seed,
            next(name for name, value in experiment.variants.items() if value is variant),
        ),
    )

    series_dir, failures = runner.run_experiment(
        experiment, tmp_path, save_models=False, show_progress=False
    )

    assert len(calls) == 8
    assert failures == 4
    assert len(list((series_dir / "commits").glob("*.json"))) == 8
```

- [ ] **Step 2: Run and verify `run_experiment` is missing**

```bash
rtk proxy python -m pytest ADP/test_experiment.py::test_runner_pairs_inputs_saves_failures_and_resumes -q
```

Expected: FAIL because `run_experiment` is undefined.

- [ ] **Step 3: Implement RSS sampling and quality helpers**

Add:

```python
import threading
import time
import traceback

import psutil
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


class _RSSSampler:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.samples = []
        self.stop = threading.Event()
        self.process = psutil.Process()
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)

    def _sample(self) -> None:
        self.samples.append(self.process.memory_info().rss / 2**20)

    def _sample_loop(self) -> None:
        while not self.stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self._sample()
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join()
        self._sample()

    def summary(self) -> dict:
        values = np.asarray(self.samples, dtype=float)
        return {
            "algorithm_rss_start_mib": float(values[0]),
            "algorithm_rss_min_mib": float(values.min()),
            "algorithm_rss_mean_mib": float(values.mean()),
            "algorithm_rss_max_mib": float(values.max()),
            "algorithm_rss_peak_delta_mib": float(values.max() - values[0]),
            "algorithm_memory_samples": int(values.size),
            "algorithm_memory_source": "psutil.Process.memory_info().rss",
        }


def _projector_error(estimated, truth) -> float:
    estimated = np.asarray(estimated, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if estimated.ndim == 1:
        estimated = estimated[:, None]
    if truth.ndim == 1:
        truth = truth[:, None]
    estimated, _ = np.linalg.qr(estimated, mode="reduced")
    truth, _ = np.linalg.qr(truth, mode="reduced")
    return float(
        np.linalg.norm(estimated @ estimated.T - truth @ truth.T, ord="fro")
        / np.sqrt(2.0 * truth.shape[1])
    )
```

- [ ] **Step 4: Implement one job execution**

Add `_execute_job(store, experiment, job, data, save_models, progress_callback)`. It must:

1. build the model with `_build_model`;
2. enter `threadpool_limits(limits=1)` and `_RSSSampler()`;
3. time only `model.fit(data.X, data.Y, progress=progress_callback)`;
4. calculate `cosine_abs` for single truth and `_projector_error` for either mode;
5. flatten `model.profile_`, stop reason, trace length, requested/effective config, and solver identity into `run`;
6. serialize actual trace rows through `_outer_rows`, implemented below in this step;
7. accept `success` only for finite fitted outputs with stop reason `h_min` or `local_mass_limit`, then write a model NPZ only when `save_models=True`;
8. catch `RuntimeError("outer_steps exhausted before reaching h_min")` as `nonconverged` and every other `Exception` or invalid fitted result as `numerical_failure`;
9. include `error_type`, `error_message`, and `traceback.format_exc()` for errors.

Use this snapshot code:

```python
def _save_model(store: _SeriesStore, job: _Job, model) -> str:
    values = {}
    if hasattr(model, "beta_"):
        values["index"] = np.asarray(model.beta_)
    if hasattr(model, "eigenvalues_"):
        values["eigenvalues"] = np.asarray(model.eigenvalues_)
    if getattr(model, "coefficients_", None) is not None:
        values["coefficients"] = np.asarray(model.coefficients_)
    path = store.model_dir / f"{job.run_id}.npz"
    _atomic_npz(path, **values)
    return str(path.relative_to(store.series_dir))
```

Implement trace serialization in this task so `_execute_job` is runnable before CSV export exists:

```python
def _json_array(value) -> str:
    return json.dumps(np.asarray(value).tolist(), separators=(",", ":"))


def _outer_rows(job, experiment, data, model, series_id):
    rows = []
    for item in model.result_.trace:
        estimate = item.get("beta", item.get("basis"))
        structural = {"k", "h", "rho", "alpha", "beta", "basis", "eigenvalues", "mean_mass", "stop_reason"}
        diagnostics = {key: _spec_value(value) for key, value in item.items() if key not in structural}
        cosine = ""
        projector = ""
        if data.true_index is not None:
            projector = _projector_error(estimate, data.true_index)
            if experiment.mode == "single":
                left = np.asarray(estimate) / np.linalg.norm(estimate)
                right = np.asarray(data.true_index) / np.linalg.norm(data.true_index)
                cosine = abs(float(left @ right))
        rows.append(
            {
                "schema_version": 1,
                "series_id": series_id,
                "run_id": job.run_id,
                "experiment": experiment.name,
                "point": job.point.name,
                "variant": job.variant_name,
                "seed": job.seed,
                "local_solver": _solver_name(model.solver.method),
                "mode": experiment.mode,
                "outer_k": int(item["k"]),
                "h_k": float(item["h"]),
                "rho_k": item.get("rho", ""),
                "alpha_k": item.get("alpha", ""),
                "beta_k": _json_array(item["beta"]) if "beta" in item else "",
                "basis_k": _json_array(item["basis"]) if "basis" in item else "",
                "eigenvalues": _json_array(item["eigenvalues"]) if "eigenvalues" in item else "",
                "cosine_abs": cosine,
                "projector_error": projector,
                "beta_delta": item.get("beta_delta", ""),
                "objective_after": item.get("objective", ""),
                "inner_iterations": item.get("inner_iterations", ""),
                "linear_solver_iterations": item.get("lsmr_iterations", ""),
                "solver_diagnostics": json.dumps(diagnostics, separators=(",", ":")),
                "local_mass_mean": item.get("mean_mass", ""),
                "stop_reason": item.get("stop_reason", ""),
            }
        )
    return rows
```

- [ ] **Step 5: Implement the serial runner and two tqdm levels**

Add:

```python
def run_experiment(
    experiment: ADP_Experiment,
    output_dir: str | Path,
    *,
    resume: str | Path | None = None,
    save_models: bool = True,
    show_progress: bool = True,
) -> tuple[Path, int]:
    experiment = validate_experiment(experiment)
    jobs = _build_jobs(experiment)
    store = (
        _SeriesStore.resume(Path(resume))
        if resume is not None
        else _SeriesStore.create(Path(output_dir), experiment)
    )
    pending = [job for job in jobs if not store.is_complete(job, experiment)]
    data_cache = {}
    outer = tqdm(
        total=len(jobs),
        initial=len(jobs) - len(pending),
        desc=experiment.name,
        unit="fit",
        dynamic_ncols=True,
        disable=not show_progress or not sys.stderr.isatty(),
    )
    try:
        for job in pending:
            key = (job.point.name, job.seed)
            if key not in data_cache:
                path = store.data_path(job.point, job.seed)
                try:
                    data_cache[key] = (
                        store.load_data(job.point, job.seed)
                        if path.exists()
                        else make_data(experiment, job.point, job.seed)
                    )
                    if not path.exists():
                        store.save_data(job.point, job.seed, data_cache[key])
                except Exception as error:
                    data_cache[key] = error
            data = data_cache[key]
            if isinstance(data, Exception):
                outcome = _failure_outcome(job, data)
            else:
                inner = tqdm(
                    desc=f"{job.point.name}/{job.variant_name}",
                    unit="outer",
                    leave=False,
                    position=1,
                    disable=outer.disable,
                )
                def advance(item):
                    inner.set_postfix(
                        h=item.get("h"),
                        localization=item.get("rho", item.get("alpha")),
                        refresh=False,
                    )
                    inner.update(1)
                try:
                    outcome = _execute_job(
                        store,
                        experiment,
                        job,
                        data,
                        save_models,
                        advance,
                    )
                finally:
                    inner.close()
            store.commit(job, experiment, outcome)
            outer.set_postfix(point=job.point.name, variant=job.variant_name, status=outcome["run"]["status"])
            outer.update(1)
            if show_progress and not sys.stderr.isatty():
                print(f"{job.run_id}: {outcome['run']['status']}")
    finally:
        outer.close()
    failures = sum(commit["run"]["status"] != "success" for commit in store.read_commits())
    return store.series_dir, failures
```

Import `sys`, `validate_experiment`, and `make_data`. Implement `_failure_outcome(job, error)` through the same base-run helper as `_execute_job`: it returns `status="numerical_failure"`, empty quality/artifact/telemetry fields, `error_type`, `error_message`, `traceback.format_exception(...)`, and `outer=[]`. Caching the exception makes both paired variants fail consistently without calling the factory twice. Do not catch `KeyboardInterrupt` as a numerical failure.

- [ ] **Step 6: Export the runner entry point and run tests**

Add `run_experiment` to `ADP/__init__.py`, then run:

```bash
rtk proxy python -m pytest ADP/test_experiment.py -q
```

Expected: the paired runner test passes. CSV/report hooks are intentionally added in Tasks 5 and 7; no temporary report stub is created.

- [ ] **Step 7: Commit**

```bash
rtk git add ADP/experiment_runner.py ADP/__init__.py ADP/test_experiment.py
rtk git commit -m "feat: execute paired ADP experiments"
```

### Task 5: Main-compatible CSV and paired comparisons

**Files:**
- Modify: `ADP/experiment_runner.py`
- Modify: `ADP/test_experiment.py`

- [ ] **Step 1: Add failing CSV assertions**

Extend the paired runner test:

```python
import csv


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


# after the first run in test_runner_pairs_inputs_saves_failures_and_resumes
runs = _read_csv(series_dir / "run_summary.csv")
outer_rows = _read_csv(series_dir / "outer_iterations.csv")
paired = _read_csv(series_dir / "paired_comparison.csv")
summary = _read_csv(series_dir / "comparison_summary.csv")
series = _read_csv(series_dir / "series.csv")

assert len(runs) == 8
assert len(outer_rows) == 8
assert len(paired) == 4
assert summary
assert series[0]["status"] == "complete"
assert {"variant", "cosine_abs", "projector_error", "fit_wall_time_sec", "algorithm_rss_max_mib"} <= set(runs[0])
assert {
    "A_variant", "B_variant", "A_seed", "B_seed",
    "A_data_artifact", "B_data_artifact", "delta_fit_wall_time_sec",
} <= set(paired[0])
assert "winner" not in paired[0]
for name in (
    "run_summary.csv",
    "outer_iterations.csv",
    "paired_comparison.csv",
    "comparison_summary.csv",
):
    text = (series_dir / name).read_text(encoding="utf-8").lower()
    assert ",nan" not in text and ",inf" not in text and ",-inf" not in text
```

- [ ] **Step 2: Run and verify missing/empty tables**

```bash
rtk proxy python -m pytest ADP/test_experiment.py::test_runner_pairs_inputs_saves_failures_and_resumes -q
```

Expected: FAIL because `_export_tables` is incomplete.

- [ ] **Step 3: Define stable base schemas**

Add tuple constants. Keep exact main names needed by reports:

```python
RUN_COLUMNS = (
    "schema_version", "series_id", "run_id", "experiment", "point", "variant",
    "variant_index", "mode", "index_dim", "seed", "n", "d", "n_over_d",
    "noise", "requested_solver", "requested_solver_settings", "local_solver",
    "effective_solver_settings", "h_initial", "h_final",
    "rho_final", "alpha_final", "outer_iterations", "cosine_abs",
    "projector_error", "fit_wall_time_sec", "algorithm_time_sec",
    "algorithm_rss_start_mib", "algorithm_rss_min_mib",
    "algorithm_rss_mean_mib", "algorithm_rss_max_mib",
    "algorithm_rss_peak_delta_mib", "algorithm_memory_samples",
    "algorithm_memory_source", "tracemalloc_peak_mib", "stop_reason", "status",
    "error_type", "error_message", "error_traceback", "data_artifact",
    "model_artifact", "outer_row_count", "inner_row_count", "local_row_count",
    "solver_row_count",
)

OUTER_COLUMNS = (
    "schema_version", "series_id", "run_id", "experiment", "point", "variant",
    "seed", "local_solver", "mode", "outer_k", "h_k", "rho_k", "alpha_k",
    "beta_k", "basis_k", "eigenvalues", "cosine_abs", "projector_error",
    "beta_delta", "objective_after", "inner_iterations",
    "linear_solver_iterations", "solver_diagnostics", "local_mass_mean", "stop_reason",
)

DETAIL_HEADERS = {
    "inner_iterations.csv": ("schema_version", "series_id", "run_id", "outer_k", "inner_k", "objective", "beta_delta"),
    "local_diagnostics.csv": ("schema_version", "series_id", "run_id", "outer_k", "center_j", "local_mass", "ess", "condition"),
    "solver_iterations.csv": ("schema_version", "series_id", "run_id", "outer_k", "inner_k", "solver_k", "relative_residual"),
}
```

Append `adp_<field>` for every requested `ADP_Config` field and `effective_<field>` for every field on the replaced `model.config`, overriding the latter with values from `model.effective_parameters_` such as computed `N_lin`, `N_phi`, `N_J`, and `h_min`. Then append stage columns `stage_<name>_time_sec` and `stage_<name>_memory_mib` plus sorted point metadata keys. `requested_solver` is the variant token (`auto|lsmr|varpro`); `local_solver` is the effective callable name (`lsmr|varpro`); requested and effective settings are separate compact-JSON columns. Serialize callables as `module:qualname`, arrays/dicts as compact JSON, `None` and nonfinite metric values as empty CSV cells, and reject metadata keys that collide with base columns.

- [ ] **Step 4: Keep detail telemetry honest**

Use the `_outer_rows` implementation introduced in Task 4 unchanged. Write its real rows to `outer_iterations.csv`. Leave `inner_iterations.csv`, `local_diagnostics.csv`, and `solver_iterations.csv` header-only until row-level telemetry actually exists; do not expand aggregate counters into invented histories.

- [ ] **Step 5: Implement atomic CSV exports and paired tables**

Use `csv.DictWriter` and `os.replace` for every public table. `_export_tables` must read all commits, flatten their `run` and `outer` rows, and write:

```python
PAIR_METRICS = (
    "cosine_abs",
    "projector_error",
    "fit_wall_time_sec",
    "algorithm_time_sec",
    "algorithm_rss_peak_delta_mib",
    "outer_iterations",
)
```

For exactly two variants, join rows on `(point, seed)`. Each row contains `A_variant,B_variant,A_seed,B_seed,A_status,B_status,A_data_artifact,B_data_artifact`, both prefixed metric values, and each finite `delta_<metric>` as `B - A`, where A/B follow `variant_index`. Extend `PAIR_METRICS` with every discovered `stage_*_time_sec` column so stage timings are compared too. For one variant, write only the stable header. Build `comparison_summary.csv` in long form with columns `point,metric,count,q05,median,q95` from finite deltas. Never add `winner`.

Write `series.csv` as one row with:

```text
schema_version,series_id,name,mode,index_dim,runs,seed,total_jobs,
completed_jobs,success_jobs,nonconverged_jobs,numerical_failure_jobs,status
```

Wire `_export_tables` into `run_experiment`: call it with `status="partial"` immediately after every atomic commit, then call it once from the runner's `finally` block with `status="complete"` only when all planned commits exist and `"partial"` otherwise. This guarantees usable CSV after ordinary failures and `KeyboardInterrupt`; let the interrupt propagate so the CLI returns `130`.

- [ ] **Step 6: Run tests**

```bash
rtk proxy python -m pytest ADP/test_experiment.py -q
```

Expected: all tests pass and no CSV contains `nan`/`inf` literals.

- [ ] **Step 7: Commit**

```bash
rtk git add ADP/experiment_runner.py ADP/test_experiment.py
rtk git commit -m "feat: persist ADP experiment comparisons"
```

### Task 6: Main-derived plotting primitives

**Files:**
- Create: `ADP/experiment_reports.py`
- Modify: `ADP/test_experiment.py`
- Read-only reference: `main:adp/evaluation/single_index/reports.py`
- Read-only reference: `main:adp/evaluation/single_index/plots.py`
- Read-only reference: `main:adp/common/plotting.py`

- [ ] **Step 1: Add failing aggregation and renderer tests**

Append:

```python
def test_report_aggregates_match_main_conventions(tmp_path: Path):
    import pandas as pd
    from ADP.experiment_reports import _quantiles, _wilson, _render_quantile

    frame = pd.DataFrame(
        {
            "variant": ["A", "A", "A", "B", "B", "B"],
            "outer_k": [0, 0, 1, 0, 0, 1],
            "cosine_abs": [0.7, 0.9, 0.95, 0.6, 0.8, 0.85],
        }
    )
    summary = _quantiles(frame, "outer_k", "cosine_abs", ("variant",))
    assert set(summary) >= {"variant", "outer_k", "q05", "median", "q95"}
    assert np.isclose(summary.loc[summary["variant"].eq("A") & summary["outer_k"].eq(0), "median"].iloc[0], 0.8)
    low, center, high = _wilson(8, 10)
    assert 0.0 <= low < center < high <= 1.0

    output = tmp_path / "quantile.png"
    _render_quantile(
        frame,
        output,
        x="outer_k",
        y="cosine_abs",
        groups=("variant",),
        title="Качество",
        xlabel="Итерация",
        ylabel="Косинус",
    )
    assert output.stat().st_size > 0
```

- [ ] **Step 2: Run and verify missing plotting functions**

```bash
rtk proxy env MPLCONFIGDIR=/tmp/adp_matplotlib python -m pytest ADP/test_experiment.py::test_report_aggregates_match_main_conventions -q
```

Expected: FAIL for missing functions.

- [ ] **Step 3: Implement style, quantiles, Wilson intervals, and figure lifecycle**

Inspect the three approved `main` sources without switching branches:

```bash
rtk git show main:adp/evaluation/single_index/reports.py
rtk git show main:adp/evaluation/single_index/plots.py
rtk git show main:adp/common/plotting.py
```

At module import, set headless config before pyplot:

```python
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/adp_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ADP_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#ea580c", "#0891b2")
ADP_AXIS_FACE = "#f8fafc"
ADP_FIGURE_FACE = "#ffffff"
ADP_GRID_COLOR = "#cbd5e1"
ADP_TEXT_COLOR = "#111827"
ADP_SPINE_COLOR = "#94a3b8"


def _quantiles(frame, x, y, groups=()):
    keys = [*groups, x]
    source = frame[[*keys, y]].copy()
    source[y] = pd.to_numeric(source[y], errors="coerce")
    source = source.loc[source[x].notna() & np.isfinite(source[y])]
    return source.groupby(keys, sort=True, dropna=False, as_index=False).agg(
        q05=(y, lambda values: values.quantile(0.05)),
        median=(y, "median"),
        q95=(y, lambda values: values.quantile(0.95)),
    )


def _wilson(successes, total, z=1.959963984540054):
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return center - half, center, center + half


def _new_axis():
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.patch.set_facecolor(ADP_FIGURE_FACE)
    ax.set_facecolor(ADP_AXIS_FACE)
    ax.set_prop_cycle(color=ADP_COLORS)
    ax.grid(axis="y", color=ADP_GRID_COLOR, alpha=0.75, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(ADP_SPINE_COLOR)
    ax.spines["bottom"].set_color(ADP_SPINE_COLOR)
    return fig, ax


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
```

- [ ] **Step 4: Implement the seven generic plot kinds**

Add `_render_quantile`, `_render_median_line`, `_render_proportion`, `_render_box`, `_render_scatter`, `_render_heatmap`, and `_render_stacked`. Each accepts a filtered frame, output path, x/y, groups, Russian title/xlabel/ylabel, and scale/limit options. Requirements:

- quantile: line+circle markers for median and `fill_between` q05/q95 at alpha `0.18`;
- median line: group median with no fabricated interval;
- proportion: group success/total and 95% Wilson error bars;
- box: raw finite values with 5/25/50/75/95 percentiles;
- scatter: raw finite x/y pairs, grouped color/legend, and no aggregation;
- heatmap: one aggregate-pivot panel per grouping (therefore one panel per variant), cell annotations, and a shared colorbar using `PlotSpec.value`; use mean for technical-success rates and median otherwise;
- stacked: median component values with grouped bars per variant, except mean status indicators, optionally normalized to one;
- no finite data: return `False` without creating a PNG;
- log/log2/symlog and `[0,1]` quality limits follow the `PlotSpec` fields.

Use the same grouping loop in line renderers:

```python
for index, (label, subset) in enumerate(summary.groupby(list(groups), sort=False, dropna=False)):
    label = label if isinstance(label, tuple) else (label,)
    color = ADP_COLORS[index % len(ADP_COLORS)]
    ax.plot(subset[x], subset["median"], marker="o", color=color, label=", ".join(map(str, label)))
    ax.fill_between(subset[x], subset["q05"], subset["q95"], color=color, alpha=0.18)
```

- [ ] **Step 5: Run the renderer test**

```bash
rtk proxy env MPLCONFIGDIR=/tmp/adp_matplotlib python -m pytest ADP/test_experiment.py::test_report_aggregates_match_main_conventions -q
```

Expected: `1 passed` and a non-empty PNG.

- [ ] **Step 6: Commit**

```bash
rtk git add ADP/experiment_reports.py ADP/test_experiment.py
rtk git commit -m "feat: add main-style ADP plot primitives"
```

### Task 7: Applicable plot manifest, multi-index adaptations, and reports-only

**Files:**
- Modify: `ADP/experiment_reports.py`
- Modify: `ADP/experiment_runner.py`
- Modify: `ADP/test_experiment.py`

- [ ] **Step 1: Add failing single/multi report tests**

Build synthetic `run_summary.csv` and `outer_iterations.csv` tables directly, call `write_reports`, and assert:

```python
def test_reports_create_single_and_multi_specific_plots(tmp_path: Path):
    import pandas as pd
    from ADP.experiment_reports import write_reports

    single = tmp_path / "single"
    single.mkdir()
    pd.DataFrame(
        [{
            "series_id": "s", "run_id": "r", "experiment": "p", "point": "p",
            "variant": "A", "mode": "single", "seed": 1, "d": 3, "n_over_d": 10,
            "sigma_eps": 0.1, "link": "sin",
            "status": "success", "cosine_abs": 0.9, "projector_error": 0.2,
            "algorithm_time_sec": 1.0, "algorithm_rss_max_mib": 20.0,
            "stage_initialization_time_sec": 0.1, "stage_directions_time_sec": 0.1,
            "stage_statistics_time_sec": 0.3, "stage_solver_time_sec": 0.4,
            "stage_update_time_sec": 0.1,
            "outer_iterations": 1,
        }]
    ).to_csv(single / "run_summary.csv", index=False)
    pd.DataFrame(
        [{
            "series_id": "s", "run_id": "r", "experiment": "p", "point": "p",
            "variant": "A", "mode": "single", "seed": 1, "outer_k": 0,
            "h_k": 1.0, "rho_k": 0.8, "alpha_k": np.nan,
            "cosine_abs": 0.9, "projector_error": 0.2,
        }]
    ).to_csv(single / "outer_iterations.csv", index=False)
    write_reports(single)
    assert (single / "plots/points/p/quality_vs_outer_iteration.png").is_file()
    assert (single / "plots/points/p/rho_vs_outer_iteration.png").is_file()
    assert (single / "plots/summary/quality_heatmap_d_nd_ratio.png").is_file()
    assert (single / "plots/summary/success_rate_vs_sigma_eps.png").is_file()
    assert (single / "plots/summary/quality_by_link_function.png").is_file()
    assert (single / "plots/summary/runtime_vs_dimension.png").is_file()
    assert (single / "plots/summary/runtime_breakdown.png").is_file()
    assert (single / "plots/summary/status_breakdown.png").is_file()

    multi = tmp_path / "multi"
    multi.mkdir()
    runs = pd.read_csv(single / "run_summary.csv").assign(mode="multi", cosine_abs=np.nan)
    outer = pd.read_csv(single / "outer_iterations.csv").assign(
        mode="multi", cosine_abs=np.nan, rho_k=np.nan, alpha_k=0.7
    )
    runs.to_csv(multi / "run_summary.csv", index=False)
    outer.to_csv(multi / "outer_iterations.csv", index=False)
    write_reports(multi)
    assert (multi / "plots/points/p/projector_error_vs_outer_iteration.png").is_file()
    assert (multi / "plots/points/p/alpha_vs_outer_iteration.png").is_file()
    assert not (multi / "plots/points/p/quality_vs_outer_iteration.png").exists()

    artifacts = pd.read_csv(multi / "artifacts.csv")
    assert {"created", "skipped"} <= set(artifacts["status"])

    truthless = tmp_path / "truthless"
    truthless.mkdir()
    runs.assign(cosine_abs=np.nan, projector_error=np.nan).to_csv(
        truthless / "run_summary.csv", index=False
    )
    outer.assign(cosine_abs=np.nan, projector_error=np.nan).to_csv(
        truthless / "outer_iterations.csv", index=False
    )
    write_reports(truthless)
    assert not (truthless / "plots/points/p/quality_vs_outer_iteration.png").exists()
    assert not (truthless / "plots/points/p/projector_error_vs_outer_iteration.png").exists()
```

Lock the approved `main` filenames and their metadata gates in the same test module:

```python
def test_plot_manifest_covers_approved_main_families():
    from ADP.experiment_reports import PLOT_MANIFEST

    diagnostic = {
        "quality_vs_outer_iteration.png",
        "projector_error_vs_outer_iteration.png",
        "bandwidth_vs_outer_iteration.png",
        "rho_vs_outer_iteration.png",
        "beta_step_vs_outer_iteration.png",
        "objective_vs_outer_iteration.png",
        "objective_vs_inner_iteration.png",
        "beta_step_vs_inner_iteration.png",
        "solver_residual_vs_iteration.png",
        "local_mass_by_outer_iteration.png",
        "effective_neighbors_by_outer_iteration.png",
        "local_condition_by_outer_iteration.png",
        "mass_vs_condition.png",
        "local_slopes_by_outer_iteration.png",
        "runtime_breakdown.png",
        "runtime_share_breakdown.png",
        "status_breakdown.png",
    }
    families = {
        ("quality_heatmap_d_nd_ratio.png", ("d", "n_over_d")),
        ("success_rate_heatmap.png", ("d", "n_over_d")),
        ("runtime_vs_dimension.png", ("d", "n_over_d")),
        ("memory_vs_dimension.png", ("d", "n_over_d")),
        ("iterations_heatmap_d_nd_ratio.png", ("d", "n_over_d")),
        ("quality_vs_sigma_eps.png", ("sigma_eps",)),
        ("success_rate_vs_sigma_eps.png", ("sigma_eps",)),
        ("runtime_vs_sigma_eps.png", ("sigma_eps",)),
        ("outer_iterations_vs_sigma_eps.png", ("sigma_eps",)),
        ("final_objective_vs_sigma_eps.png", ("sigma_eps",)),
        ("quality_vs_correlation.png", ("rho_corr",)),
        ("success_rate_vs_correlation.png", ("rho_corr",)),
        ("local_condition_vs_correlation.png", ("rho_corr",)),
        ("solver_iterations_vs_correlation.png", ("rho_corr",)),
        ("runtime_vs_correlation.png", ("rho_corr",)),
        ("singular_fraction_vs_correlation.png", ("rho_corr",)),
        ("quality_vs_sigma_x.png", ("sigma_x",)),
        ("h0_vs_sigma_x.png", ("sigma_x",)),
        ("final_bandwidth_vs_sigma_x.png", ("sigma_x",)),
        ("local_mass_vs_sigma_x.png", ("sigma_x",)),
        ("runtime_vs_sigma_x.png", ("sigma_x",)),
        ("bandwidth_ratio_vs_sigma_x.png", ("sigma_x",)),
        ("quality_by_link_function.png", ("link",)),
        ("success_rate_by_link_function.png", ("link",)),
        ("outer_iterations_by_link_function.png", ("link",)),
        ("objective_by_link_function.png", ("link",)),
        ("local_slopes_by_link_function.png", ("link",)),
        ("quality_by_x_distribution.png", ("x_distribution",)),
        ("quality_by_noise_distribution.png", ("noise_distribution",)),
        ("quality_by_heteroscedasticity.png", ("heteroscedastic",)),
        ("quality_vs_outlier_fraction.png", ("effective_outlier_fraction",)),
        ("failure_rate_vs_outliers.png", ("effective_outlier_fraction",)),
        ("quality_vs_model_misspecification.png", ("delta",)),
        ("objective_vs_model_misspecification.png", ("delta",)),
    }

    assert diagnostic <= {spec.filename for spec in PLOT_MANIFEST}
    assert families <= {(spec.filename, spec.required_metadata) for spec in PLOT_MANIFEST}
    shared_distributions = {
        spec.filename: spec.required_any_metadata
        for spec in PLOT_MANIFEST
        if spec.filename in {"failure_rate_by_distribution.png", "runtime_by_distribution.png"}
    }
    assert shared_distributions == {
        "failure_rate_by_distribution.png": ("x_distribution", "noise_distribution"),
        "runtime_by_distribution.png": ("x_distribution", "noise_distribution"),
    }
    assert all("variant" in spec.groups for spec in PLOT_MANIFEST)
    assert "correctness_rate.png" not in {spec.filename for spec in PLOT_MANIFEST}
```

- [ ] **Step 2: Run and verify manifest/report failure**

```bash
rtk proxy env MPLCONFIGDIR=/tmp/adp_matplotlib python -m pytest ADP/test_experiment.py::test_reports_create_single_and_multi_specific_plots -q
```

Expected: FAIL because `write_reports` and the manifest are not implemented yet.

- [ ] **Step 3: Add compact data-driven manifest**

Define:

```python
@dataclass(frozen=True, slots=True)
class PlotSpec:
    filename: str
    table: str
    kind: str
    x: str
    y: str
    title: str
    xlabel: str
    ylabel: str
    scope: str = "summary"
    groups: tuple[str, ...] = ("variant",)
    required_metadata: tuple[str, ...] = ()
    required_any_metadata: tuple[str, ...] = ()
    value: str | None = None
    components: tuple[str, ...] = ()
    xscale: str = "linear"
    yscale: str = "linear"
    ylim: tuple[float, float] | None = None
    value_limits: tuple[float, float] | None = None
    normalize: bool = False
    aggregate: str = "median"
```

Populate `PLOT_MANIFEST` from the exact filename families locked by the preceding test. Mark outer/inner/local/solver diagnostic specs as `scope="point"`; metadata families and the three runtime/status breakdowns remain `scope="summary"`. Include all base filenames from `main` except `correctness_rate.png`; make `required_metadata` explicit for `sigma_eps`, `rho_corr`, `sigma_x`, `link`, distributions, heteroscedasticity, outliers, and `delta`. Every comparison grouping includes `variant` in addition to the original `main` grouping fields. Set `aggregate="mean"` for `success_rate_heatmap.png` and `status_breakdown.png`; other aggregate plots use medians or their explicitly named Wilson/proportion path. Use run-table `stage_initialization_time_sec`, `stage_directions_time_sec`, `stage_statistics_time_sec`, `stage_solver_time_sec`, and `stage_update_time_sec` components for `runtime_breakdown.png` and `runtime_share_breakdown.png` because the current models expose aggregate stage timings rather than per-outer timing.

For mode adaptation, use one pure function:

```python
def _for_mode(spec: PlotSpec, mode: str) -> PlotSpec | None:
    if mode == "single":
        return spec
    if spec.filename == "rho_vs_outer_iteration.png":
        return replace(
            spec,
            filename="alpha_vs_outer_iteration.png",
            y="alpha_k",
            ylabel="Параметр локализации alpha",
        )
    if "quality" in spec.filename:
        heatmap = spec.kind == "heatmap"
        return replace(
            spec,
            filename=spec.filename.replace("quality", "projector_error"),
            y=spec.y if heatmap else "projector_error",
            value="projector_error" if heatmap else spec.value,
            ylabel="Ошибка проектора",
            ylim=spec.ylim if heatmap else (0.0, 1.0),
            value_limits=(0.0, 1.0) if heatmap else spec.value_limits,
        )
    if spec.filename.startswith("beta_step_"):
        return replace(spec, ylabel="Шаг базиса")
    if spec.y in {"rho_k", "slope"}:
        return None
    return spec
```

- [ ] **Step 4: Implement report selection and artifact manifest**

`write_reports(series_dir)` must:

1. read `run_summary.csv`, `outer_iterations.csv`, `inner_iterations.csv`, `local_diagnostics.csv`, and `solver_iterations.csv`; treat a missing optional detail table as empty so `--reports-only` can still document skipped plots, while `run_summary.csv` remains required;
2. left-join point metadata from runs into every detail table by `run_id`;
3. process `scope="point"` specs separately for every point into `plots/points/<point>`;
4. process `scope="summary"` specs once into `plots/summary`;
5. add `success_value = status == 'success'`, `failure_value = status == 'numerical_failure'`, and one numeric indicator column for each technical status used by `status_breakdown.png`, without quality thresholds; derive `distribution` by coalescing `x_distribution` and `noise_distribution` for the two shared main plots;
6. require every `required_metadata` field, at least one field from `required_any_metadata`, all x/y/value/group/component columns, and at least one finite metric observation;
7. write one `artifacts.csv` row per considered spec with `filename,path,status,source_tables,error`;
8. remove a stale PNG when a rerender changes its artifact from created to skipped;
9. deduplicate mode-adapted specs by final `(path, table)` so multi-index `quality` adaptation cannot overwrite an existing projector plot;
10. return the `artifacts.csv` path.

Dispatch exactly by `kind`; errors in one plot become `status=error` and do not cancel later plots.

Finally, import `write_reports` lazily in `run_experiment` and call it after the final `_export_tables` in the existing `finally` block. This produces complete or partial plots on normal completion, ordinary job errors, and `KeyboardInterrupt` without importing Matplotlib during library-only model use.

- [ ] **Step 5: Run all report tests**

```bash
rtk proxy env MPLCONFIGDIR=/tmp/adp_matplotlib python -m pytest ADP/test_experiment.py -q
```

Expected: all tests pass, including single `quality/rho` and multi `projector_error/alpha` PNGs.

- [ ] **Step 6: Commit**

```bash
rtk git add ADP/experiment_reports.py ADP/experiment_runner.py ADP/test_experiment.py
rtk git commit -m "feat: generate ADP experiment reports"
```

### Task 8: Unified CLI and example experiment file

**Files:**
- Modify: `ADP/cli.py:1-130`
- Create: `ADP/examples/experiment_compare.py`
- Modify: `ADP/test_experiment.py`

- [ ] **Step 1: Add failing parser, dry-run, and reports-only tests**

Append:

```python
def test_cli_builds_multi_manual_experiment():
    from ADP.cli import build_parser, experiment_from_args

    parser = build_parser()
    args = parser.parse_args(
        ["--mode", "multi", "--index-dim", "2", "--solver", "lsmr", "--runs", "3"]
    )
    experiment = experiment_from_args(args, parser)

    assert experiment.mode == "multi"
    assert experiment.index_dim == 2
    assert experiment.runs == 3
    assert experiment.variants["default"].solver == "lsmr"


def test_cli_dry_run_creates_nothing(tmp_path: Path, capsys):
    from ADP.cli import main

    source = tmp_path / "exp.py"
    source.write_text(
        """
from ADP import ADP_Config, ADP_Experiment, ADP_ExperimentPoint, ADP_ExperimentVariant
experiment = ADP_Experiment(
    name="dry",
    mode="single",
    runs=2,
    points=(ADP_ExperimentPoint("p", 24, 3),),
    variants={"A": ADP_ExperimentVariant(ADP_Config()), "B": ADP_ExperimentVariant(ADP_Config())},
)
""",
        encoding="utf-8",
    )

    code = main(["--experiment-file", str(source), "--output-dir", str(tmp_path), "--dry-run"])

    assert code == 0
    assert "total jobs: 4" in capsys.readouterr().out
    assert not (tmp_path / "dry").exists()


def test_cli_reports_only_uses_archive_without_experiment(tmp_path: Path, monkeypatch):
    import ADP.experiment_reports as reports
    from ADP.cli import main

    called = []
    monkeypatch.setattr(
        reports,
        "write_reports",
        lambda path: called.append(Path(path)) or Path(path) / "artifacts.csv",
    )

    assert main(["--reports-only", str(tmp_path)]) == 0
    assert called == [tmp_path]
```

- [ ] **Step 2: Run and verify missing parser helpers**

```bash
rtk proxy python -m pytest ADP/test_experiment.py -q
```

Expected: failures for missing `build_parser`, `experiment_from_args`, or `main(argv)`.

- [ ] **Step 3: Rewrite the parser around one runner path**

Keep `parse_kernel`, delete the empty `parse_solver`, and implement:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ADP single/multi-index experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("single", "multi"), default="single")
    parser.add_argument("--index-dim", type=int, default=1)
    parser.add_argument("--solver", choices=("auto", "lsmr", "varpro"), default="auto")
    parser.add_argument("--solver-tol", type=float)
    parser.add_argument("--solver-max-steps", type=int)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--experiment-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("ADP/experiment_outputs"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--reports-only", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--N_loc", "--n-loc", dest="N_loc", type=int, default=10)
    parser.add_argument("--N_lin", "--n-lin", dest="N_lin", type=int)
    parser.add_argument("--N_J", "--J", dest="N_J", type=int, default=64)
    parser.add_argument("--N_phi", "--n-phi", dest="N_phi", type=int)
    parser.add_argument("--outer_steps", "--outer-steps", dest="outer_steps", type=int)
    parser.add_argument("--lambda_penalty", "--lambda-penalty", dest="lambda_penalty", type=float, default=100.0)
    parser.add_argument("--local_ridge", "--local-ridge", dest="local_ridge", type=float, default=1e-8)
    parser.add_argument("--kernel", type=parse_kernel, default="epanechnikov")
    parser.add_argument("--a", type=float, default=np.sqrt(2))
    parser.add_argument("--h_min", "--h-min", dest="h_min", type=float)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=32)
    parser.add_argument("--index_init", "--index-init", dest="index_init", choices=("local", "random"), default="local")
    return parser
```

`experiment_from_args(args, parser)` validates direct values, creates one point named `manual`, one variant named `default`, and uses only non-`None` solver settings:

```python
try:
    config = ADP_Config(
        seed=args.seed,
        N_loc=args.N_loc,
        N_lin=args.N_lin,
        N_J=args.N_J,
        N_phi=args.N_phi,
        outer_steps=args.outer_steps,
        lambda_penalty=args.lambda_penalty,
        local_ridge=args.local_ridge,
        kernel=args.kernel,
        a=args.a,
        h_min=args.h_min,
        batch_size=args.batch_size,
        index_init=args.index_init,
    )
except (TypeError, ValueError) as error:
    parser.error(str(error))
settings = {}
if args.solver_tol is not None:
    settings["tol"] = args.solver_tol
if args.solver_max_steps is not None:
    settings["max_steps"] = args.solver_max_steps
return validate_experiment(
    ADP_Experiment(
        name="manual",
        mode=args.mode,
        index_dim=args.index_dim,
        runs=args.runs,
        seed=args.seed,
        points=(ADP_ExperimentPoint("manual", args.n, args.d, args.noise),),
        variants={
            "default": ADP_ExperimentVariant(config, args.solver, settings)
        },
    )
)
```

- [ ] **Step 4: Implement CLI operation routing**

Use:

```python
def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.reports_only is not None:
        if args.experiment_file is not None or args.resume is not None or args.dry_run:
            parser.error("--reports-only cannot be combined with run options")
        from ADP.experiment_reports import write_reports
        artifacts = write_reports(args.reports_only)
        print(f"отчёты обновлены: {artifacts}")
        return 0

    try:
        experiment = (
            load_experiment(args.experiment_file)
            if args.experiment_file is not None
            else experiment_from_args(args, parser)
        )
        jobs = _build_jobs(experiment)
        if args.dry_run:
            for point in experiment.points:
                print(f"point={point.name} runs={experiment.runs}")
            print(f"variants: {', '.join(experiment.variants)}")
            print(f"total jobs: {len(jobs)}")
            return 0
        series_dir, failures = run_experiment(
            experiment,
            args.output_dir,
            resume=args.resume,
            save_models=not args.no_save_models,
            show_progress=not args.no_progress,
        )
    except KeyboardInterrupt:
        return 130
    except (ImportError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(f"серия сохранена: {series_dir}")
    print(f"ошибок: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

When `--experiment-file` is present, do not read direct model/data flags. `--output-dir`, `--resume`, `--dry-run`, `--no-save-models`, and `--no-progress` remain operational.

- [ ] **Step 5: Add the example file**

Create `ADP/examples/experiment_compare.py`:

```python
from ADP import ADP_Config, ADP_Experiment, ADP_ExperimentPoint, ADP_ExperimentVariant

common = ADP_Config(
    N_loc=10,
    N_lin=20,
    N_J=64,
    N_phi=10,
    lambda_penalty=100.0,
    local_ridge=1e-8,
)

experiment = ADP_Experiment(
    name="lsmr_vs_varpro",
    mode="single",
    runs=10,
    seed=7,
    points=(
        ADP_ExperimentPoint("noise_005", 500, 10, 0.05, {"sigma_eps": 0.05}),
        ADP_ExperimentPoint("noise_020", 500, 10, 0.20, {"sigma_eps": 0.20}),
    ),
    variants={
        "lsmr": ADP_ExperimentVariant(common, "lsmr", {"max_steps": 5}),
        "varpro": ADP_ExperimentVariant(common, "varpro", {"max_steps": 100}),
    },
)
```

- [ ] **Step 6: Run CLI tests and help**

```bash
rtk proxy python -m pytest ADP/test_experiment.py -q
rtk python ADP/cli.py --help
rtk python ADP/cli.py --experiment-file ADP/examples/experiment_compare.py --dry-run
```

Expected: tests pass; help lists all new flags; dry-run reports 40 jobs and creates no output directory.

- [ ] **Step 7: Commit**

```bash
rtk git add ADP/cli.py ADP/examples/experiment_compare.py ADP/test_experiment.py
rtk git commit -m "feat: expose unified ADP experiment CLI"
```

### Task 9: Real smoke runs and final regression verification

**Files:**
- Modify only if verification exposes an in-scope defect.

- [ ] **Step 1: Compile every changed module**

```bash
rtk python -m compileall -q ADP
```

Expected: exit code `0` with no syntax errors.

- [ ] **Step 2: Run focused and existing regressions**

```bash
rtk proxy env MPLCONFIGDIR=/tmp/adp_matplotlib python -m pytest ADP/test_experiment.py test_ADP_multi_index.py test_ADP_statistic.py -q
```

Expected: all tests pass. A failure in the pre-existing dirty `VarPro.py` must be diagnosed without restoring that file.

- [ ] **Step 3: Run a real single-index manual smoke**

```bash
rtk python ADP/cli.py --mode single --solver lsmr --runs 2 --n 40 --d 3 --N_loc 5 --N_lin 8 --N_J 8 --N_phi 3 --h_min 1000000 --output-dir /tmp/adp-unified-single --no-progress --no-save-models
```

Expected: exit `0`, two commits, one raw data NPZ per seed, no model NPZ, CSV files, and created diagnostic PNGs.

- [ ] **Step 4: Run a real multi-index manual smoke**

```bash
rtk python ADP/cli.py --mode multi --index-dim 2 --solver lsmr --runs 2 --n 48 --d 4 --N_loc 6 --N_lin 10 --N_J 8 --N_phi 3 --h_min 1000000 --index-init random --output-dir /tmp/adp-unified-multi --no-progress
```

Expected: exit `0`; model NPZ contains `index`, `eigenvalues`, and `coefficients`; plots include `projector_error_vs_outer_iteration.png` and `alpha_vs_outer_iteration.png`, not cosine/rho versions.

- [ ] **Step 5: Run a small real paired experiment file**

Create `/tmp/adp-paired-smoke.py` with `apply_patch` and this exact content:

```python
from ADP import ADP_Config, ADP_Experiment, ADP_ExperimentPoint, ADP_ExperimentVariant

common = ADP_Config(
    N_loc=5,
    N_lin=8,
    N_J=8,
    N_phi=3,
    h_min=1e6,
    index_init="random",
)

experiment = ADP_Experiment(
    name="paired_smoke",
    mode="single",
    runs=2,
    seed=31,
    points=(
        ADP_ExperimentPoint("clean", 40, 3, 0.0, {"sigma_eps": 0.0}),
        ADP_ExperimentPoint("noisy", 40, 3, 0.1, {"sigma_eps": 0.1}),
    ),
    variants={
        "A": ADP_ExperimentVariant(common, "lsmr", {"max_steps": 2}),
        "B": ADP_ExperimentVariant(common, "varpro", {"max_steps": 2}),
    },
)
```

Then run:

```bash
rtk python ADP/cli.py --experiment-file /tmp/adp-paired-smoke.py --output-dir /tmp/adp-unified-paired --no-progress
```

Expected: eight jobs for two points, two seeds, and two variants; `paired_comparison.csv` has four rows; `comparison_summary.csv` has no `winner` column; both variants for each pair reference the same data artifact.

- [ ] **Step 6: Verify resume and reports-only against the newest series**

```bash
rtk python -c 'from pathlib import Path; from ADP.cli import main; root = Path("/tmp/adp-unified-paired/paired_smoke"); series = max((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime_ns); before = {path.name: path.stat().st_mtime_ns for path in (series / "commits").glob("*.json")}; assert main(["--experiment-file", "/tmp/adp-paired-smoke.py", "--resume", str(series), "--no-progress"]) == 0; assert before == {path.name: path.stat().st_mtime_ns for path in (series / "commits").glob("*.json")}; assert main(["--reports-only", str(series)]) == 0; print(series)'
```

Expected: resume performs zero fits and preserves commit mtimes; reports-only recreates `artifacts.csv` and applicable PNGs without importing the experiment file.

- [ ] **Step 7: Inspect persisted invariants programmatically**

```bash
rtk python -c 'import csv; from pathlib import Path; root = Path("/tmp/adp-unified-paired/paired_smoke"); series = max((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime_ns); read = lambda name: list(csv.DictReader((series / name).open(encoding="utf-8", newline=""))); runs = read("run_summary.csv"); pairs = read("paired_comparison.csv"); assert len(runs) == 8; assert len(pairs) == 4; assert "winner" not in pairs[0]; assert all(row["status"] in {"success", "nonconverged", "numerical_failure"} for row in runs); assert all(row["A_seed"] == row["B_seed"] for row in pairs); assert all(Path(row["A_data_artifact"]) == Path(row["B_data_artifact"]) for row in pairs); print(series)'
```

- [ ] **Step 8: Check scope and staged files**

```bash
rtk git diff --check 7c74fb9..HEAD -- ADP
rtk git status --short
rtk proxy git diff -- ADP/single_index/solvers/VarPro.py
```

Expected: no whitespace errors; the user's `VarPro.py` diff remains intact and was not absorbed into any feature commit.

- [ ] **Step 9: Commit verification-only fixes if any**

If verification required an in-scope correction, stage only the corrected feature files and commit:

```bash
rtk git diff --cached --name-only
rtk git commit -m "fix: harden ADP experiment workflow"
```

If no correction was required, do not create an empty commit.
