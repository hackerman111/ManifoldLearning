# ADP Adaptive Conservative Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Добавить opt-in режим `adaptive_conservative`, который ускоряет точные NumPy-статистики, сокращает стоимость beta solve и адаптивно уменьшает `J/P` без падения `cosine_abs` более чем на `0.005` и без ухудшения полного objective более чем на `1%`.

**Architecture:** Строгий путь остаётся неизменным reference/fallback. Отдельные `StatisticsBudget`, `SolverBudget`, `QualityGuard` и incremental statistics workspace управляют вложенными уровнями `L0/L1/L2`; новый PCG stage выполняет inexact ранние solve и строгую финальную коррекцию. Все решения и fallback сохраняются в существующих benchmark CSV.

**Tech Stack:** Python 3.14, NumPy, SciPy, threadpoolctl, psutil, pandas, pytest, существующие `StageRegistry`, `ADPAlgorithm` и `run_benchmarks.py single-index`.

---

## Execution Preconditions

Текущий checkout содержит незакоммиченные изменения в файлах, которые этот
план будет менять. Перед Task 1:

1. сохранить их в owner-approved snapshot или отдельном commit;
2. создать dedicated worktree от этого snapshot;
3. убедиться, что `rtk git status --short` в worktree пуст;
4. не переносить generated benchmark outputs в git;
5. хранить временные before/after artifacts только под `/tmp`.

Не использовать broad `git add`. Каждый commit ниже перечисляет точные файлы.

## File Map

### Новые файлы

- `experiments/benchmark_numpy_statistics.py` — прогретый component benchmark
  точного statistics path.
- `tests/test_statistics_benchmark.py` — контракт JSON component benchmark.
- `adp/solvers/__init__.py` — публичные внутренние exports solver package.
- `adp/solvers/pcg.py` — dtype-stable PCG с recurrence residual trace.
- `adp/solvers/sketched_preconditioner.py` — создаётся только при срабатывании
  измеренного iteration trigger в conditional Task 5A.
- `adp/optimization/__init__.py` — exports adaptive policy package.
- `adp/optimization/solver_budget.py` — выбор early/stable/final solve request.
- `adp/optimization/budget.py` — deterministic `L0/L1/L2`, bucket и profile.
- `adp/optimization/statistics_workspace.py` — incremental rectangular tiles
  `J x P` для random-projection statistics.
- `adp/optimization/guard.py` — единственная реализация runtime guard rules.
- `adp/optimization/controller.py` — каскад accept/escalate/strict fallback.
- `tests/test_pcg.py` — unit tests PCG и matvec accounting.
- `tests/test_adaptive_solver_budget.py` — solver schedule и relative stop.
- `tests/test_adaptive_statistics_budget.py` — nesting, profile и workspace.
- `tests/test_adaptive_quality_guard.py` — каждая причина escalation/fallback.
- `tests/test_adaptive_controller.py` — orchestration без численного fit.
- `experiments/soak_adaptive_performance.py` — bounded long-run resource log.
- `tests/test_adaptive_soak.py` — контракт soak artifact без долгого запуска.

### Изменяемые файлы

- `adp/common/types.py` — config knobs и adaptive telemetry fields.
- `adp/core.py`, `adp/__init__.py` — `ComputeMode` export.
- `adp/backends/numpy_backend.py` — reusable scratch и fused exact arithmetic.
- `adp/variants/random_projection.py` — shared beta system и workspace factory.
- `adp/stages/builtins.py`, `adp/stages/registry.py` — `pcg` и
  `adaptive_convergence` stages.
- `adp/engine/algorithm.py` — adaptive default stage selection, outer branch
  and strict fallback.
- `adp/evaluation/single_index/types.py`, `adp/evaluation/cli.py` — benchmark
  mode/profile arguments.
- `adp/evaluation/single_index/schema.py`,
  `adp/evaluation/single_index/executors.py` — persisted adaptive telemetry.
- `adp/evaluation/single_index/runner.py` — physical-core resource budget.
- `experiments/compare_model_efficiency.py` — paired quality/performance gate.
- Focused existing tests — strict compatibility, stage registration, schema,
  storage и executor persistence.

## Phase A: Reference and Exact Hot Paths

### Task 1: Add a repeatable NumPy statistics benchmark

**Files:**
- Create: `experiments/benchmark_numpy_statistics.py`
- Create: `tests/test_statistics_benchmark.py`

- [ ] **Step 1: Write the benchmark contract test**

```python
from experiments.benchmark_numpy_statistics import StatisticsCase, run_case


def test_statistics_benchmark_returns_reproducible_record():
    record = run_case(
        StatisticsCase("tiny", n=24, d=3, centers=5, directions=2, h_scale=1.0),
        repetitions=2,
        seed=7,
    )

    assert record["name"] == "tiny"
    assert record["shape"] == {"n": 24, "d": 3, "J": 5, "P": 2}
    assert record["repetitions"] == 2
    assert len(record["times_sec"]) == 2
    assert record["median_sec"] >= 0.0
    assert 0.0 <= record["active_fraction"] <= 1.0
    assert record["statistics_shapes"] == {
        "imav": [5, 2],
        "S": [5, 2],
        "U": [5, 2, 3],
        "N": [5],
    }
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
rtk python -m pytest tests/test_statistics_benchmark.py -q
```

Expected: collection fails with `ModuleNotFoundError` for
`experiments.benchmark_numpy_statistics`.

- [ ] **Step 3: Implement the benchmark data path**

Create the following types and `run_case`; the CLI below serializes a
`{"records": [record]}` payload to `--output`.

```python
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from adp import ADP, ADPConfig


@dataclass(frozen=True, slots=True)
class StatisticsCase:
    name: str
    n: int
    d: int
    centers: int
    directions: int
    h_scale: float
    anisotropy: float | None = 0.5


CASES = (
    StatisticsCase("sparse", 1000, 100, 150, 16, 0.75),
    StatisticsCase("primary", 1000, 100, 150, 16, 1.00),
    StatisticsCase("dense", 1000, 100, 150, 16, 1.25),
)


def run_case(case: StatisticsCase, *, repetitions: int, seed: int) -> dict[str, object]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    model = ADP.create(
        "new",
        ADPConfig(
            n_centers=case.centers,
            n_directions=case.directions,
            min_neighbors=4.0,
            center_noise_scale=0.1,
            use_neighbor_index=False,
            statistics_workers=1,
            record_telemetry=True,
            show_progress=False,
            random_state=seed,
        ),
    )
    data = model.generate_data(n=case.n, d=case.d, noise=0.05, link="linear")
    selected_h = model._select_isotropic_bandwidth(data.X, data.centers, None)
    h = float(selected_h * case.h_scale)
    norm2 = model._cached_pairwise_norm2(data.X, data.centers)
    projection2 = (
        None
        if case.anisotropy is None
        else model._cached_pairwise_projection2(data.X, data.centers, data.beta)
    )
    q = model.backend.kernel_argument(
        norm2,
        h=h,
        projection2=projection2,
        anisotropy=case.anisotropy,
    )
    active_fraction = float(np.mean(np.asarray(q) < 1.0))

    result = model._compute_statistics_default(
        data.X,
        data.y,
        data.centers,
        h,
        data.beta,
        data.directions,
        case.anisotropy,
    )
    times: list[float] = []
    for _ in range(repetitions):
        for key in tuple(model._pairwise_cache):
            if key[0] in {"proj2", "proj2_beta"}:
                model._pairwise_cache.pop(key)
        started = time.perf_counter()
        result = model._compute_statistics_default(
            data.X,
            data.y,
            data.centers,
            h,
            data.beta,
            data.directions,
            case.anisotropy,
        )
        times.append(time.perf_counter() - started)
    return {
        "name": case.name,
        "case": asdict(case),
        "shape": {"n": case.n, "d": case.d, "J": case.centers, "P": case.directions},
        "active_fraction": active_fraction,
        "cache_policy": "warm_norm2_cold_projection2",
        "repetitions": repetitions,
        "times_sec": times,
        "median_sec": float(statistics.median(times)),
        "statistics_shapes": {
            "imav": list(result.imav.shape),
            "S": list(np.asarray(result.S).shape),
            "U": list(np.asarray(result.U).shape),
            "N": list(np.asarray(result.N).shape),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("all",) + tuple(item.name for item in CASES), default="all")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = CASES if args.case == "all" else tuple(item for item in CASES if item.name == args.case)
    payload = {"records": [run_case(item, repetitions=args.repetitions, seed=args.seed) for item in selected]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verify GREEN and record the strict baseline**

Run:

```bash
rtk python -m pytest tests/test_statistics_benchmark.py -q
rtk env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/benchmark_numpy_statistics.py --repetitions 7 --output /tmp/adp_stats_before.json
```

Expected: `1 passed`; `/tmp/adp_stats_before.json` contains three records and
finite nonnegative timings.

- [ ] **Step 5: Record an end-to-end experiment-2 baseline**

Run:

```bash
rtk python run_benchmarks.py single-index --profile full --experiments 2 --seeds 0 --diagnostic-seeds 0 --jobs 1 --output /tmp/adp_adaptive_strict_before
```

Expected: completed series with `run_summary.csv`, `outer_iterations.csv`,
`inner_iterations.csv` and `solver_iterations.csv`.

- [ ] **Step 6: Commit the harness**

```bash
rtk git add experiments/benchmark_numpy_statistics.py tests/test_statistics_benchmark.py
rtk git commit -m "test: add ADP statistics performance harness"
```

### Task 2: Add adaptive configuration without changing strict defaults

**Files:**
- Modify: `adp/common/types.py:12-161`
- Modify: `adp/core.py:6-68`
- Modify: `adp/__init__.py:10-40`
- Create: `tests/test_adaptive_config.py`

- [ ] **Step 1: Write failing config tests**

```python
import pytest

from adp import ADP, ADPConfig


def test_adaptive_config_defaults_are_conservative():
    config = ADPConfig(compute_mode="adaptive_conservative")
    assert config.adaptive_start_level == "auto"
    assert config.cg_rtol_final <= config.cg_rtol_stable <= config.cg_rtol_early
    assert config.inner_beta_tol == pytest.approx(1e-4)
    assert config.inner_objective_rtol == pytest.approx(1e-4)
    assert config.audit_fraction == pytest.approx(0.10)
    assert config.audit_min_centers == 4


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("compute_mode", "fast"),
        ("adaptive_start_level", "L3"),
        ("cg_rtol_early", 0.0),
        ("cg_rtol_final", 2.0),
        ("inner_beta_tol", -1.0),
        ("audit_fraction", 0.0),
        ("audit_min_centers", 0),
    ),
)
def test_adaptive_config_rejects_invalid_values(name, value):
    with pytest.raises(ValueError, match=name):
        ADPConfig(**{name: value})


def test_default_model_keeps_strict_stage_names():
    model = ADP.create("new", ADPConfig(show_progress=False))
    assert model.config.compute_mode == "strict"
    assert model.algorithm.stage_names["beta_solver"] == "cg"
    assert model.algorithm.stage_names["stop_rule"] == "convergence"
```

- [ ] **Step 2: Run RED**

```bash
rtk python -m pytest tests/test_adaptive_config.py -q
```

Expected: failures report unknown `ADPConfig` fields.

- [ ] **Step 3: Add typed fields and validation**

Add these aliases and fields:

```python
ComputeMode = Literal["strict", "adaptive_conservative"]
AdaptiveStartLevel = Literal["auto", "L0", "L1", "L2"]

# ADPConfig fields
compute_mode: ComputeMode = "strict"
adaptive_start_level: AdaptiveStartLevel = "auto"
adaptive_profile: str | None = None
cg_rtol_early: float = 1e-3
cg_rtol_stable: float = 1e-4
cg_rtol_final: float = 1e-6
inner_beta_tol: float = 1e-4
inner_objective_rtol: float = 1e-4
audit_fraction: float = 0.10
audit_min_centers: int = 4
adaptive_min_centers: int = 16
```

Add this validation inside `__post_init__`:

```python
if self.compute_mode not in {"strict", "adaptive_conservative"}:
    raise ValueError("compute_mode должен быть 'strict' или 'adaptive_conservative'")
if self.adaptive_start_level not in {"auto", "L0", "L1", "L2"}:
    raise ValueError("adaptive_start_level должен быть auto, L0, L1 или L2")
for name in (
    "cg_rtol_early",
    "cg_rtol_stable",
    "cg_rtol_final",
    "inner_beta_tol",
    "inner_objective_rtol",
):
    value = getattr(self, name)
    if not _is_finite_number(value) or not 0.0 < float(value) <= 1.0:
        raise ValueError(f"{name} должен быть в диапазоне (0, 1]")
if not self.cg_rtol_final <= self.cg_rtol_stable <= self.cg_rtol_early:
    raise ValueError("cg_rtol должны удовлетворять final <= stable <= early")
if not _is_finite_number(self.audit_fraction) or not 0.0 < self.audit_fraction < 0.5:
    raise ValueError("audit_fraction должен быть в диапазоне (0, 0.5)")
for name in ("audit_min_centers", "adaptive_min_centers"):
    value = getattr(self, name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} должен быть положительным целым")
```

Export `ComputeMode` through `adp/core.py` and `adp/__init__.py`.

- [ ] **Step 4: Verify strict compatibility**

```bash
rtk python -m pytest tests/test_adaptive_config.py tests/test_stage_factories.py -q
```

Expected: all tests pass and default stages remain `cg`/`convergence`.

- [ ] **Step 5: Commit**

```bash
rtk git add adp/common/types.py adp/core.py adp/__init__.py tests/test_adaptive_config.py
rtk git commit -m "feat: add conservative adaptive ADP configuration"
```

### Task 3: Reuse exact NumPy statistics scratch buffers

**Files:**
- Modify: `adp/backends/numpy_backend.py:15-430`
- Modify: `tests/test_performance_optimizations.py:159-311,660-700,930-975`

- [ ] **Step 1: Add allocation-reuse and numerical tests**

Append a test that monkeypatches `_StatisticsScratch.__init__`, calls a serial
compact-kernel block with four centers, and asserts one scratch allocation.
Reuse `reference_random_projection_sums` for exact values:

```python
def test_serial_statistics_reuses_one_scratch_buffer(monkeypatch):
    allocations = 0
    original = numpy_backend._StatisticsScratch.__init__

    def counted(self, *args, **kwargs):
        nonlocal allocations
        allocations += 1
        original(self, *args, **kwargs)

    monkeypatch.setattr(numpy_backend._StatisticsScratch, "__init__", counted)
    rng = np.random.default_rng(4)
    X = rng.normal(size=(32, 5))
    y = rng.normal(size=32)
    centers = rng.normal(size=(4, 5))
    directions = rng.normal(size=(4, 3, 5))
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    q = pairwise_norm2(X, centers) / 9.0

    actual = NumpyBackend().random_projection_sums(
        X=X, y=y, centers=centers, directions=directions, q=q,
        kernel="epanechnikov",
    )
    expected = reference_random_projection_sums(
        X, y, centers, directions, q, "epanechnikov"
    )

    assert allocations == 1
    for actual_part, expected_part in zip(actual, expected):
        np.testing.assert_allclose(actual_part, expected_part, rtol=1e-11, atol=1e-12)
```

- [ ] **Step 2: Run RED**

```bash
rtk python -m pytest tests/test_performance_optimizations.py::test_serial_statistics_reuses_one_scratch_buffer -q
```

Expected: `_StatisticsScratch` is not defined.

- [ ] **Step 3: Add one reusable scratch object per worker**

Add this private class near the backend constants:

```python
class _StatisticsScratch:
    def __init__(self, n: int, d: int, p: int, dtype: np.dtype) -> None:
        self.active_x = np.empty((n, d), dtype=dtype)
        self.active_y = np.empty(n, dtype=dtype)
        self.differences = np.empty((n, d), dtype=dtype)
        self.projected = np.empty((n, p), dtype=dtype)
        self.weights = np.empty(n, dtype=dtype)

    def load(self, x, y, q_row, active, center, directions, kernel):
        indices = np.flatnonzero(active)
        count = int(indices.size)
        np.take(x, indices, axis=0, out=self.active_x[:count])
        np.take(y, indices, out=self.active_y[:count])
        np.take(q_row, indices, out=self.weights[:count])
        if kernel == "gaussian":
            np.multiply(self.weights[:count], -0.5, out=self.weights[:count])
            np.exp(self.weights[:count], out=self.weights[:count])
        else:
            np.subtract(1.0, self.weights[:count], out=self.weights[:count])
            np.maximum(self.weights[:count], 0.0, out=self.weights[:count])
            if kernel == "quartic":
                np.square(self.weights[:count], out=self.weights[:count])
        np.subtract(self.active_x[:count], center, out=self.differences[:count])
        np.matmul(
            self.differences[:count],
            directions.T,
            out=self.projected[:count],
        )
        np.multiply(
            self.projected[:count],
            self.weights[:count, None],
            out=self.projected[:count],
        )
        return count
```

In `_centered_random_projection_sums`, create one serial scratch. For the
threaded path, use `threading.local()` and lazily create one scratch per worker.
Replace per-center `x_active`, `y_active`, `differences`, `projected` and
`weights` allocations with slices from `scratch`.

- [ ] **Step 4: Run correctness and thread-lifecycle tests**

```bash
rtk python -m pytest tests/test_performance_optimizations.py -q
```

Expected: all tests pass for compact, Gaussian, float32 and multi-worker paths.

- [ ] **Step 5: Measure the exact gate**

```bash
rtk env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/benchmark_numpy_statistics.py --repetitions 7 --output /tmp/adp_stats_scratch.json
```

Expected: no case regresses by more than `5%`. Keep this commit only if the
primary median improves; later adaptive work remains independent of whether
this step alone reaches `1.5x`.

- [ ] **Step 6: Commit**

```bash
rtk git add adp/backends/numpy_backend.py tests/test_performance_optimizations.py
rtk git commit -m "perf: reuse NumPy statistics work buffers"
```

## Phase B: PCG and Adaptive Inner Solve

### Task 4: Implement a dtype-stable PCG kernel

**Files:**
- Create: `adp/solvers/__init__.py`
- Create: `adp/solvers/pcg.py`
- Create: `tests/test_pcg.py`

- [ ] **Step 1: Write PCG behavior tests**

```python
import numpy as np

from adp.solvers.pcg import solve_pcg


def test_pcg_matches_dense_spd_solution_and_preserves_dtype():
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    rhs = np.array([1.0, 2.0], dtype=np.float32)
    result = solve_pcg(
        lambda vector: matrix @ vector,
        rhs,
        x0=np.zeros(2, dtype=np.float32),
        precondition=lambda vector: vector / np.diag(matrix),
        rtol=1e-6,
        maxiter=20,
        record_trace=True,
    )
    np.testing.assert_allclose(result.x, np.linalg.solve(matrix, rhs), rtol=2e-6, atol=2e-6)
    assert result.x.dtype == np.float32
    assert result.status == "converged"
    assert result.iterations == len(result.residual_trace)


def test_pcg_trace_does_not_add_one_matvec_per_iteration():
    calls = 0
    matrix = np.diag(np.arange(1.0, 9.0))

    def matvec(vector):
        nonlocal calls
        calls += 1
        return matrix @ vector

    result = solve_pcg(
        matvec,
        np.ones(8),
        x0=np.zeros(8),
        precondition=None,
        rtol=1e-12,
        maxiter=20,
        record_trace=True,
    )
    assert calls <= result.iterations + 2


def test_pcg_reports_non_spd_breakdown():
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = solve_pcg(
        lambda vector: matrix @ vector,
        np.array([1.0, -1.0]),
        x0=np.zeros(2),
        precondition=None,
        rtol=1e-6,
        maxiter=5,
        record_trace=False,
    )
    assert result.status == "breakdown"
```

- [ ] **Step 2: Run RED**

```bash
rtk python -m pytest tests/test_pcg.py -q
```

Expected: import failure for `adp.solvers.pcg`.

- [ ] **Step 3: Implement PCG with recurrence residuals**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from adp.common.utils import stable_l2_norm

PCGStatus = Literal[
    "converged",
    "max_iterations",
    "breakdown",
    "invalid_system",
    "residual_mismatch",
]


@dataclass(frozen=True, slots=True)
class PCGResult:
    x: np.ndarray
    status: PCGStatus
    iterations: int
    relative_residual: float
    residual_trace: tuple[float, ...]


def solve_pcg(
    matvec: Callable[[np.ndarray], np.ndarray],
    rhs: np.ndarray,
    *,
    x0: np.ndarray,
    precondition: Callable[[np.ndarray], np.ndarray] | None,
    rtol: float,
    maxiter: int,
    record_trace: bool,
) -> PCGResult:
    b = np.asarray(rhs)
    x = np.asarray(x0, dtype=b.dtype).copy()
    scale = max(stable_l2_norm(b), float(np.finfo(b.dtype).eps))
    residual = b - np.asarray(matvec(x), dtype=b.dtype)
    if not np.all(np.isfinite(residual)):
        return PCGResult(x, "invalid_system", 0, float("nan"), ())
    trace: list[float] = []

    def finish(status: PCGStatus, iterations: int) -> PCGResult:
        exact_residual = b - np.asarray(matvec(x), dtype=b.dtype)
        exact_relative = stable_l2_norm(exact_residual) / scale
        final_status = status
        if not np.isfinite(exact_relative):
            final_status = "invalid_system"
        elif status == "converged" and exact_relative > rtol:
            final_status = "residual_mismatch"
        return PCGResult(x, final_status, iterations, exact_relative, tuple(trace))

    relative = stable_l2_norm(residual) / scale
    if relative <= rtol:
        return finish("converged", 0)
    z = residual.copy() if precondition is None else np.asarray(precondition(residual), dtype=b.dtype)
    direction = z.copy()
    rz_old = float(np.dot(residual, z))
    if not np.isfinite(rz_old) or rz_old <= 0.0:
        return finish("breakdown", 0)
    for iteration in range(1, maxiter + 1):
        operator_direction = np.asarray(matvec(direction), dtype=b.dtype)
        denominator = float(np.dot(direction, operator_direction))
        if not np.isfinite(denominator) or denominator <= 0.0 or not np.isfinite(rz_old):
            return finish("breakdown", iteration - 1)
        alpha = rz_old / denominator
        x += alpha * direction
        residual -= alpha * operator_direction
        relative = stable_l2_norm(residual) / scale
        if record_trace:
            trace.append(relative)
        if relative <= rtol:
            return finish("converged", iteration)
        z = residual.copy() if precondition is None else np.asarray(precondition(residual), dtype=b.dtype)
        rz_new = float(np.dot(residual, z))
        if not np.isfinite(rz_new) or rz_new <= 0.0:
            return finish("breakdown", iteration)
        direction *= rz_new / rz_old
        direction += z
        rz_old = rz_new
    return finish("max_iterations", maxiter)
```

Export `PCGResult`, `PCGStatus` and `solve_pcg` from
`adp/solvers/__init__.py` with an explicit `__all__`.

- [ ] **Step 4: Verify PCG tests**

```bash
rtk python -m pytest tests/test_pcg.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
rtk git add adp/solvers/__init__.py adp/solvers/pcg.py tests/test_pcg.py
rtk git commit -m "feat: add recurrence-traced PCG solver"
```

### Task 5: Integrate PCG as a separate beta-solver stage

**Files:**
- Modify: `adp/variants/random_projection.py:250-410`
- Modify: `adp/stages/builtins.py:97-169`
- Modify: `adp/stages/registry.py:19-56`
- Modify: `tests/test_stage_factories.py`
- Modify: `tests/test_solver_objective_consistency.py`
- Modify: `tests/test_performance_optimizations.py:977-1024`

- [ ] **Step 1: Add stage and numerical-equivalence tests**

```python
from adp.common.types import LocalStatistics


def _small_spd_statistics():
    return LocalStatistics(
        variant="new",
        imav=np.array([[1.0, 0.2], [0.3, -0.1]]),
        centers=np.zeros((2, 3)),
        h=1.0,
        weights_mean=4.0,
        S=np.array([[0.2, -0.1], [0.1, 0.3]]),
        U=np.array(
            [
                [[1.0, 0.2, 0.0], [0.1, 0.8, 0.3]],
                [[0.4, 0.0, 0.7], [0.3, 0.5, 0.2]],
            ]
        ),
    )


def _direct_beta_solution(model, stats, intercepts, slopes, prior, penalty):
    residual = stats.imav - intercepts[:, None] * stats.S
    u_flat = stats.U.reshape(-1, stats.U.shape[-1])
    slope_flat = np.broadcast_to(slopes[:, None], stats.imav.shape).reshape(-1)
    lhs = u_flat.T @ ((slope_flat * slope_flat)[:, None] * u_flat)
    lhs += (penalty + model.config.ridge) * np.eye(u_flat.shape[1])
    rhs = u_flat.T @ (slope_flat * residual.reshape(-1)) + penalty * prior
    return np.linalg.solve(lhs, rhs)


def test_builtin_registry_exposes_pcg_without_changing_default():
    registry = StageRegistry.with_defaults()
    assert {"cg", "pcg"} <= set(registry.available("beta_solver"))
    assert DEFAULT_STAGE_NAMES["beta_solver"] == "cg"


def test_pcg_beta_stage_matches_direct_solution():
    stats = _small_spd_statistics()
    model = ADP.create(
        "new",
        ADPConfig(tol=1e-10, show_progress=False),
        stages={"beta_solver": "pcg"},
    )
    intercepts = np.zeros(stats.imav.shape[0])
    slopes = np.ones(stats.imav.shape[0])
    prior = np.array([1.0, 0.0, 0.0])
    actual = model.algorithm.components["beta_solver"].solve(
        stats, intercepts, slopes, prior, 0.2, x0=prior
    )
    expected = _direct_beta_solution(model, stats, intercepts, slopes, prior, 0.2)
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)
```

Keep the current float32 consistency test for both stages.

- [ ] **Step 2: Run RED**

```bash
rtk python -m pytest tests/test_stage_factories.py tests/test_solver_objective_consistency.py -q
```

Expected: registry rejects unknown `pcg`.

- [ ] **Step 3: Share one beta linear-system builder**

Refactor `_solve_beta_default` so SciPy CG and PCG use the same `u_flat`, RHS,
regularization and Jacobi diagonal. Add `_build_beta_linear_system` returning
these callables and arrays:

```python
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True)
class _BetaLinearSystem:
    rhs: np.ndarray
    matvec: Callable[[np.ndarray], np.ndarray]
    precondition: Callable[[np.ndarray], np.ndarray] | None
    residual_scale: float
    dtype: np.dtype
```

Build `slope_sq` with broadcasting and reshape, not `np.repeat`:

```python
slope_sq = np.square(np.asarray(slopes, dtype=solver_dtype))[:, None]
slope_sq_grid = np.broadcast_to(slope_sq, residual.shape)
weighted_u = u_values * slope_sq_grid[:, :, None]
rhs = np.einsum(
    "jpd,jp->d",
    u_values,
    np.asarray(slopes, dtype=solver_dtype)[:, None] * residual,
    optimize=True,
)
rhs += lambda_value * prior_values
diagonal = np.einsum(
    "jpd,jpd,jp->d",
    u_values,
    u_values,
    slope_sq_grid,
    optimize=True,
)
diagonal += regularization
```

The operator remains matrix-free and adds regularization exactly once.

- [ ] **Step 4: Add `PCGBetaSolver`**

`PCGBetaSolver.solve` calls `model._solve_beta_pcg` with
`rtol=config.cg_rtol_final` and the final cap
`max(50, min(500, 5 * d))`. Add a second method:

```python
def solve_with_request(
    self,
    statistics,
    intercepts,
    slopes,
    prior,
    lambda_penalty,
    *,
    x0,
    request,
):
    return self.model._solve_beta_pcg(
        statistics,
        intercepts,
        slopes,
        prior,
        lambda_penalty,
        x0=x0,
        rtol=request.rtol,
        maxiter=request.maxiter,
        phase=request.phase,
    )
```

Register `pcg` as an extra builtin while keeping `cg` as the default. Store
the exact final residual in `_last_solver_telemetry`; `solver_residual_trace`
uses the recurrence trace and never calls `matvec` itself.

`StageRegistry.with_defaults()` must register every implementation from
`BUILTIN_STAGE_TYPES`, not only the names in `DEFAULT_STAGE_NAMES`:

```python
@classmethod
def with_defaults(cls) -> "StageRegistry":
    from .builtins import BUILTIN_STAGE_TYPES

    registry = cls()
    for category, implementations in BUILTIN_STAGE_TYPES.items():
        for name in implementations:
            registry.register(
                category,
                name,
                _deferred_builtin_factory(category, name),
            )
    return registry
```

Keep `DEFAULT_STAGE_NAMES["beta_solver"] == "cg"`; registration availability
and default selection are separate contracts.

- [ ] **Step 5: Run focused solver tests**

```bash
rtk python -m pytest tests/test_pcg.py tests/test_stage_factories.py tests/test_solver_edge_cases.py tests/test_solver_objective_consistency.py tests/test_performance_optimizations.py -q
```

Expected: all tests pass; diagnostic trace length equals PCG iterations and
does not double matvec count.

- [ ] **Step 6: Commit**

```bash
rtk git add adp/variants/random_projection.py adp/stages/builtins.py adp/stages/registry.py tests/test_stage_factories.py tests/test_solver_objective_consistency.py tests/test_performance_optimizations.py
rtk git commit -m "feat: expose PCG beta solver stage"
```

### Conditional Task 5A: Add a deterministic sketched preconditioner

Execute this task only when representative PCG telemetry shows median
iterations greater than `max(10, ceil(d / 8))`. Otherwise record the measured
iteration counts in the Task 5 handoff and continue to Task 6 without creating
the file.

**Files:**
- Create: `adp/solvers/sketched_preconditioner.py`
- Create: `tests/test_sketched_preconditioner.py`
- Modify: `adp/variants/random_projection.py`

- [ ] **Step 1: Write SPD and iteration-count tests**

Construct a deterministic ill-conditioned `u_flat`, build Jacobi and sketched
preconditioners, and assert:

```python
sketch = SketchedPreconditioner.build(
    u_flat,
    slope_sq_flat,
    regularization=1e-3,
    seed=11,
    sample_rows=min(u_flat.shape[0], max(64, 2 * u_flat.shape[1])),
)
probe = np.linspace(0.1, 1.0, u_flat.shape[1])
assert np.all(np.isfinite(sketch(probe)))
assert float(probe @ sketch(probe)) > 0.0
assert sketched_result.iterations <= jacobi_result.iterations
```

- [ ] **Step 2: Run RED**

```bash
rtk python -m pytest tests/test_sketched_preconditioner.py -q
```

Expected: module import fails.

- [ ] **Step 3: Implement the deterministic SPD approximation**

Form rows as `sqrt(slope_sq_flat)[:, None] * u_flat`. Sample them without
replacement using `np.random.default_rng(seed)`, with
`sample_rows <= u_flat.shape[0]`, scale the sample Gram matrix by
`m / sample_rows`, add `regularization * I`, and store its Cholesky factor.
Apply the preconditioner with two triangular solves. Reuse it while relative
slope change is below `0.10`; rebuild after that threshold.

- [ ] **Step 4: Verify numerical and timing gates**

```bash
rtk python -m pytest tests/test_sketched_preconditioner.py tests/test_pcg.py tests/test_solver_objective_consistency.py -q
```

Expected: all pass. Keep the preconditioner only when representative median
beta-solver time improves by at least `10%`; otherwise remove both new files
and continue with Jacobi.

- [ ] **Step 5: Commit only when accepted**

```bash
rtk git add adp/solvers/sketched_preconditioner.py adp/variants/random_projection.py tests/test_sketched_preconditioner.py
rtk git commit -m "perf: add gated sketched PCG preconditioner"
```

### Task 6: Add SolverBudget and relative adaptive convergence

**Files:**
- Create: `adp/optimization/__init__.py`
- Create: `adp/optimization/solver_budget.py`
- Create: `tests/test_adaptive_solver_budget.py`
- Modify: `adp/stages/builtins.py:120-155`
- Modify: `adp/stages/registry.py`

- [ ] **Step 1: Write solver schedule tests**

```python
from types import SimpleNamespace

import numpy as np

from adp import ADPConfig
from adp.optimization.solver_budget import SolverBudget
from adp.stages.builtins import AdaptiveConvergenceStopRule
from adp.stages.contracts import ADPState


def _adaptive_rule():
    context = SimpleNamespace(
        config=ADPConfig(compute_mode="adaptive_conservative")
    )
    return AdaptiveConvergenceStopRule(context)


def _state():
    return ADPState(X=np.empty((0, 1)), y=np.empty(0))


def test_solver_budget_uses_early_stable_and_final_requests():
    budget = SolverBudget(ADPConfig(compute_mode="adaptive_conservative"))
    assert budget.request(d=100, beta_delta=1.0).phase == "early"
    stable = budget.request(d=100, beta_delta=9e-3)
    assert stable.phase == "stable"
    assert stable.rtol == 1e-4
    final = budget.final_request(d=100)
    assert final.phase == "final"
    assert final.rtol == 1e-6
    assert final.maxiter == 500


def test_adaptive_stop_requires_two_consecutive_complete_checks():
    rule = _adaptive_rule()
    metrics = {
        "outer": 0,
        "beta_delta": 5e-5,
        "relative_objective_change": 5e-5,
        "relative_linear_residual": 5e-7,
        "requested_rtol": 1e-6,
        "linear_solver_status": "converged",
        "audit_ok": True,
    }
    assert not rule.should_stop("inner", _state(), **metrics)
    assert rule.should_stop("inner", _state(), **metrics)
```

- [ ] **Step 2: Run RED**

```bash
rtk python -m pytest tests/test_adaptive_solver_budget.py -q
```

Expected: imports or `adaptive_convergence` stage are missing.

- [ ] **Step 3: Implement `SolverRequest` and `SolverBudget`**

```python
from dataclasses import dataclass
from typing import Literal

from adp.common.types import ADPConfig


@dataclass(frozen=True, slots=True)
class SolverRequest:
    phase: Literal["early", "stable", "final"]
    rtol: float
    maxiter: int


class SolverBudget:
    def __init__(self, config: ADPConfig) -> None:
        self.config = config

    def request(self, *, d: int, beta_delta: float) -> SolverRequest:
        if beta_delta < 1e-2:
            return SolverRequest("stable", self.config.cg_rtol_stable, max(30, min(250, 3 * d)))
        return SolverRequest("early", self.config.cg_rtol_early, max(20, min(100, 2 * d)))

    def final_request(self, *, d: int) -> SolverRequest:
        return SolverRequest("final", self.config.cg_rtol_final, max(50, min(500, 5 * d)))
```

Export `SolverRequest` and `SolverBudget` from
`adp/optimization/__init__.py`; later tasks extend the same explicit
`__all__` instead of replacing it.

- [ ] **Step 4: Add `AdaptiveConvergenceStopRule`**

```python
class AdaptiveConvergenceStopRule:
    def __init__(self, context: StageContext) -> None:
        self.config = context.config
        self._outer: int | None = None
        self._consecutive = 0

    def should_stop(self, phase: str, state: ADPState, *, step=None, **metrics):
        if phase == "outer":
            anisotropy = state.anisotropy
            return (
                self.config.anisotropy_min is not None
                and anisotropy is not None
                and float(anisotropy) <= self.config.anisotropy_min
            )
        if phase != "inner":
            raise ValueError("phase должен быть 'inner' или 'outer'")
        outer = int(metrics.get("outer", -1))
        if outer != self._outer:
            self._outer = outer
            self._consecutive = 0
        successful = (
            float(metrics.get("beta_delta", math.inf)) <= self.config.inner_beta_tol
            and float(metrics.get("relative_objective_change", math.inf))
            <= self.config.inner_objective_rtol
            and str(metrics.get("linear_solver_status", "")) == "converged"
            and float(metrics.get("relative_linear_residual", math.inf))
            <= float(metrics.get("requested_rtol", 0.0))
            and bool(metrics.get("audit_ok", False))
        )
        self._consecutive = self._consecutive + 1 if successful else 0
        return self._consecutive >= 2
```

Register it as `adaptive_convergence`; keep `convergence` as the strict default.

- [ ] **Step 5: Verify focused tests**

```bash
rtk python -m pytest tests/test_adaptive_solver_budget.py tests/test_stage_factories.py tests/test_solver_objective_consistency.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
rtk git add adp/optimization/__init__.py adp/optimization/solver_budget.py adp/stages/builtins.py adp/stages/registry.py tests/test_adaptive_solver_budget.py
rtk git commit -m "feat: add adaptive solver schedule and stop rule"
```

## Phase C: Adaptive Statistics and Runtime Guard

### Task 7: Implement deterministic StatisticsBudget and calibration profile

**Files:**
- Create: `adp/optimization/budget.py`
- Create: `tests/test_adaptive_statistics_budget.py`
- Modify: `adp/optimization/__init__.py`

- [ ] **Step 1: Write deterministic nesting tests**

```python
import json

import numpy as np

from adp import ADPConfig
from adp.optimization.budget import CalibrationBucket, CalibrationProfile, StatisticsBudget


def _bucket():
    return CalibrationBucket(
        d_band="51-100",
        ratio_band="6-10",
        active_band="0.50-0.75",
        kernel="epanechnikov",
        dtype="float64",
    )


def test_budget_levels_are_nested_and_unknown_bucket_starts_at_l1(tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({"schema_version": 1, "status": "approved", "l0_buckets": []}))
    budget = StatisticsBudget(
        total_centers=100,
        total_directions=32,
        seed=17,
        config=ADPConfig(
            compute_mode="adaptive_conservative",
            adaptive_profile=str(profile_path),
        ),
    )
    l1 = budget.selection("L1")
    l2 = budget.selection("L2")
    assert budget.start_level(_bucket()) == "L1"
    assert set(l1.fit_centers) < set(l2.fit_centers)
    assert set(l1.directions) < set(l2.directions)
    assert set(l1.audit_centers).isdisjoint(l1.fit_centers)


def test_small_center_count_forces_l2():
    budget = StatisticsBudget(
        total_centers=8,
        total_directions=4,
        seed=3,
        config=ADPConfig(compute_mode="adaptive_conservative"),
    )
    assert budget.start_level(_bucket()) == "L2"
```

Add a parametrized boundary test for `d={10,25,50,100}`, ratios
`{2,6,10}`, active fractions `{0.25,0.50,0.75}`, dtype normalization and
rejection of nonfinite observables. Bucket labels are part of the persisted
profile contract and must not depend on scenario names.

- [ ] **Step 2: Run RED**

```bash
rtk python -m pytest tests/test_adaptive_statistics_budget.py -q
```

Expected: `adp.optimization.budget` is missing.

- [ ] **Step 3: Implement immutable budget types**

```python
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


BudgetName = Literal["L0", "L1", "L2"]


@dataclass(frozen=True, slots=True)
class BudgetSelection:
    name: BudgetName
    fit_centers: tuple[int, ...]
    audit_centers: tuple[int, ...]
    directions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CalibrationBucket:
    d_band: str
    ratio_band: str
    active_band: str
    kernel: str
    dtype: str

    def key(self) -> str:
        return "|".join((self.d_band, self.ratio_band, self.active_band, self.kernel, self.dtype))

    @classmethod
    def from_observables(cls, *, d, n_over_d, active_fraction, kernel, dtype):
        if d < 1 or not math.isfinite(n_over_d) or n_over_d <= 0.0:
            raise ValueError("invalid adaptive bucket dimensions")
        if not math.isfinite(active_fraction) or not 0.0 <= active_fraction <= 1.0:
            raise ValueError("invalid adaptive active fraction")
        d_band = (
            "1-10" if d <= 10 else
            "11-25" if d <= 25 else
            "26-50" if d <= 50 else
            "51-100" if d <= 100 else
            "101+"
        )
        ratio_band = (
            "0-2" if n_over_d <= 2.0 else
            "2-6" if n_over_d <= 6.0 else
            "6-10" if n_over_d <= 10.0 else
            "10+"
        )
        active_band = (
            "0-0.25" if active_fraction <= 0.25 else
            "0.25-0.50" if active_fraction <= 0.50 else
            "0.50-0.75" if active_fraction <= 0.75 else
            "0.75-1.00"
        )
        return cls(d_band, ratio_band, active_band, str(kernel), np.dtype(dtype).name)


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    l0_buckets: frozenset[str]
    chunk_sizes: dict[str, int]
    float32_buckets: frozenset[str]

    @classmethod
    def empty(cls):
        return cls(frozenset(), {}, frozenset())

    @classmethod
    def load(cls, path: str | None):
        if path is None:
            return cls.empty()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("status") != "approved":
            raise ValueError("adaptive profile must be approved schema version 1")
        chunks = {str(key): int(value) for key, value in payload.get("chunk_sizes", {}).items()}
        if any(value not in {8, 16, 32, 64} for value in chunks.values()):
            raise ValueError("adaptive profile contains invalid chunk size")
        return cls(
            frozenset(str(value) for value in payload.get("l0_buckets", ())),
            chunks,
            frozenset(str(value) for value in payload.get("float32_buckets", ())),
        )

    def allows_l0(self, bucket: CalibrationBucket) -> bool:
        return bucket.key() in self.l0_buckets

    def chunk_size_for(self, bucket: CalibrationBucket, default: int) -> int:
        return self.chunk_sizes.get(bucket.key(), default)

    def allows_float32(self, bucket: CalibrationBucket) -> bool:
        return bucket.key() in self.float32_buckets


class StatisticsBudget:
    _fractions = {"L0": 0.50, "L1": 0.75, "L2": 1.00}

    def __init__(self, *, total_centers, total_directions, seed, config):
        self.config = config
        self.profile = CalibrationProfile.load(config.adaptive_profile)
        rng = np.random.default_rng(seed)
        order = tuple(int(value) for value in rng.permutation(total_centers))
        audit_count = min(
            total_centers,
            max(config.audit_min_centers, math.ceil(config.audit_fraction * total_centers)),
        )
        self.fit_order = order[:-audit_count]
        self.audit_order = order[-audit_count:]
        self.all_order = self.fit_order + self.audit_order
        self.direction_order = tuple(
            int(value) for value in rng.permutation(total_directions)
        )
        self.total_centers = total_centers
        self.total_directions = total_directions

    def start_level(self, bucket: CalibrationBucket) -> BudgetName:
        if self.total_centers < self.config.adaptive_min_centers:
            return "L2"
        if self.config.adaptive_start_level != "auto":
            return self.config.adaptive_start_level
        return "L0" if self.profile.allows_l0(bucket) else "L1"

    def selection(self, name: BudgetName) -> BudgetSelection:
        if name == "L2":
            return BudgetSelection(name, self.all_order, (), self.direction_order)
        fraction = self._fractions[name]
        fit_count = min(len(self.fit_order), math.ceil(fraction * self.total_centers))
        direction_count = max(1, math.ceil(fraction * self.total_directions))
        return BudgetSelection(
            name,
            self.fit_order[:fit_count],
            self.audit_order,
            self.direction_order[:direction_count],
        )
```

`CalibrationProfile.load` accepts only JSON with `schema_version == 1` and
`status == "approved"`; malformed or unapproved files raise `ValueError`.
It also exposes `chunk_size_for(bucket, default)` from a validated
`chunk_sizes: dict[str, int]` and `allows_float32(bucket)` from
`float32_buckets: list[str]`. Chunk sizes must be positive integers from
`{8, 16, 32, 64}`.

Use a seed-dependent center permutation and direction permutation. Reserve
`max(audit_min_centers, ceil(audit_fraction * J))` audit centers from the end.
`L0/L1` fit counts are `ceil(0.50 * J)` and `ceil(0.75 * J)`; `L2` contains all
centers and has an empty audit set. If `J < adaptive_min_centers`, force `L2`.

`adaptive_start_level="L0"` is a calibration/test override used only by Task
13. The public benchmark CLI and README expose production `auto`, under which
an unknown bucket can never start below `L1`. If the override is used, record
`forced_unapproved_L0=true` in adaptive telemetry so such runs cannot be
mistaken for policy-approved production evidence.

Export `BudgetName`, `BudgetSelection`, `CalibrationBucket`,
`CalibrationProfile` and `StatisticsBudget` from
`adp/optimization/__init__.py`.

- [ ] **Step 4: Verify profile validation and nesting**

```bash
rtk python -m pytest tests/test_adaptive_statistics_budget.py -q
```

Expected: all tests pass, including deterministic repeat with the same seed.

- [ ] **Step 5: Commit**

```bash
rtk git add adp/optimization/budget.py adp/optimization/__init__.py tests/test_adaptive_statistics_budget.py
rtk git commit -m "feat: add nested ADP statistics budgets"
```

### Task 8: Add incremental random-projection statistics workspace

**Files:**
- Create: `adp/optimization/statistics_workspace.py`
- Modify: `adp/optimization/__init__.py`
- Modify: `adp/variants/random_projection.py:20-182`
- Modify: `adp/stages/builtins.py:70-87`
- Modify: `tests/test_adaptive_statistics_budget.py`

- [ ] **Step 1: Add a missing-tile reuse test**

Build deterministic `X/y/centers/directions`, then compare these paths:

```python
workspace = model._create_statistics_workspace_default(
    X, y, centers, h, beta, directions, None
)
l1 = workspace.materialize(budget.selection("L1"))
computed_after_l1 = workspace.computed_tile_count
l2 = workspace.materialize(budget.selection("L2"))

strict = model._compute_statistics_default(X, y, centers, h, beta, directions, None)
assert workspace.reused_tile_count > 0
assert workspace.computed_tile_count > computed_after_l1
np.testing.assert_allclose(l2.imav, strict.imav, rtol=1e-11, atol=1e-12)
np.testing.assert_allclose(l2.S, strict.S, rtol=1e-11, atol=1e-12)
np.testing.assert_allclose(l2.U, strict.U, rtol=1e-11, atol=1e-12)
np.testing.assert_allclose(l2.N, strict.N, rtol=1e-11, atol=1e-12)
```

- [ ] **Step 2: Run RED**

```bash
rtk python -m pytest tests/test_adaptive_statistics_budget.py -q
```

Expected: model has no workspace factory.

- [ ] **Step 3: Implement one outer-scoped workspace**

The constructor computes full `norm2`, optional `projection2`, one full `q`,
and preallocates `imav`, `S`, `U`, `N` plus a `J x P` boolean mask. The core
write method is:

```python
def _compute_tile(self, center_positions: np.ndarray, direction_positions: np.ndarray) -> None:
    if center_positions.size == 0 or direction_positions.size == 0:
        return
    missing = ~self.computed[np.ix_(center_positions, direction_positions)]
    if not np.any(missing):
        self.reused_tile_count += int(missing.size)
        return
    if not np.all(missing):
        raise RuntimeError("adaptive statistics tiles must be rectangular and nested")
    center_index = center_positions[:, None]
    direction_index = direction_positions[None, :]
    chunk = self.backend.random_projection_sums(
        X=self.X,
        y=self.y,
        centers=self.centers[center_positions],
        directions=self.directions[center_index, direction_index, :],
        q=self.q[center_positions],
        kernel=self.kernel,
        record_telemetry=self.record_telemetry,
    )
    self.imav[np.ix_(center_positions, direction_positions)] = chunk[0]
    self.S[np.ix_(center_positions, direction_positions)] = chunk[1]
    self.U[center_index, direction_index, :] = chunk[2]
    unset_n = ~self.counts_computed[center_positions]
    if np.any(unset_n):
        rows = center_positions[unset_n]
        self.N[rows] = np.asarray(chunk[3])[unset_n]
        self.counts_computed[rows] = True
    self.computed[np.ix_(center_positions, direction_positions)] = True
    self.computed_tile_count += int(missing.size)
```

`materialize` groups rows by their current direction prefix, computes only the
missing suffix, and returns a compact `LocalStatistics` for fit centers. Add
`materialize_audit` for the reserved centers. Both preserve the original
center and direction order in returned arrays: sort selected global indices
before slicing the buffers, even though the budget permutations determine
membership. This is required for assembled `L2` to compare elementwise with
the strict result.

Split center positions into the calibrated chunk size returned by
`CalibrationProfile.chunk_size_for`. When no profile entry exists, retain
`config.chunk_size`; never run an online autotuning loop inside `fit`. Because
the active-fraction bucket is known only after full `q` exists, construct the
workspace with the config default, derive the bucket from `q`, then call a
validated `workspace.set_chunk_size(...)` before computing the first tile.

Export the workspace type from `adp/optimization/__init__.py`.

- [ ] **Step 4: Expose workspace only on the builtin statistics stage**

Add `RandomProjectionStatisticsBuilder.create_workspace` delegating to
`model._create_statistics_workspace_default`. Custom statistics stages still
need only `compute`; the adaptive controller will strict-fallback if
`create_workspace` is absent.

- [ ] **Step 5: Verify exact L2 and no duplicate tiles**

```bash
rtk python -m pytest tests/test_adaptive_statistics_budget.py tests/test_performance_optimizations.py tests/test_stage_factories.py -q
```

Expected: all pass; assembled L2 equals strict full statistics.

- [ ] **Step 6: Commit**

```bash
rtk git add adp/optimization/statistics_workspace.py adp/optimization/__init__.py adp/variants/random_projection.py adp/stages/builtins.py tests/test_adaptive_statistics_budget.py
rtk git commit -m "feat: reuse nested ADP statistics tiles"
```

### Task 9: Implement QualityGuard with explicit failure reasons

**Files:**
- Create: `adp/optimization/guard.py`
- Create: `tests/test_adaptive_quality_guard.py`
- Modify: `adp/optimization/__init__.py`

- [ ] **Step 1: Write one table-driven guard test**

```python
from dataclasses import replace

import pytest

from adp.optimization.guard import GuardInput, QualityGuard


@pytest.fixture
def base_guard_input():
    return GuardInput(
        level="L1",
        finite=True,
        telemetry_complete=True,
        cg_status="converged",
        relative_residual=5e-7,
        requested_rtol=1e-6,
        audit_relative_increase=0.0,
        mass_below_fraction=0.0,
        beta_deltas=(1.0, 0.5, 0.2, 0.05),
        correction_beta_delta=5e-5,
        inner_beta_tol=1e-4,
    )


@pytest.mark.parametrize(
    ("changes", "action", "reason"),
    (
        ({"finite": False}, "strict_fallback", "nonfinite"),
        ({"telemetry_complete": False}, "strict_fallback", "incomplete_telemetry"),
        ({"cg_status": "max_iterations"}, "escalate", "cg_max_iterations"),
        ({"relative_residual": 2e-3, "requested_rtol": 1e-3}, "escalate", "cg_residual"),
        ({"audit_relative_increase": 0.006}, "escalate", "audit_objective"),
        ({"mass_below_fraction": 0.06}, "escalate", "local_mass"),
        ({"beta_deltas": (1.0, 0.98, 0.97, 0.96)}, "escalate", "beta_oscillation"),
        ({"correction_beta_delta": 2e-4}, "escalate", "strict_correction"),
    ),
)
def test_quality_guard_decisions(base_guard_input, changes, action, reason):
    decision = QualityGuard().evaluate(replace(base_guard_input, **changes))
    assert decision.action == action
    assert decision.reason == reason
```

Add positive cases for `L1 accept` and conversion of an `L2 escalate` into
`strict_fallback`.

- [ ] **Step 2: Run RED**

```bash
rtk python -m pytest tests/test_adaptive_quality_guard.py -q
```

Expected: guard module is missing.

- [ ] **Step 3: Implement immutable input and decision types**

```python
from dataclasses import dataclass
from typing import Literal

from adp.optimization.budget import BudgetName


GuardAction = Literal["accept", "escalate", "strict_fallback"]


@dataclass(frozen=True, slots=True)
class GuardInput:
    level: BudgetName
    finite: bool
    telemetry_complete: bool
    cg_status: str
    relative_residual: float
    requested_rtol: float
    audit_relative_increase: float
    mass_below_fraction: float
    beta_deltas: tuple[float, ...]
    correction_beta_delta: float
    inner_beta_tol: float


@dataclass(frozen=True, slots=True)
class GuardDecision:
    action: GuardAction
    reason: str


class QualityGuard:
    @staticmethod
    def _escalate(level: BudgetName, reason: str) -> GuardDecision:
        action: GuardAction = "strict_fallback" if level == "L2" else "escalate"
        return GuardDecision(action, reason)

    def evaluate(self, item: GuardInput) -> GuardDecision:
        if not item.finite:
            return GuardDecision("strict_fallback", "nonfinite")
        if not item.telemetry_complete:
            return GuardDecision("strict_fallback", "incomplete_telemetry")
        if item.cg_status == "max_iterations":
            return self._escalate(item.level, "cg_max_iterations")
        if item.cg_status != "converged":
            return GuardDecision("strict_fallback", "cg_failure")
        if item.relative_residual > item.requested_rtol:
            return self._escalate(item.level, "cg_residual")
        if item.audit_relative_increase > 0.005:
            return self._escalate(item.level, "audit_objective")
        if item.mass_below_fraction > 0.05:
            return self._escalate(item.level, "local_mass")
        recent = item.beta_deltas[-4:]
        if len(recent) == 4 and all(
            current > 0.95 * previous
            for previous, current in zip(recent, recent[1:])
        ):
            return self._escalate(item.level, "beta_oscillation")
        if item.correction_beta_delta > item.inner_beta_tol:
            return self._escalate(item.level, "strict_correction")
        return GuardDecision("accept", "accepted")
```

The order is safety-first: nonfinite, incomplete telemetry, CG, audit, mass,
three-step oscillation, correction delta, accept.

Export `GuardAction`, `GuardInput`, `GuardDecision` and `QualityGuard` from
`adp/optimization/__init__.py`.

- [ ] **Step 4: Verify every branch**

```bash
rtk python -m pytest tests/test_adaptive_quality_guard.py -q
```

Expected: branch table and accept cases pass.

- [ ] **Step 5: Commit**

```bash
rtk git add adp/optimization/guard.py adp/optimization/__init__.py tests/test_adaptive_quality_guard.py
rtk git commit -m "feat: add conservative adaptive quality guard"
```

### Task 10: Add the adaptive cascade and strict fallback

**Files:**
- Create: `adp/optimization/controller.py`
- Create: `tests/test_adaptive_controller.py`
- Modify: `adp/optimization/__init__.py`
- Modify: `adp/engine/algorithm.py:120-330,380-590`
- Modify: `tests/test_adp.py`
- Modify: `tests/test_stage_factories.py`
- Modify: `tests/test_solver_objective_consistency.py`

- [ ] **Step 1: Test controller ordering without running ADP**

```python
from adp.optimization.controller import AdaptiveCandidate, AdaptiveController
from adp.optimization.guard import GuardDecision


def _candidate(level, *, action):
    return AdaptiveCandidate(
        level=level,
        payload={"level": level},
        decision=GuardDecision(action=action, reason=f"{level}_{action}"),
        telemetry={"budget_level": level, "guard_action": action},
    )


def _fail():
    raise AssertionError("strict fallback must not run")


def _strict_candidate():
    return {"level": "strict"}


def test_controller_accepts_l1_without_l2():
    calls = []

    def solve(level):
        calls.append(level)
        return _candidate(level, action="accept")

    result = AdaptiveController().run("L1", solve_level=solve, strict_solve=_fail)
    assert result.accepted_level == "L1"
    assert calls == ["L1"]


def test_controller_escalates_then_strict_fallbacks():
    calls = []

    def solve(level):
        calls.append(level)
        action = "escalate" if level == "L1" else "strict_fallback"
        return _candidate(level, action=action)

    result = AdaptiveController().run("L1", solve_level=solve, strict_solve=_strict_candidate)
    assert result.accepted_level == "strict"
    assert calls == ["L1", "L2"]
    assert result.fallback_reason != ""
    assert result.level_telemetry[-1]["budget_level"] == "strict"
    assert result.level_telemetry[-1]["fallback_time_sec"] >= 0.0
```

- [ ] **Step 2: Run RED**

```bash
rtk python -m pytest tests/test_adaptive_controller.py -q
```

Expected: controller module is missing.

- [ ] **Step 3: Implement the generic controller**

```python
import time
from dataclasses import dataclass
from typing import Any

from adp.optimization.budget import BudgetName
from adp.optimization.guard import GuardDecision


@dataclass(slots=True)
class AdaptiveCandidate:
    level: BudgetName
    payload: Any
    decision: GuardDecision
    telemetry: dict[str, Any]


@dataclass(slots=True)
class AdaptiveOutcome:
    accepted_level: str
    payload: Any
    level_telemetry: list[dict[str, Any]]
    fallback_reason: str


class AdaptiveController:
    _next = {"L0": "L1", "L1": "L2"}

    def run(self, start_level, *, solve_level, strict_solve):
        level = start_level
        telemetry = []

        def strict_outcome(reason):
            fallback_started = time.perf_counter()
            payload = strict_solve()
            telemetry.append(
                {
                    "budget_level": "strict",
                    "level_attempt": len(telemetry),
                    "guard_action": "strict_fallback",
                    "fallback_reason": reason,
                    "fallback_time_sec": time.perf_counter() - fallback_started,
                }
            )
            return AdaptiveOutcome("strict", payload, telemetry, reason)

        while True:
            candidate = solve_level(level)
            telemetry.append(dict(candidate.telemetry))
            if candidate.decision.action == "accept":
                return AdaptiveOutcome(level, candidate.payload, telemetry, "")
            if candidate.decision.action == "strict_fallback":
                return strict_outcome(candidate.decision.reason)
            next_level = self._next.get(level)
            if next_level is None:
                return strict_outcome(candidate.decision.reason)
            level = next_level
```

Export `AdaptiveCandidate`, `AdaptiveOutcome` and `AdaptiveController` from
`adp/optimization/__init__.py`.

- [ ] **Step 4: Select adaptive builtin stages without changing strict mode**

In `ADPAlgorithm.__init__`, preserve copies of the caller-provided `stages` and
`stage_factories` before merging automatic adaptive defaults. Set an adaptive
default only when the user did not override that category:

```python
requested_stages = dict(stages or {})
requested_factories = dict(stage_factories or {})
selected_stages = dict(requested_stages)
if context.config.compute_mode == "adaptive_conservative":
    if "beta_solver" not in requested_factories:
        selected_stages.setdefault("beta_solver", "pcg")
    if "stop_rule" not in requested_factories:
        selected_stages.setdefault("stop_rule", "adaptive_convergence")
```

Build a second private reference component mapping from the original requested
mappings: explicit user overrides remain explicit, while unspecified
categories use `DEFAULT_STAGE_NAMES`. Instantiate it separately so stateful
stop rules cannot leak state between adaptive and fallback solves. In strict
mode, do not build or select any adaptive component.

- [ ] **Step 5: Add a separate adaptive outer branch**

Keep the existing body as `_run_strict_outer`. Add
`_run_adaptive_outer`, which:

1. requires builtin `statistics_builder.create_workspace`; otherwise calls
   `_run_strict_outer` and records `custom_statistics_builder`;
2. creates the outer-scoped workspace through the builtin builder, which
   computes full `q` but no `J x P` statistics tiles;
3. derives the observable calibration bucket from `d`, `n/d`, the workspace
   active fraction, kernel and dtype, loads `StatisticsBudget`, then applies its
   calibrated chunk size to the workspace;
4. builds `SolverBudget`, `QualityGuard` and `AdaptiveController` once per
   outer step;
5. materializes fit/audit statistics for each requested level;
6. calls `_alternating_solve` with `solve_with_request` and
   `AdaptiveConvergenceStopRule`;
7. performs one final PCG correction, constructs `GuardInput`, and returns an
   `AdaptiveCandidate`;
8. uses the original full-statistics SciPy-CG path for `strict_solve`.

Every level attempt starts from the same `beta_before_outer`. Do not append its
history, progress or mutable `ADPState` into the accepted fit state until the
controller accepts that level. Escalation reuses statistics tiles but not the
rejected numerical trajectory. Strict fallback likewise restarts from
`beta_before_outer`; keep rejected-attempt telemetry only in
`adaptive_telemetry`.

The final correction uses `SolverBudget.final_request`. Set
`correction_beta_delta` to the sign-invariant minimum norm between the
pre-correction and corrected unit vectors. Compute fitted and audit local-mass
failure fractions separately from `N_j < config.min_neighbors`, then pass their
maximum as `mass_below_fraction`; either group above `5%` must escalate.

If `dtype="float32"` and the loaded profile does not list the bucket in
`float32_buckets`, skip reduced levels and record strict fallback reason
`unapproved_float32_bucket`.

Refactor `_alternating_solve` to accept an optional `SolverBudget`. When it is
present, call `solve_with_request`; when absent, retain the exact existing
`solve` call. Never pass new kwargs to third-party beta solver stages.

Compute the adaptive stop metric only on objective-check iterations, using the
same before/after values and prior already used by `_objective`:

```python
relative_objective_change = abs(
    evaluated_objective_after - evaluated_objective_before
) / max(abs(evaluated_objective_before), np.finfo(float).eps)
```

On skipped objective checks pass `math.inf`; therefore they can never count as
one of the two consecutive successful stop checks.

For the runtime audit, use one outer-fixed audit statistics object and optimize
local coefficients separately for the before/after beta. Both objectives use
the outer input beta as the proximal prior:

```python
before_intercepts, before_slopes = local_solver.solve(audit_stats, outer_prior)
after_intercepts, after_slopes = local_solver.solve(audit_stats, candidate_beta)
audit_before = model._objective(
    audit_stats,
    outer_prior,
    before_intercepts,
    before_slopes,
    outer_prior,
    lambda_penalty,
)
audit_after = model._objective(
    audit_stats,
    candidate_beta,
    after_intercepts,
    after_slopes,
    outer_prior,
    lambda_penalty,
)
audit_relative_increase = (audit_after - audit_before) / max(
    abs(audit_before), np.finfo(float).eps
)
```

Add an optional `audit_check(beta) -> AuditCheck` callback to
`_alternating_solve`. Invoke it on every objective-check iteration and after
the final correction. Initialize its baseline with `outer_prior`; after a
finite check with relative increase `<= 0.005`, advance the baseline to that
beta/objective. On a failed check, keep the previous baseline, mark the current
candidate for escalation and set `inner_stop_reason="guard_escalation"`.
Pass `audit_ok` and `audit_relative_increase` to the adaptive stop rule and
`QualityGuard`. If either objective is nonfinite, pass `finite=False` to the
guard. This makes the audit requirement real on both consecutive stop checks,
not only a post-hoc telemetry field.

Keep the reference fallback independent from adaptive defaults.
`_run_reference_outer` uses the private reference components, full `J/P`
statistics and the original alternating loop; for an uncustomized model this
is exactly builtin SciPy CG plus `ConvergenceStopRule`, never selected
`pcg`/adaptive components. If the caller explicitly supplied any of
`statistics_builder`, `local_solver`, `beta_solver` or `stop_rule`, bypass the
cascade and run the pre-refactor strict path with the private reference
components. Record `custom_<category>` and manually account fallback time.

- [ ] **Step 6: Add integration tests**

Cover:

- strict default gives the same beta/history as before;
- adaptive known-good small data returns finite beta and records an accepted
  level;
- injected guard escalation calls `L2`;
- injected PCG failure calls strict SciPy fallback;
- custom statistics stage causes strict fallback, not a hidden partial fit.

Run:

```bash
rtk python -m pytest tests/test_adaptive_controller.py tests/test_adp.py tests/test_stage_factories.py tests/test_solver_objective_consistency.py -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
rtk git add adp/optimization/controller.py adp/optimization/__init__.py adp/engine/algorithm.py tests/test_adaptive_controller.py tests/test_adp.py tests/test_stage_factories.py tests/test_solver_objective_consistency.py
rtk git commit -m "feat: add adaptive ADP compute cascade"
```

## Phase D: Persistence, Resources, Calibration and Acceptance

### Task 11: Persist adaptive telemetry in benchmark artifacts

**Files:**
- Modify: `adp/common/types.py:228-288`
- Modify: `adp/engine/algorithm.py`
- Modify: `adp/evaluation/single_index/types.py:220-226`
- Modify: `adp/evaluation/single_index/schema.py`
- Modify: `adp/evaluation/single_index/executors.py:180-390`
- Modify: `adp/evaluation/single_index/storage.py`
- Modify: `tests/test_single_index_benchmark_schema.py`
- Modify: `tests/test_single_index_benchmark_executors.py`
- Modify: `tests/test_single_index_benchmark_storage.py`

- [ ] **Step 1: Add schema assertions first**

Require these outer columns:

```python
{
    "budget_level",
    "effective_J",
    "effective_P",
    "new_statistics_tiles",
    "reused_statistics_tiles",
    "audit_objective",
    "guard_action",
    "escalation_reason",
    "fallback_reason",
    "guard_time_sec",
    "fallback_time_sec",
} <= set(OUTER_ITERATION_COLUMNS)
```

Require these inner columns:

```python
{
    "budget_level",
    "solver_phase",
    "requested_cg_rtol",
    "inner_stop_reason",
} <= set(INNER_ITERATION_COLUMNS)
```

Require run summary counts for accepted `L0/L1/L2/strict` levels.

Add a separate `ADAPTIVE_LEVEL_COLUMNS` schema and
`adaptive_levels.csv`, because one outer step can try several levels:

```python
ADAPTIVE_LEVEL_COLUMNS = RUN_IDENTITY_COLUMNS + (
    "experiment",
    "seed",
    "outer_k",
    "level_attempt",
    "budget_level",
    "effective_J",
    "effective_P",
    "fit_center_count",
    "audit_center_count",
    "new_statistics_tiles",
    "reused_statistics_tiles",
    "requested_cg_rtol",
    "cg_iterations",
    "cg_status",
    "relative_cg_residual",
    "audit_objective_before",
    "audit_objective_after",
    "audit_relative_increase",
    "mass_below_fraction",
    "correction_beta_delta",
    "guard_action",
    "escalation_reason",
    "fallback_reason",
    "statistics_time_sec",
    "optimization_time_sec",
    "guard_time_sec",
    "fallback_time_sec",
)
```

- [ ] **Step 2: Run RED**

```bash
rtk python -m pytest tests/test_single_index_benchmark_schema.py tests/test_single_index_benchmark_executors.py -q
```

Expected: missing columns.

- [ ] **Step 3: Extend in-memory telemetry**

Add to `TrainingStep`:

```python
budget_level: str = "strict"
solver_phase: str = "strict"
requested_cg_rtol: float | None = None
```

Add
`adaptive_telemetry: list[dict[str, Any]] = field(default_factory=list)` to
`ADPResult`. Populate the outer fields from workspace/controller telemetry and
the inner fields from `SolverRequest`.

- [ ] **Step 4: Bump single-index schema to version 5 and persist fields**

Update column tuples and executor row builders. Run summary must derive:

```python
level_counts = Counter(str(row.get("budget_level", "strict")) for row in outer_rows)
strict_fallback_count = sum(
    str(row.get("budget_level", "strict")) == "strict"
    and bool(str(row.get("fallback_reason", "")))
    for row in outer_rows
)
values.update(
    {
        "accepted_L0_outer_count": level_counts["L0"],
        "accepted_L1_outer_count": level_counts["L1"],
        "accepted_L2_outer_count": level_counts["L2"],
        "strict_fallback_outer_count": strict_fallback_count,
    }
)
```

Add the four `accepted_L0_outer_count`, `accepted_L1_outer_count`,
`accepted_L2_outer_count` and `strict_fallback_outer_count` names to
`RUN_SUMMARY_COLUMNS`, plus `adaptive_row_count`; do not write keys outside the
schema tuple.

Extend `RunOutcome` with `adaptive_rows`, add `_adaptive_rows` in the executor,
and register `adaptive_levels` in `PUBLIC_TABLE_COLUMNS`, `_SHARD_COLUMNS`,
`_OUTCOME_ATTRIBUTES`, `_TABLE_KEYS` and `_DETAIL_TABLES`. Its unique key is
`(run_id, outer_k, level_attempt)`. Strict runs emit no adaptive-level rows but
retain explicit `budget_level="strict"` on their outer rows.
An adaptive strict-fallback row fills nonapplicable numeric fields with `NaN`
and counters with zero; it must never omit schema columns.

Update storage fixtures to include the new fields with strict-compatible
defaults.

- [ ] **Step 5: Verify persisted CSV contracts**

```bash
rtk python -m pytest tests/test_single_index_benchmark_schema.py tests/test_single_index_benchmark_executors.py tests/test_single_index_benchmark_storage.py tests/test_single_index_benchmark_reports.py -q
```

Expected: all pass; existing strict rows serialize with explicit defaults.

- [ ] **Step 6: Commit**

```bash
rtk git add adp/common/types.py adp/engine/algorithm.py adp/evaluation/single_index/types.py adp/evaluation/single_index/schema.py adp/evaluation/single_index/executors.py adp/evaluation/single_index/storage.py tests/test_single_index_benchmark_schema.py tests/test_single_index_benchmark_executors.py tests/test_single_index_benchmark_storage.py
rtk git commit -m "feat: persist adaptive ADP telemetry"
```

### Task 12: Enforce a physical-core resource budget and add soak logging

**Files:**
- Modify: `adp/evaluation/single_index/types.py:164-217`
- Modify: `adp/evaluation/cli.py:87-188`
- Modify: `adp/evaluation/single_index/executors.py:140-160`
- Modify: `adp/evaluation/single_index/runner.py:28-33,113-176,314-326`
- Create: `experiments/soak_adaptive_performance.py`
- Create: `tests/test_adaptive_soak.py`
- Modify: `tests/test_single_index_benchmark_runner.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write resource-budget tests**

```python
def test_auto_jobs_uses_physical_core_budget(monkeypatch):
    monkeypatch.setattr(runner.psutil, "cpu_count", lambda logical: 6 if not logical else 12)
    assert runner._resolve_process_jobs("auto") == 6


def test_explicit_oversubscription_is_rejected(monkeypatch):
    monkeypatch.setattr(runner, "_physical_cpu_budget", lambda: 6)
    with pytest.raises(ValueError, match="physical CPU budget"):
        runner._validate_parallel_budget(process_jobs=7, statistics_workers=1, blas_threads=1)
```

Add a soak-contract test that injects a two-iteration fake fit and asserts JSON
rows contain `rss_mib`, `thread_count`, `process_count`, `open_fds` and
`elapsed_sec`.

- [ ] **Step 2: Run RED**

```bash
rtk python -m pytest tests/test_single_index_benchmark_runner.py tests/test_adaptive_soak.py -q
```

Expected: physical-budget helpers and soak module are missing.

- [ ] **Step 3: Implement conservative CPU accounting**

```python
def _physical_cpu_budget() -> int:
    physical = psutil.cpu_count(logical=False)
    fallback = max(1, (os.cpu_count() or 1) // 2)
    available = int(physical) if physical is not None else fallback
    if hasattr(os, "sched_getaffinity"):
        available = min(available, len(os.sched_getaffinity(0)))
    return max(1, available)


def _validate_parallel_budget(*, process_jobs: int, statistics_workers: int, blas_threads: int) -> None:
    requested = process_jobs * statistics_workers * blas_threads
    available = _physical_cpu_budget()
    if requested > available:
        raise ValueError(
            f"requested parallelism {requested} exceeds physical CPU budget {available}"
        )
```

`auto` returns `_physical_cpu_budget()`. The single-index benchmark keeps
`statistics_workers=1` and `threadpool_limits(limits=1)`. Do not add worker
recycling; retain the fixed pool and true serial `jobs=1` path already present.

- [ ] **Step 4: Add CLI mode/profile wiring**

Add `compute_mode` and `adaptive_profile` to `SingleIndexSeriesConfig`, then
add:

```bash
--compute-mode {strict,adaptive_conservative}
--adaptive-profile PATH
```

Pass both into `_benchmark_adp_config`. Reject `--adaptive-profile` with
`--compute-mode strict`.

- [ ] **Step 5: Implement bounded soak artifact**

`experiments/soak_adaptive_performance.py` exposes a testable
`run_soak(fit_factory, *, duration_sec, sample_interval_sec, jobs)` and a CLI
accepting `--duration-sec`, `--sample-interval-sec`, `--jobs`,
`--adaptive-profile` and `--output`. The CLI factory generates the fixed
experiment-2 hard shape used by the old ~8-second run, cycling deterministic
seeds. Keep one spawned `ProcessPoolExecutor(max_workers=jobs)` alive for the
whole run, maintain at most `jobs` in-flight fits, and do not recycle workers;
the point is to expose lifecycle leaks under sustained load.
Each worker sets `statistics_workers=1` and wraps fit in
`threadpool_limits(limits=1)`, matching the benchmark resource contract.

The parent samples itself plus recursive children with psutil. Aggregate RSS,
thread count, process count and open descriptors into one row. Write every
sample to stdout with `flush=True` so redirected logs remain useful, and write
this JSON artifact on graceful completion:

```json
{
  "samples": [],
  "summary": {
    "warmup_samples": 5,
    "rss_slope_mib_per_min": 0.0,
    "max_thread_count": 0,
    "max_process_count": 0,
    "max_open_fds": 0
  }
}
```

Use least-squares slope over samples after the first five. Return exit code `2`
if RSS slope exceeds `1 MiB/min` or any resource count grows monotonically for
the last ten samples. Validate `jobs <= _physical_cpu_budget()` before creating
the pool. Once the deadline is reached, stop submitting work, await only the
already-running fits, shut down the pool, and atomically replace the output
JSON via a temporary file in the same directory.

Define growth as all nine consecutive differences `>= 0` and at least one
difference `> 0`; a stable constant count is not a failure.

- [ ] **Step 6: Verify runner and soak contracts**

```bash
rtk python -m pytest tests/test_single_index_benchmark_runner.py tests/test_cli.py tests/test_adaptive_soak.py -q
```

Expected: all pass without starting a real long benchmark.

- [ ] **Step 7: Commit**

```bash
rtk git add adp/evaluation/single_index/types.py adp/evaluation/cli.py adp/evaluation/single_index/executors.py adp/evaluation/single_index/runner.py experiments/soak_adaptive_performance.py tests/test_adaptive_soak.py tests/test_single_index_benchmark_runner.py tests/test_cli.py
rtk git commit -m "feat: bound ADP benchmark resources"
```

### Task 13: Add paired calibration and hard quality gates

**Files:**
- Modify: `experiments/compare_model_efficiency.py`
- Modify: `tests/test_model_efficiency_comparison.py`
- Modify: `adp/optimization/budget.py`

- [ ] **Step 1: Write gate tests**

```python
def _paired_frame(*, cosine_drop, objective_increase):
    baseline_cosine = 0.95
    baseline_objective = 100.0
    return pd.DataFrame(
        {
            "baseline_cosine_abs": [baseline_cosine],
            "candidate_cosine_abs": [baseline_cosine - cosine_drop],
            "baseline_full_evaluation_objective": [baseline_objective],
            "candidate_full_evaluation_objective": [baseline_objective * (1.0 + objective_increase)],
            "baseline_result_finite": [True],
            "candidate_result_finite": [True],
            "baseline_status": ["success"],
            "candidate_status": ["success"],
        }
    )


def test_adaptive_gate_accepts_conservative_pair():
    paired = pd.DataFrame(
        {
            "baseline_cosine_abs": [0.95],
            "candidate_cosine_abs": [0.947],
            "baseline_full_evaluation_objective": [100.0],
            "candidate_full_evaluation_objective": [100.5],
            "baseline_result_finite": [True],
            "candidate_result_finite": [True],
            "baseline_status": ["success"],
            "candidate_status": ["success"],
        }
    )
    result = evaluate_adaptive_gate(paired)
    assert result["gate_pass"].tolist() == [True]


def test_adaptive_gate_rejects_quality_and_objective_regression():
    paired = _paired_frame(cosine_drop=0.006, objective_increase=0.011)
    result = evaluate_adaptive_gate(paired)
    assert result["gate_pass"].tolist() == [False]
    assert "cosine_abs" in result.loc[0, "gate_reasons"]
    assert "objective" in result.loc[0, "gate_reasons"]
```

- [ ] **Step 2: Run RED**

```bash
rtk python -m pytest tests/test_model_efficiency_comparison.py -q
```

Expected: `evaluate_adaptive_gate` is missing.

- [ ] **Step 3: Persist statuses and adaptive telemetry in comparison rows**

Extend `_MODEL_METRICS` and `_execute_fit_task` with `status`, `stop_reason`,
accepted budget level, fallback count, total CG iterations, total inner
iterations and `full_evaluation_objective`. Keep AB/BA order, one CPU per pair
and fresh process isolation.

Classify a finite result as `nonconverged` when any accepted outer step ends
with `inner_stop_reason == "iteration_limit"`; otherwise classify it as
`success`.
Catch only expected numerical failures (`StageExecutionError`,
`FloatingPointError`, `np.linalg.LinAlgError`) inside `_execute_fit_task` and
emit `status="numerical_failure"` with nonfinite result metrics. Unexpected
programming errors still propagate and fail the comparison. Add tests for all
three statuses and for a candidate failure being retained as a paired row.
Update `pair_model_runs` so an invalid encoded beta yields `NaN` comparison
metrics and `numerically_equivalent=False` instead of aborting the whole
paired frame; `evaluate_adaptive_gate` then rejects that row explicitly.

Add `evaluation_h: float` to `_FitTask`. In `_iter_fit_tasks`, deserialize a
fresh copy of the baseline model once per generated dataset, configure its
`J/P`, and derive a deterministic common evaluation bandwidth from the input
data and strict schedule:

```python
evaluation_model = cloudpickle.loads(model_payloads[0])
_configure_child_model(
    evaluation_model,
    data.centers.shape[0],
    data.directions.shape[1],
)
initial_h = evaluation_model._select_isotropic_bandwidth_default(
    data.X, data.centers, None
)
evaluation_h = float(
    initial_h
    * evaluation_model.config.initial_bandwidth_inflation
    / evaluation_model.config.bandwidth_decay
    ** max(0, evaluation_model.config.outer_steps - 1)
)
```

Put that same scalar into both fit tasks; this work happens in the parent and
cannot warm either child. After `fit_finished_ns`, evaluate each successful
returned beta on the same full centers, input directions, `task.evaluation_h`,
and `anisotropy=None`. This standardized full evaluation is excluded from
`fit_time_sec`:

```python
full_statistics = model._compute_statistics_default(
    task.X,
    task.y,
    task.centers,
    task.evaluation_h,
    result.beta,
    task.directions,
    None,
)
full_intercepts, full_slopes = model._solve_local_coefficients_default(
    full_statistics,
    result.beta,
)
full_evaluation_objective = model._objective(
    full_statistics,
    result.beta,
    full_intercepts,
    full_slopes,
    result.beta,
    model.config.resolved_lambda(),
)
```

At `compare_models` entry, require both ADP configs to agree on
`outer_steps`, `bandwidth_decay`, `initial_bandwidth_inflation`, `kernel`,
`ridge` and resolved lambda; raise `ValueError` otherwise. Add a focused test
so the metric cannot silently compare different objectives.

Using `prior=result.beta` makes the proximal penalty identically zero. The two
statistics objects still depend on their respective beta, as required by the
model, but share the full data, centers, directions, bandwidth and isotropic
geometry; they therefore do not compare objectives from different reduced
budgets. Numerical-failure rows receive `NaN` for this metric.

- [ ] **Step 4: Implement the exact gate formula**

```python
def evaluate_adaptive_gate(paired: pd.DataFrame) -> pd.DataFrame:
    result = paired.copy()
    result["cosine_drop"] = (
        pd.to_numeric(result["baseline_cosine_abs"], errors="coerce")
        - pd.to_numeric(result["candidate_cosine_abs"], errors="coerce")
    )
    baseline = pd.to_numeric(
        result["baseline_full_evaluation_objective"], errors="coerce"
    )
    candidate = pd.to_numeric(
        result["candidate_full_evaluation_objective"], errors="coerce"
    )
    result["objective_relative_increase"] = (candidate - baseline) / np.maximum(
        np.abs(baseline), np.finfo(float).eps
    )
    result["finite_pair"] = _boolean_series(result["baseline_result_finite"]) & _boolean_series(
        result["candidate_result_finite"]
    )
    baseline_status = result["baseline_status"].astype(str)
    candidate_status = result["candidate_status"].astype(str)
    new_numerical_failure = candidate_status.eq("numerical_failure") & ~baseline_status.eq(
        "numerical_failure"
    )
    new_nonconvergence = candidate_status.eq("nonconverged") & baseline_status.eq(
        "success"
    )
    result["status_ok"] = ~(new_numerical_failure | new_nonconvergence)
    result["gate_pass"] = (
        result["finite_pair"]
        & result["status_ok"]
        & result["cosine_drop"].le(0.005)
        & result["objective_relative_increase"].le(0.01)
    )
    result["gate_reasons"] = result.apply(_adaptive_gate_reasons, axis=1)
    return result


def _adaptive_gate_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    if not bool(row["finite_pair"]):
        reasons.append("nonfinite")
    if not bool(row["status_ok"]):
        reasons.append("status")
    if not math.isfinite(float(row["cosine_drop"])) or float(row["cosine_drop"]) > 0.005:
        reasons.append("cosine_abs")
    objective_increase = float(row["objective_relative_increase"])
    if not math.isfinite(objective_increase) or objective_increase > 0.01:
        reasons.append("objective")
    return ",".join(reasons)
```

The helper emits a comma-separated stable order:
`nonfinite,status,cosine_abs,objective`.

- [ ] **Step 5: Add calibration profile output**

Group calibration rows by observable bucket. Write a bucket to `l0_buckets`
only when every pair passes the quality gate, no candidate fit is more than
`5%` slower than its baseline, and median `time_speedup >= 1.10`. A bucket that
passes quality but misses the speed condition remains on `L1`; never trade
quality tolerance for speed. The JSON must contain:

```json
{
  "schema_version": 1,
  "status": "approved",
  "quality_gate": {"max_cosine_drop": 0.005, "max_objective_increase": 0.01},
  "l0_buckets": [],
  "chunk_sizes": {},
  "float32_buckets": []
}
```

Add CLI options `--adaptive-gate`, `--adaptive-profile PATH` and
`--write-profile PATH`. During calibration, the candidate uses
`adaptive_start_level="L0"`. During holdout, it uses
`adaptive_start_level="auto"` and loads `--adaptive-profile`.

For every approved bucket, microbenchmark candidate center chunks
`(8, 16, 32, 64)` with the same arrays and store the fastest median in
`chunk_sizes`. Add a bucket to `float32_buckets` only when a separate float32
candidate passes the same quality gate and is at least `10%` faster than the
float64 candidate. Return `2` when any paired holdout row fails.

Add profile-writer tests for: an all-pass and fast bucket entering
`l0_buckets`; a quality-pass but slow bucket remaining on `L1`; one failed seed
rejecting the whole bucket; and float32 requiring both the same quality gate
and `>= 1.10x` speedup. Stub the chunk timer so the chosen median chunk is
deterministic in unit tests.

- [ ] **Step 6: Verify comparison tests**

```bash
rtk python -m pytest tests/test_model_efficiency_comparison.py -q
```

Expected: all existing isolation/memory/equivalence tests and new gates pass.

- [ ] **Step 7: Run calibration seeds and holdout seeds**

```bash
rtk python experiments/compare_model_efficiency.py --profile full --seeds 0:9 --jobs 1 --adaptive-gate --write-profile /tmp/adp_adaptive_profile.json --output /tmp/adp_adaptive_calibration
rtk python experiments/compare_model_efficiency.py --profile full --seeds 10:29 --jobs 1 --adaptive-gate --adaptive-profile /tmp/adp_adaptive_profile.json --output /tmp/adp_adaptive_holdout
```

Expected: both commands exit `0`; every holdout pair passes quality gates.
If calibration fails, leave the failing bucket on `L1` or `L2`; do not widen
tolerances.

- [ ] **Step 8: Commit**

```bash
rtk git add experiments/compare_model_efficiency.py tests/test_model_efficiency_comparison.py adp/optimization/budget.py
rtk git commit -m "test: gate adaptive ADP quality and speed"
```

### Task 14: Run performance gates, long-run verification and document usage

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-22-adp-adaptive-conservative-performance-design.md` only if measured results require correcting a factual assumption

- [ ] **Step 1: Run focused unit suites**

```bash
rtk python -m pytest tests/test_statistics_benchmark.py tests/test_pcg.py tests/test_adaptive_config.py tests/test_adaptive_solver_budget.py tests/test_adaptive_statistics_budget.py tests/test_adaptive_quality_guard.py tests/test_adaptive_controller.py tests/test_adaptive_soak.py -q
```

Expected: all pass.

- [ ] **Step 2: Run algorithm and benchmark integration suites**

```bash
rtk python -m pytest tests/test_adp.py tests/test_performance_optimizations.py tests/test_solver_edge_cases.py tests/test_solver_objective_consistency.py tests/test_stage_factories.py tests/test_single_index_benchmark_schema.py tests/test_single_index_benchmark_executors.py tests/test_single_index_benchmark_storage.py tests/test_single_index_benchmark_runner.py tests/test_single_index_benchmark_reports.py tests/test_model_efficiency_comparison.py -q
```

Expected: all pass.

- [ ] **Step 3: Run the complete test suite and static checks**

```bash
rtk python -m pytest -q
rtk python -m compileall -q adp experiments
rtk git diff --check
```

Expected: full suite passes, compilation exits `0`, diff check is empty.

- [ ] **Step 4: Measure exact statistics before/after**

```bash
rtk env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/benchmark_numpy_statistics.py --repetitions 7 --output /tmp/adp_stats_after.json
```

Compare medians against `/tmp/adp_stats_before.json`. Acceptance:

- primary exact statistics speedup `>= 1.5x`, or a documented lower exact
  speedup only when final end-to-end gates below all pass;
- no small/sparse case regression above `5%`;
- output shapes and dtype unchanged.

- [ ] **Step 5: Run strict and adaptive single-index series**

```bash
rtk python run_benchmarks.py single-index --profile full --experiments 2 --seeds 10:29 --diagnostic-seeds 10 --jobs 1 --compute-mode strict --output /tmp/adp_strict_holdout_series
rtk python run_benchmarks.py single-index --profile full --experiments 2 --seeds 10:29 --diagnostic-seeds 10 --jobs 1 --compute-mode adaptive_conservative --adaptive-profile /tmp/adp_adaptive_profile.json --output /tmp/adp_adaptive_holdout_series
```

Acceptance:

- old `~8.02 s` shape has adaptive median `<= 4 s`;
- CG-heavy median speedup `>= 2x`;
- large-case p90 speedup `>= 1.5x`;
- peak RSS increase `<= 10%`;
- every paired quality row passes.

- [ ] **Step 6: Run the bounded 30-minute soak**

```bash
rtk python experiments/soak_adaptive_performance.py --duration-sec 1800 --sample-interval-sec 5 --jobs 6 --adaptive-profile /tmp/adp_adaptive_profile.json --output /tmp/adp_adaptive_soak.json
```

Expected: exit `0`, RSS slope `<= 1 MiB/min`, no monotonic growth of process,
thread or file-descriptor counts after warm-up.

- [ ] **Step 7: Document exact commands and fallback semantics**

Add to `README.md`:

```python
config = ADPConfig(
    compute_mode="adaptive_conservative",
    adaptive_profile="/path/to/adp_adaptive_profile.json",
    show_progress=False,
)
model = ADP.create("new", config)
```

Document that unknown buckets start at `L1`, guard failure escalates to `L2`,
and unresolved failure reruns strict full `J/P` with SciPy CG.

- [ ] **Step 8: Commit documentation**

```bash
rtk git add README.md
rtk git commit -m "docs: describe adaptive conservative ADP mode"
```

- [ ] **Step 9: Capture the final evidence**

Run:

```bash
rtk git status --short
rtk git log --oneline -14
```

Expected: implementation worktree is clean and the task commits appear in the
order defined above. Report exact test counts, before/after medians, p90,
quality failures, peak RSS and soak slopes; do not report projected values as
measured results.

## Stop/Go Rules

- A code optimization that fails numerical tests is removed, not hidden by
  fallback.
- A reduced bucket that fails calibration or holdout remains on `L1/L2`.
- A change that passes quality but regresses its target hot path by more than
  `5%` is reverted before the next task.
- If the exact statistics kernel plus adaptive cascade miss the end-to-end
  gates, write a separate compiled-kernel design; do not add Numba/Cython as an
  unplanned dependency inside this implementation.
- Do not run the full or 30-minute workload with oversubscribed settings.
