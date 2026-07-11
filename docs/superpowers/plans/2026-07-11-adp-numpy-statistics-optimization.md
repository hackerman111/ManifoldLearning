# ADP NumPy Statistics Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accelerate the warmed NumPy compact-kernel statistics stage by at least 1.5x while preserving ADP numerical behavior, dtype, memory bounds, stage contracts, and CuPy behavior.

**Architecture:** Keep the existing stage-factory boundary and move only backend arithmetic. Add one shared backend method for building the kernel argument, fuse the compact NumPy reductions to eliminate temporary arrays and mathematically redundant `S` work, then activate an explicit bounded thread path only if the serial benchmark misses the 1.5x gate.

**Tech Stack:** Python 3.14, NumPy, CuPy-compatible backend protocol, SciPy-dependent ADP runtime, `concurrent.futures.ThreadPoolExecutor` only if required, pytest, JSON benchmark artifacts.

---

## Execution Preconditions

The current checkout contains pre-existing uncommitted stage-factory changes in files this plan may also modify, including `adp/common/types.py`, `adp/engine/base.py`, and `adp/variants/random_projection.py`.

Before Task 1:

1. preserve those changes in an owner-approved commit or a dedicated snapshot worktree;
2. execute this plan on top of that exact snapshot;
3. do not stage unrelated files or generated outputs;
4. keep all benchmark outputs under `/tmp`.

If the stage-factory changes are still uncommitted and their owner has not authorized a snapshot commit, stop before implementation and request that authorization. Do not use a broad `git add` command in the dirty checkout.

## File Map

- Create `experiments/benchmark_numpy_statistics.py`: repeatable warmed statistics microbenchmark with primary, sparser, and denser localization cases.
- Create `tests/test_statistics_benchmark.py`: functional contract for the benchmark artifact; it never asserts wall-clock speed.
- Modify `adp/backends/numpy_backend.py`: fused kernel-argument construction, fused compact sums, and conditional bounded center parallelism.
- Modify `adp/backends/cupy_backend.py`: shared kernel-argument API with unchanged GPU accumulation behavior.
- Modify `adp/variants/random_projection.py`: delegate `q` construction to the backend.
- Conditionally modify `adp/common/types.py`: explicit `statistics_workers` configuration and validation.
- Conditionally modify `adp/engine/base.py`: pass the explicit worker count only to `NumpyBackend`.
- Modify `tests/test_performance_optimizations.py`: reference-equivalence, dtype, empty-neighborhood, backend-argument, and optional worker tests.

### Task 1: Add the repeatable statistics benchmark and record the baseline

**Files:**
- Create: `experiments/benchmark_numpy_statistics.py`
- Create: `tests/test_statistics_benchmark.py`

- [ ] **Step 1: Write the failing benchmark-contract test**

Create `tests/test_statistics_benchmark.py`:

```python
from experiments.benchmark_numpy_statistics import StatisticsBenchmarkCase, run_case


def test_statistics_benchmark_returns_repeatable_record():
    case = StatisticsBenchmarkCase(
        name="tiny",
        n=24,
        d=3,
        n_centers=5,
        n_directions=2,
        h_multiplier=1.0,
    )

    record = run_case(case, repetitions=2, seed=7)

    assert record["name"] == "tiny"
    assert record["shape"] == {"n": 24, "d": 3, "J": 5, "P": 2}
    assert 0.0 <= record["active_fraction"] <= 1.0
    assert record["repetitions"] == 2
    assert len(record["times_sec"]) == 2
    assert record["median_sec"] >= 0.0
    assert record["peak_memory_kib"] > 0.0
    assert record["statistics_shapes"] == {
        "imav": [5, 2],
        "S": [5, 2],
        "U": [5, 2, 3],
        "N": [5],
    }
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
python -m pytest tests/test_statistics_benchmark.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'experiments.benchmark_numpy_statistics'`.

- [ ] **Step 3: Implement the benchmark artifact**

Create `experiments/benchmark_numpy_statistics.py`:

```python
from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from adp import ADP, ADPConfig


@dataclass(frozen=True, slots=True)
class StatisticsBenchmarkCase:
    name: str
    n: int
    d: int
    n_centers: int
    n_directions: int
    h_multiplier: float


DEFAULT_CASES = (
    StatisticsBenchmarkCase("sparser", 1000, 100, 150, 16, 0.75),
    StatisticsBenchmarkCase("primary", 1000, 100, 150, 16, 1.00),
    StatisticsBenchmarkCase("denser", 1000, 100, 150, 16, 1.25),
)


def run_case(
    case: StatisticsBenchmarkCase,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    config = ADPConfig(
        n_centers=case.n_centers,
        n_directions=case.n_directions,
        min_neighbors=16.0,
        chunk_size=32,
        kernel="epanechnikov",
        backend="numpy",
        dtype="float64",
        center_noise_scale=0.1,
        use_neighbor_index=False,
        show_progress=False,
        random_state=seed,
    )
    model = ADP.create("new", config)
    data = model.generate_data(
        n=case.n,
        d=case.d,
        noise=0.05,
        sigma_x=1.0,
        corr=0.0,
        link="linear",
    )
    if data.directions is None:
        raise RuntimeError("new ADP benchmark requires random directions")

    selected_h = model._select_isotropic_bandwidth(data.X, data.centers, None)
    h = float(selected_h * case.h_multiplier)
    norm2 = model._cached_pairwise_norm2(data.X, data.centers)
    active_fraction = float(np.mean(np.asarray(norm2) / (h * h) < 1.0))

    statistics_result = model._compute_statistics(
        data.X,
        data.y,
        data.centers,
        h,
        data.beta,
        data.directions,
        None,
    )
    times: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        statistics_result = model._compute_statistics(
            data.X,
            data.y,
            data.centers,
            h,
            data.beta,
            data.directions,
            None,
        )
        times.append(time.perf_counter() - started)

    tracemalloc.start()
    model._compute_statistics(
        data.X,
        data.y,
        data.centers,
        h,
        data.beta,
        data.directions,
        None,
    )
    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "name": case.name,
        "case": asdict(case),
        "shape": {
            "n": case.n,
            "d": case.d,
            "J": case.n_centers,
            "P": case.n_directions,
        },
        "h": h,
        "active_fraction": active_fraction,
        "repetitions": repetitions,
        "times_sec": times,
        "median_sec": float(statistics.median(times)),
        "min_sec": float(min(times)),
        "peak_memory_kib": float(peak_memory_bytes / 1024.0),
        "statistics_shapes": {
            "imav": list(statistics_result.imav.shape),
            "S": list(statistics_result.S.shape) if statistics_result.S is not None else None,
            "U": list(statistics_result.U.shape) if statistics_result.U is not None else None,
            "N": list(statistics_result.N.shape) if statistics_result.N is not None else None,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark warmed NumPy ADP statistics.")
    parser.add_argument("--case", choices=("all",) + tuple(case.name for case in DEFAULT_CASES), default="all")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = DEFAULT_CASES if args.case == "all" else tuple(case for case in DEFAULT_CASES if case.name == args.case)
    records = [run_case(case, repetitions=args.repetitions, seed=args.seed) for case in cases]
    payload = {"records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the focused benchmark test to verify GREEN**

Run:

```bash
python -m pytest tests/test_statistics_benchmark.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Record the isolated and end-to-end baselines**

Run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/benchmark_numpy_statistics.py --repetitions 7 --output /tmp/adp_numpy_statistics_before.json
MPLCONFIGDIR=/tmp/adp-mpl-smoke-before python run_benchmarks.py stress --profile smoke --max-cases 1 --output /tmp/adp_numpy_statistics_smoke_before --no-latex
MPLCONFIGDIR=/tmp/adp-mpl-before python run_benchmarks.py stress --profile large --max-cases 1 --output /tmp/adp_numpy_statistics_large_before --no-latex
```

Expected: three timing and peak-memory records in `/tmp/adp_numpy_statistics_before.json`, one smoke record, and one successful large stress record.

- [ ] **Step 6: Commit the benchmark artifact**

```bash
git add experiments/benchmark_numpy_statistics.py tests/test_statistics_benchmark.py
git commit -m "test: add NumPy statistics benchmark harness"
```

### Task 2: Fuse backend construction of the kernel argument

**Files:**
- Modify: `tests/test_performance_optimizations.py`
- Modify: `adp/backends/numpy_backend.py:176-192`
- Modify: `adp/backends/cupy_backend.py:218-232`
- Modify: `adp/variants/random_projection.py:53-74`

- [ ] **Step 1: Add failing NumPy and CuPy kernel-argument tests**

Append to `tests/test_performance_optimizations.py`:

```python
@pytest.mark.parametrize("dtype", ("float64", "float32"))
def test_numpy_kernel_argument_matches_isotropic_and_anisotropic_formulas(dtype):
    backend = NumpyBackend(dtype)
    norm2 = np.array([[1.0, 4.0], [9.0, 16.0]], dtype=dtype)
    projection2 = np.array([[0.25, 1.0], [2.25, 4.0]], dtype=dtype)

    isotropic = backend.kernel_argument(norm2, h=2.0)
    anisotropic = backend.kernel_argument(
        norm2,
        h=2.0,
        projection2=projection2,
        anisotropy=0.5,
    )

    np.testing.assert_allclose(isotropic, norm2 / 4.0)
    np.testing.assert_allclose(anisotropic, (0.25 * norm2 + projection2) / 4.0)
    assert isotropic.dtype == np.dtype(dtype)
    assert anisotropic.dtype == np.dtype(dtype)


def test_kernel_argument_requires_projection_for_anisotropy():
    with pytest.raises(ValueError, match="projection2"):
        NumpyBackend().kernel_argument(np.ones((2, 3)), h=1.0, anisotropy=0.5)


def test_cupy_kernel_argument_matches_numpy_with_fake_cupy(monkeypatch):
    from adp.backends.cupy_backend import CupyBackend

    install_fake_cupy(monkeypatch)
    norm2 = np.array([[1.0, 4.0], [9.0, 16.0]])
    projection2 = np.array([[0.25, 1.0], [2.25, 4.0]])
    backend = CupyBackend()

    actual = backend.kernel_argument(
        norm2,
        h=2.0,
        projection2=projection2,
        anisotropy=0.5,
    )

    np.testing.assert_allclose(backend.to_numpy(actual), (0.25 * norm2 + projection2) / 4.0)
```

- [ ] **Step 2: Run the new tests to verify RED**

Run:

```bash
python -m pytest \
  tests/test_performance_optimizations.py::test_numpy_kernel_argument_matches_isotropic_and_anisotropic_formulas \
  tests/test_performance_optimizations.py::test_kernel_argument_requires_projection_for_anisotropy \
  tests/test_performance_optimizations.py::test_cupy_kernel_argument_matches_numpy_with_fake_cupy -q
```

Expected: all selected tests fail with `AttributeError` for missing `kernel_argument`.

- [ ] **Step 3: Add the fused NumPy kernel-argument method**

Insert in `NumpyBackend` immediately before `local_mass_score`:

```python
    def kernel_argument(
        self,
        norm2: np.ndarray,
        *,
        h: float,
        projection2: np.ndarray | None = None,
        anisotropy: float | None = None,
    ) -> np.ndarray:
        """Builds the kernel quadratic-form argument in one output buffer."""

        xnorm2 = self.asarray(norm2)
        inverse_h2 = self.dtype(1.0 / (float(h) * float(h)))
        q = np.empty_like(xnorm2)
        if anisotropy is None:
            np.multiply(xnorm2, inverse_h2, out=q)
            return q
        if projection2 is None:
            raise ValueError("projection2 is required when anisotropy is set")
        np.multiply(xnorm2, self.dtype(float(anisotropy) ** 2), out=q)
        np.add(q, self.asarray(projection2), out=q)
        np.multiply(q, inverse_h2, out=q)
        return q
```

- [ ] **Step 4: Add the equivalent CuPy backend method**

Insert in `CupyBackend` immediately before `local_mass_score`:

```python
    def kernel_argument(
        self,
        norm2: Any,
        *,
        h: float,
        projection2: Any | None = None,
        anisotropy: float | None = None,
    ) -> Any:
        """Builds the kernel quadratic-form argument on the GPU."""

        xnorm2 = self._gpu_array(norm2)
        inverse_h2 = self.dtype(1.0 / (float(h) * float(h)))
        if anisotropy is None:
            return xnorm2 * inverse_h2
        if projection2 is None:
            raise ValueError("projection2 is required when anisotropy is set")
        return (
            self.dtype(float(anisotropy) ** 2) * xnorm2
            + self._gpu_array(projection2)
        ) * inverse_h2
```

- [ ] **Step 5: Delegate chunk `q` construction to the backend**

Replace the current isotropic/anisotropic arithmetic inside the center-chunk loop in `RandomProjectionADP._compute_statistics_default` with:

```python
            proj2 = None if proj2_all is None else proj2_all[start:stop]
            q = self.backend.kernel_argument(
                norm2,
                h=h,
                projection2=proj2,
                anisotropy=anisotropy,
            )
```

Keep the existing guard before this block:

```python
            if anisotropy is not None and proj2_all is None:
                raise RuntimeError("projection cache не подготовлен")
```

- [ ] **Step 6: Run focused and backend regression tests**

Run:

```bash
python -m pytest tests/test_performance_optimizations.py -q
```

Expected: all tests in the file pass, including fake-CuPy transfer-count tests.

- [ ] **Step 7: Commit the kernel-argument change**

```bash
git add adp/backends/numpy_backend.py adp/backends/cupy_backend.py adp/variants/random_projection.py tests/test_performance_optimizations.py
git commit -m "perf: fuse ADP kernel argument construction"
```

### Task 3: Fuse the serial compact-kernel statistics loop

**Files:**
- Modify: `tests/test_performance_optimizations.py`
- Modify: `adp/backends/numpy_backend.py:262-302`

- [ ] **Step 1: Add failing exact-zero and empty-neighborhood tests**

Append to `tests/test_performance_optimizations.py`:

```python
@pytest.mark.parametrize("kernel", ("epanechnikov", "quartic"))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (("float64", 1e-11, 1e-12), ("float32", 2e-5, 2e-6)),
)
def test_fused_compact_statistics_match_reference_and_make_s_exact_zero(
    kernel,
    dtype,
    rtol,
    atol,
):
    rng = np.random.default_rng(43)
    X = rng.normal(size=(30, 5)).astype(dtype)
    y = rng.normal(size=30).astype(dtype)
    centers = rng.normal(size=(4, 5)).astype(dtype)
    directions = rng.normal(size=(4, 3, 5)).astype(dtype)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    q = (pairwise_norm2(X, centers) / 8.0).astype(dtype)
    backend = NumpyBackend(dtype)

    actual = backend.random_projection_sums(
        X=X,
        y=y,
        centers=centers,
        directions=directions,
        q=q,
        kernel=kernel,
    )
    expected = reference_random_projection_sums(X, y, centers, directions, q, kernel)

    np.testing.assert_allclose(actual[0], expected[0], rtol=rtol, atol=atol)
    np.testing.assert_array_equal(actual[1], np.zeros_like(actual[1]))
    np.testing.assert_allclose(actual[2], expected[2], rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual[3], expected[3], rtol=rtol, atol=atol)
    assert actual[0].dtype == np.dtype(dtype)
    assert actual[1].dtype == np.dtype(dtype)
    assert actual[2].dtype == np.dtype(dtype)
    assert actual[3].dtype == np.dtype(dtype)


def test_fused_compact_statistics_keep_empty_center_zero():
    X = np.array([[0.0, 0.0], [0.2, 0.1], [1.0, -0.5]])
    y = np.array([1.0, -0.5, 0.25])
    centers = np.array([[10.0, 10.0], [0.0, 0.0]])
    directions = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    q = np.array([[4.0, 5.0, 6.0], [0.0, 0.25, 2.0]])

    imav, s_vec, u_mat, counts, _ = NumpyBackend().random_projection_sums(
        X=X,
        y=y,
        centers=centers,
        directions=directions,
        q=q,
        kernel="epanechnikov",
    )

    np.testing.assert_array_equal(imav[0], np.zeros_like(imav[0]))
    np.testing.assert_array_equal(s_vec[0], np.zeros_like(s_vec[0]))
    np.testing.assert_array_equal(u_mat[0], np.zeros_like(u_mat[0]))
    assert counts[0] == 0.0
    assert counts[1] > 0.0
```

- [ ] **Step 2: Run the exact-zero test to verify RED**

Run:

```bash
python -m pytest \
  tests/test_performance_optimizations.py::test_fused_compact_statistics_match_reference_and_make_s_exact_zero \
  tests/test_performance_optimizations.py::test_fused_compact_statistics_keep_empty_center_zero -q
```

Expected: the exact-zero assertion for `S` fails against the current floating reduction.

- [ ] **Step 3: Replace the compact loop with the fused serial arithmetic**

Replace `NumpyBackend._compact_random_projection_sums` with:

```python
    def _compact_random_projection_sums(
        self,
        x: np.ndarray,
        xy: np.ndarray,
        xdirs: np.ndarray,
        xq: np.ndarray,
        kernel: KernelName,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Computes compact-kernel sums with in-place projected weights."""

        c_count, p_count, d = xdirs.shape
        imav = np.zeros((c_count, p_count), dtype=self.dtype)
        s_vec = np.zeros((c_count, p_count), dtype=self.dtype)
        u_mat = np.zeros((c_count, p_count, d), dtype=self.dtype)
        counts = np.zeros(c_count, dtype=self.dtype)
        tiny = np.finfo(self.dtype).eps

        for center_index in range(c_count):
            active = xq[center_index] < 1.0
            if not np.any(active):
                continue
            weights = self.kernel(xq[center_index, active], kernel).astype(
                self.dtype,
                copy=False,
            )
            count = weights.sum(dtype=self.dtype)
            counts[center_index] = count
            safe_count = max(float(count), float(tiny))
            x_active = x[active]
            y_active = xy[active]
            xbar = (weights @ x_active) / safe_count
            centered = x_active - xbar[None, :]
            projected = centered @ xdirs[center_index].T
            projected *= weights[:, None]
            imav[center_index] = y_active @ projected
            u_mat[center_index] = projected.T @ centered

        return (
            self.to_numpy(imav),
            self.to_numpy(s_vec),
            self.to_numpy(u_mat),
            self.to_numpy(counts),
            float(counts.mean()),
        )
```

- [ ] **Step 4: Run focused numerical tests**

Run:

```bash
python -m pytest tests/test_performance_optimizations.py tests/test_adp.py -q
```

Expected: all selected tests pass for compact, Gaussian, float64, and float32 behavior.

- [ ] **Step 5: Measure the serial optimization gate**

Run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/benchmark_numpy_statistics.py --repetitions 7 --output /tmp/adp_numpy_statistics_after_serial.json
python -c 'import json; b=json.load(open("/tmp/adp_numpy_statistics_before.json")); a=json.load(open("/tmp/adp_numpy_statistics_after_serial.json")); print({x["name"]: x["median_sec"] / y["median_sec"] for x, y in zip(b["records"], a["records"])})'
```

Expected: a dictionary of before/after speedup ratios. If `primary >= 1.5`, skip Task 4. If `primary < 1.5`, execute every step in Task 4.

- [ ] **Step 6: Commit the serial fused kernel**

```bash
git add adp/backends/numpy_backend.py tests/test_performance_optimizations.py
git commit -m "perf: fuse compact NumPy statistics"
```

### Task 4: Add explicit bounded center parallelism if the serial gate fails

Execute this task only when Task 3 records `primary < 1.5`.

**Files:**
- Modify: `tests/test_performance_optimizations.py`
- Modify: `tests/test_statistics_benchmark.py`
- Modify: `experiments/benchmark_numpy_statistics.py`
- Modify: `adp/common/types.py:33-65`
- Modify: `adp/engine/base.py:122-135`
- Modify: `adp/backends/numpy_backend.py:1-30,262-302`

- [ ] **Step 1: Add failing worker validation and equivalence tests**

Append to `tests/test_performance_optimizations.py`:

```python
def test_statistics_workers_must_be_positive():
    with pytest.raises(ValueError, match="statistics_workers"):
        ADPConfig(statistics_workers=0)

    with pytest.raises(ValueError, match="statistics_workers"):
        ADPConfig(statistics_workers=1.5)


def test_numpy_backend_receives_statistics_workers():
    model = ADP.create(
        "new",
        ADPConfig(statistics_workers=3, show_progress=False),
    )

    assert model.backend.statistics_workers == 3


def test_parallel_compact_statistics_match_serial(monkeypatch):
    monkeypatch.setattr(numpy_backend, "PARALLEL_STATISTICS_MIN_WORK", 0)
    rng = np.random.default_rng(59)
    X = rng.normal(size=(40, 6))
    y = rng.normal(size=40)
    centers = rng.normal(size=(6, 6))
    directions = rng.normal(size=(6, 4, 6))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    q = pairwise_norm2(X, centers) / 10.0

    serial = NumpyBackend(statistics_workers=1).random_projection_sums(
        X=X,
        y=y,
        centers=centers,
        directions=directions,
        q=q,
        kernel="epanechnikov",
    )
    parallel = NumpyBackend(statistics_workers=2).random_projection_sums(
        X=X,
        y=y,
        centers=centers,
        directions=directions,
        q=q,
        kernel="epanechnikov",
    )

    for serial_part, parallel_part in zip(serial, parallel):
        np.testing.assert_allclose(parallel_part, serial_part, rtol=1e-12, atol=1e-12)
```

- [ ] **Step 2: Run worker tests to verify RED**

Run:

```bash
python -m pytest \
  tests/test_performance_optimizations.py::test_statistics_workers_must_be_positive \
  tests/test_performance_optimizations.py::test_numpy_backend_receives_statistics_workers \
  tests/test_performance_optimizations.py::test_parallel_compact_statistics_match_serial -q
```

Expected: failures for the missing config field, backend constructor parameter, and parallel threshold constant.

- [ ] **Step 3: Add and validate the explicit worker setting**

Add to `ADPConfig` after `chunk_size`:

```python
    statistics_workers: int = 1
```

Add to `ADPConfig.__post_init__`:

```python
        if (
            isinstance(self.statistics_workers, bool)
            or not isinstance(self.statistics_workers, int)
            or self.statistics_workers < 1
        ):
            raise ValueError("statistics_workers должен быть положительным")
```

Update the NumPy branch of `ADPBase._make_backend`:

```python
            return NumpyBackend(
                self.config.dtype,
                statistics_workers=self.config.statistics_workers,
            )
```

Leave the CuPy constructor call unchanged.

- [ ] **Step 4: Add bounded center parallelism to `NumpyBackend`**

Add the import and module constant:

```python
from concurrent.futures import ThreadPoolExecutor

PARALLEL_STATISTICS_MIN_WORK = 1_000_000
```

Extend the constructor:

```python
    def __init__(
        self,
        dtype: str = "float64",
        *,
        statistics_workers: int = 1,
    ) -> None:
        self.name = "numpy"
        self.dtype_name = dtype
        if dtype not in {"float64", "float32"}:
            raise ValueError("dtype должен быть 'float64' или 'float32'")
        if (
            isinstance(statistics_workers, bool)
            or not isinstance(statistics_workers, int)
            or statistics_workers < 1
        ):
            raise ValueError("statistics_workers должен быть положительным")
        self.dtype = np.float64 if dtype == "float64" else np.float32
        self.statistics_workers = int(statistics_workers)
```

Inside `_compact_random_projection_sums`, replace the direct `for center_index in range(c_count)` loop with a nested worker and dispatch:

```python
        def compute_center(center_index: int) -> None:
            active = xq[center_index] < 1.0
            if not np.any(active):
                return
            weights = self.kernel(xq[center_index, active], kernel).astype(
                self.dtype,
                copy=False,
            )
            count = weights.sum(dtype=self.dtype)
            counts[center_index] = count
            safe_count = max(float(count), float(tiny))
            x_active = x[active]
            y_active = xy[active]
            xbar = (weights @ x_active) / safe_count
            centered = x_active - xbar[None, :]
            projected = centered @ xdirs[center_index].T
            projected *= weights[:, None]
            imav[center_index] = y_active @ projected
            u_mat[center_index] = projected.T @ centered

        work_proxy = c_count * x.shape[0] * p_count * d
        use_parallel = (
            self.statistics_workers > 1
            and c_count > 1
            and work_proxy >= PARALLEL_STATISTICS_MIN_WORK
        )
        if use_parallel:
            worker_count = min(self.statistics_workers, c_count)
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                list(executor.map(compute_center, range(c_count)))
        else:
            for center_index in range(c_count):
                compute_center(center_index)
```

- [ ] **Step 5: Expose the worker count in the benchmark artifact**

Change `run_case` in `experiments/benchmark_numpy_statistics.py` to accept:

```python
    statistics_workers: int = 1,
```

Pass it to `ADPConfig`:

```python
        statistics_workers=statistics_workers,
```

Add it to the returned record:

```python
        "statistics_workers": statistics_workers,
```

Add the CLI option:

```python
    parser.add_argument("--statistics-workers", type=int, default=1)
```

Pass `statistics_workers=args.statistics_workers` from `main` to every `run_case` call.

Update the call in `tests/test_statistics_benchmark.py`:

```python
    record = run_case(case, repetitions=2, seed=7, statistics_workers=1)
```

Add this assertion:

```python
    assert record["statistics_workers"] == 1
```

- [ ] **Step 6: Run focused worker and benchmark tests**

Run:

```bash
python -m pytest tests/test_performance_optimizations.py tests/test_statistics_benchmark.py -q
```

Expected: all tests pass, including exact serial/parallel equivalence.

- [ ] **Step 7: Benchmark explicit worker counts and select the fastest accepted count**

Run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/benchmark_numpy_statistics.py --case primary --statistics-workers 2 --repetitions 7 --output /tmp/adp_numpy_statistics_workers_2.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/benchmark_numpy_statistics.py --case primary --statistics-workers 4 --repetitions 7 --output /tmp/adp_numpy_statistics_workers_4.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/benchmark_numpy_statistics.py --case primary --statistics-workers 8 --repetitions 7 --output /tmp/adp_numpy_statistics_workers_8.json
python -c 'import json; p="/tmp/adp_numpy_statistics_before.json"; before=next(x for x in json.load(open(p))["records"] if x["name"]=="primary"); files=[f"/tmp/adp_numpy_statistics_workers_{n}.json" for n in (2,4,8)]; print({f: before["median_sec"] / json.load(open(f))["records"][0]["median_sec"] for f in files})'
```

Expected: printed speedups for 2, 4, and 8 workers. Inspect each JSON record's `peak_memory_kib`, reject any count above 105% of the matching baseline, then select the remaining count with the lowest median time that reaches at least 1.5x.

- [ ] **Step 8: Commit the explicit parallel path**

```bash
git add adp/common/types.py adp/engine/base.py adp/backends/numpy_backend.py experiments/benchmark_numpy_statistics.py tests/test_performance_optimizations.py tests/test_statistics_benchmark.py
git commit -m "perf: add bounded NumPy statistics workers"
```

### Task 5: Run final correctness, performance, memory, and stress verification

**Files:**
- Verify only; no source edits expected.

- [ ] **Step 1: Run focused tests**

```bash
python -m pytest tests/test_performance_optimizations.py tests/test_statistics_benchmark.py tests/test_adp.py tests/test_stage_factories.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run the complete test suite**

```bash
python -m pytest -q
```

Expected: all tests pass with no failures or errors.

- [ ] **Step 3: Run compilation and diff checks**

```bash
python -m py_compile \
  adp/common/types.py \
  adp/backends/numpy_backend.py \
  adp/backends/cupy_backend.py \
  adp/engine/base.py \
  adp/variants/random_projection.py \
  experiments/benchmark_numpy_statistics.py
git diff --check
```

Expected: both commands exit successfully with no output from `git diff --check`.

- [ ] **Step 4: Record the final isolated benchmark**

If Task 4 was skipped, run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/benchmark_numpy_statistics.py --repetitions 7 --output /tmp/adp_numpy_statistics_after.json
```

If Task 4 was executed, add the selected explicit worker option:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/benchmark_numpy_statistics.py --statistics-workers 4 --repetitions 7 --output /tmp/adp_numpy_statistics_after.json
```

Replace `4` with the accepted worker count selected in Task 4 Step 7.

Compare medians:

```bash
python -c 'import json; b=json.load(open("/tmp/adp_numpy_statistics_before.json")); a=json.load(open("/tmp/adp_numpy_statistics_after.json")); print({x["name"]: {"before": x["median_sec"], "after": y["median_sec"], "speedup": x["median_sec"] / y["median_sec"], "memory_ratio": y["peak_memory_kib"] / x["peak_memory_kib"]} for x, y in zip(b["records"], a["records"])})'
```

Expected: the `primary` record reports `speedup >= 1.5`; sparser and denser records are present and finite; serial `memory_ratio <= 1.0` within measurement noise or explicit parallel `memory_ratio <= 1.05`.

- [ ] **Step 5: Run the final smoke and large stress cases**

```bash
MPLCONFIGDIR=/tmp/adp-mpl-smoke-after python run_benchmarks.py stress --profile smoke --max-cases 1 --output /tmp/adp_numpy_statistics_smoke_after --no-latex
MPLCONFIGDIR=/tmp/adp-mpl-after python run_benchmarks.py stress --profile large --max-cases 1 --output /tmp/adp_numpy_statistics_large_after --no-latex
```

Expected: both commands succeed and write records, summaries, manifests, and plots.

- [ ] **Step 6: Compare end-to-end quality, time, and memory**

Run:

```bash
python -c 'import csv; paths=("/tmp/adp_numpy_statistics_large_before/adp_single_index_stress_records.csv", "/tmp/adp_numpy_statistics_large_after/adp_single_index_stress_records.csv"); rows=[next(csv.DictReader(open(path))) for path in paths]; keys=("cosine_abs","objective","statistics_time_sec","fit_time_sec","peak_memory_kib"); print({label:{key:float(row[key]) for key in keys} for label,row in zip(("before","after"),rows)})'
python -c 'import csv; paths=("/tmp/adp_numpy_statistics_smoke_before/adp_single_index_stress_records.csv", "/tmp/adp_numpy_statistics_smoke_after/adp_single_index_stress_records.csv"); rows=[next(csv.DictReader(open(path))) for path in paths]; print({"before":float(rows[0]["fit_time_sec"]),"after":float(rows[1]["fit_time_sec"]),"ratio":float(rows[1]["fit_time_sec"])/float(rows[0]["fit_time_sec"])})'
```

Expected:

- `cosine_abs`, `objective`, and the final beta behavior remain within numerical tolerance;
- serial peak memory does not increase beyond measurement noise;
- an explicit parallel run, if used, remains within the 5% peak-memory allowance;
- smoke/small runtime regression remains below 5%;
- statistics timing reflects the isolated benchmark improvement without a correctness regression.

- [ ] **Step 7: Confirm commit and worktree hygiene**

```bash
git log --oneline -5
git status --short
```

Expected: optimization commits contain only the files named in their task; unrelated pre-existing changes and generated `/tmp` artifacts are not staged.
