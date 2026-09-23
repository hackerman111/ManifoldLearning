# ADP Statistics Methods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and benchmark direct NumPy, stable dense-matrix, and exact sparse-CSR calculations of ADP single-index statistics.

**Architecture:** A hypothesis-specific subclass keeps the unfinished production class unchanged and presents one dispatcher over three mathematically equivalent kernels. A local benchmark prepares shared inputs once, measures only each kernel, and is launched through a generic hypothesis-tester CLI.

**Tech Stack:** Python 3.14, NumPy, SciPy sparse, standard-library `argparse`, `csv`, `statistics`, `subprocess`, `time`, and `tracemalloc`.

---

## File Map

- Modify `ADP/__init__.py`: remove the stale deleted-base import and export the current single-index class.
- Create `hypo/statistic/ADP_single_index_hypo_stat.py`: input validation, dispatcher, and all three statistics kernels.
- Create `hypo/statistic/tester.py`: deterministic experiments, correctness gates, time/memory/stability measurements, CSV output.
- Modify `hypo/tester.py`: discover and launch testers under `hypo/*/tester.py`.
- Create `hypo/statistic/benchmark_results.csv`: persisted output from the final standard run.

The existing dirty files `ADP/ADP_Data.py`, `ADP/ADP_single_index.py`,
`adp_matrix_statistics.tex`, and `manifold_v2.tex` belong to the user. Do not
rewrite, stage, or commit them as part of this implementation.

### Task 1: Repair the ADP package import

**Files:**

- Modify: `ADP/__init__.py:1-5`
- Read only: `ADP/ADP_single_index.py`

- [ ] **Step 1: Confirm the current import failure**

Run:

```bash
rtk python -c "from ADP.ADP_single_index import ADP_single_index"
```

Expected: failure mentioning `ADP.ADP_model_base`, because
`ADP/__init__.py` imports a deleted file.

- [ ] **Step 2: Replace the stale package exports**

Set `ADP/__init__.py` to:

```python
from .ADP_Config import ADP_Config
from .ADP_Data import ADP_Data
from .ADP_single_index import ADP_single_index

__all__ = ["ADP_Config", "ADP_Data", "ADP_single_index"]
```

- [ ] **Step 3: Verify the package can expose the current class**

Run:

```bash
rtk python -c "from ADP import ADP_Config, ADP_Data, ADP_single_index; print(ADP_single_index.__name__)"
```

Expected:

```text
ADP_single_index
```

### Task 2: Lock correctness first, then implement the three kernels

**Files:**

- Create: `hypo/statistic/tester.py`
- Create: `hypo/statistic/ADP_single_index_hypo_stat.py`

- [ ] **Step 1: Add a small failing cross-method check**

Create the first version of `hypo/statistic/tester.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ADP import ADP_Config
from hypo.statistic.ADP_single_index_hypo_stat import (
    ADP_single_index_hypo_stat,
)


def correctness_check() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(24, 5))
    Y = rng.normal(size=24)
    weights = rng.uniform(size=(4, 24))
    weights[weights < 0.55] = 0.0
    directions = rng.normal(size=(4, 3, 5))
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    model = ADP_single_index_hypo_stat(
        ADP_Config(n=1, d=1, N_J=1, N_phi=1)
    )

    direct = model.Calculate_statistic(
        weights,
        directions,
        method="direct",
        X=X,
        Y=Y,
    )
    matrix = model.Calculate_statistic(
        weights,
        directions,
        method="matrix",
        X=X,
        Y=Y,
        batch_size=2,
    )
    sparse_result = model.Calculate_statistic(
        sparse.csr_matrix(weights),
        directions,
        method="sparse",
        X=X,
        Y=Y,
    )

    for result in (matrix, sparse_result):
        np.testing.assert_allclose(result["I"], direct["I"], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(result["U"], direct["U"], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(
            result["mass"], direct["mass"], rtol=1e-13, atol=1e-13
        )
        np.testing.assert_allclose(
            result["mean"], direct["mean"], rtol=1e-12, atol=1e-12
        )


if __name__ == "__main__":
    correctness_check()
    print("correctness: ok")
```

- [ ] **Step 2: Run the check and observe the missing implementation**

Run:

```bash
rtk python hypo/statistic/tester.py
```

Expected: failure because
`hypo/statistic/ADP_single_index_hypo_stat.py` does not exist.

- [ ] **Step 3: Implement the modified class**

Create `hypo/statistic/ADP_single_index_hypo_stat.py`:

```python
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from ADP.ADP_single_index import ADP_single_index


class ADP_single_index_hypo_stat(ADP_single_index):
    """ADP single-index class with interchangeable statistics kernels."""

    def Calculate_statistic(
        self,
        weights: np.ndarray | sparse.spmatrix,
        directions: np.ndarray,
        *,
        method: str = "direct",
        X: np.ndarray | None = None,
        Y: np.ndarray | None = None,
        batch_size: int = 32,
    ) -> dict[str, np.ndarray]:
        if method == "direct":
            return self.Calculate_statistic_direct(
                weights, directions, X=X, Y=Y
            )
        if method == "matrix":
            return self.Calculate_statistic_matrix(
                weights,
                directions,
                X=X,
                Y=Y,
                batch_size=batch_size,
            )
        if method == "sparse":
            return self.Calculate_statistic_sparse(
                weights, directions, X=X, Y=Y
            )
        raise ValueError("method должен быть 'direct', 'matrix' или 'sparse'")

    def _statistics_inputs(
        self,
        weights: np.ndarray | sparse.spmatrix,
        directions: np.ndarray,
        X: np.ndarray | None,
        Y: np.ndarray | None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray | sparse.csr_matrix,
        np.ndarray,
        np.ndarray,
    ]:
        if X is None:
            if not hasattr(self, "X"):
                raise ValueError("X обязателен для неинициализированного класса")
            X = np.asarray(self.X).T
        if Y is None:
            if not hasattr(self, "Y"):
                raise ValueError("Y обязателен для неинициализированного класса")
            Y = self.Y

        x = np.asarray(X)
        y = np.asarray(Y)
        phi = np.asarray(directions)
        is_sparse = sparse.issparse(weights)
        w = weights.tocsr(copy=True) if is_sparse else np.asarray(weights)

        if x.ndim != 2:
            raise ValueError("X должен иметь форму (n, d)")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("Y должен иметь форму (n,)")
        if w.ndim != 2 or w.shape[1] != x.shape[0]:
            raise ValueError("weights должен иметь форму (J, n)")
        if (
            phi.ndim != 3
            or phi.shape[0] != w.shape[0]
            or phi.shape[2] != x.shape[1]
            or phi.shape[1] == 0
        ):
            raise ValueError("directions должен иметь форму (J, P, d), P > 0")

        try:
            dtype = np.result_type(
                x.dtype,
                y.dtype,
                w.dtype,
                phi.dtype,
                np.float32,
            )
        except TypeError as error:
            raise ValueError("входы должны быть числовыми") from error
        if not np.issubdtype(dtype, np.floating):
            raise ValueError("входы должны быть вещественными")

        x = np.asarray(x, dtype=dtype)
        y = np.asarray(y, dtype=dtype)
        phi = np.asarray(phi, dtype=dtype)
        if is_sparse:
            w = w.astype(dtype, copy=False)
            w.sum_duplicates()
            w.eliminate_zeros()
            w.sort_indices()
            weight_values = w.data
        else:
            w = np.asarray(w, dtype=dtype)
            weight_values = w

        for name, value in (("X", x), ("Y", y), ("directions", phi)):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} содержит нечисловые значения")
        if not np.all(np.isfinite(weight_values)):
            raise ValueError("weights содержит нечисловые значения")
        if np.any(weight_values < 0):
            raise ValueError("weights должен быть неотрицательным")

        mass = (
            np.asarray(w.sum(axis=1)).ravel()
            if is_sparse
            else w.sum(axis=1)
        )
        if np.any(mass <= 0):
            raise ValueError("каждый центр должен иметь положительную массу")
        return x, y, w, phi, mass

    def Calculate_statistic_direct(
        self,
        weights: np.ndarray | sparse.spmatrix,
        directions: np.ndarray,
        *,
        X: np.ndarray | None = None,
        Y: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        x, y, w, phi, mass = self._statistics_inputs(
            weights, directions, X, Y
        )
        if sparse.issparse(w):
            w = w.toarray()

        mean = (w @ x) / mass[:, None]
        differences = x[None, :, :] - mean[:, None, :]
        weighted_projection = np.einsum(
            "jnd,jpd->jpn",
            differences,
            phi,
            optimize=True,
        )
        weighted_projection *= w[:, None, :]
        I = weighted_projection @ y
        U = np.einsum(
            "jpn,jnd->jpd",
            weighted_projection,
            differences,
            optimize=True,
        )
        sum_w2 = np.square(w).sum(axis=1)
        eps = np.finfo(x.dtype).eps
        eta = np.abs(weighted_projection.sum(axis=2)) / (
            np.abs(weighted_projection).sum(axis=2) + eps
        )
        return {
            "I": I,
            "U": U,
            "mass": mass,
            "mean": mean,
            "n_eff": np.square(mass) / sum_w2,
            "eta": eta,
        }

    def Calculate_statistic_matrix(
        self,
        weights: np.ndarray | sparse.spmatrix,
        directions: np.ndarray,
        *,
        X: np.ndarray | None = None,
        Y: np.ndarray | None = None,
        batch_size: int = 32,
    ) -> dict[str, np.ndarray]:
        x, y, w, phi, mass = self._statistics_inputs(
            weights, directions, X, Y
        )
        if sparse.issparse(w):
            w = w.toarray()
        if batch_size < 1:
            raise ValueError("batch_size должен быть положительным")

        J, P, d = phi.shape
        I = np.empty((J, P), dtype=x.dtype)
        U = np.empty((J, P, d), dtype=x.dtype)
        mean = np.empty((J, d), dtype=x.dtype)
        eta = np.empty((J, P), dtype=x.dtype)
        x0 = x.mean(axis=0)
        xc = x - x0
        y_bar = (w @ y) / mass
        eps = np.finfo(x.dtype).eps

        for start in range(0, J, batch_size):
            stop = min(start + batch_size, J)
            wb = w[start:stop]
            mb = mass[start:stop]
            a = wb / mb[:, None]
            mc = a @ xc
            mean[start:stop] = mc + x0
            q = np.matmul(phi[start:stop], xc.T)
            q -= np.einsum(
                "bpd,bd->bp",
                phi[start:stop],
                mc,
                optimize=True,
            )[:, :, None]
            q -= np.einsum("bpn,bn->bp", q, a, optimize=True)[:, :, None]
            q *= a[:, None, :]
            s = q.sum(axis=2)
            I[start:stop] = mb[:, None] * (
                q @ y - s * y_bar[start:stop, None]
            )
            U[start:stop] = mb[:, None, None] * (
                q @ xc - s[:, :, None] * mc[:, None, :]
            )
            eta[start:stop] = np.abs(s) / (
                np.abs(q).sum(axis=2) + eps
            )

        return {
            "I": I,
            "U": U,
            "mass": mass,
            "mean": mean,
            "n_eff": np.square(mass) / np.square(w).sum(axis=1),
            "eta": eta,
        }

    def Calculate_statistic_sparse(
        self,
        weights: np.ndarray | sparse.spmatrix,
        directions: np.ndarray,
        *,
        X: np.ndarray | None = None,
        Y: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        x, y, w, phi, mass = self._statistics_inputs(
            weights, directions, X, Y
        )
        if not sparse.issparse(w):
            w = sparse.csr_matrix(w)

        J, P, d = phi.shape
        I = np.empty((J, P), dtype=x.dtype)
        U = np.empty((J, P, d), dtype=x.dtype)
        eta = np.empty((J, P), dtype=x.dtype)
        x0 = x.mean(axis=0)
        xc = x - x0
        mc = (w @ xc) / mass[:, None]
        mean = mc + x0
        y_bar = (w @ y) / mass
        row = np.repeat(np.arange(J), np.diff(w.indptr))
        edge_difference = xc[w.indices] - mc[row]
        normalized_weight = w.data / mass[row]
        eps = np.finfo(x.dtype).eps

        for direction_index in range(P):
            q = np.einsum(
                "ed,ed->e",
                edge_difference,
                phi[row, direction_index],
                optimize=True,
            )
            weighted_q = w.data * q
            eta[:, direction_index] = np.abs(
                np.add.reduceat(weighted_q, w.indptr[:-1])
            ) / (
                np.add.reduceat(np.abs(weighted_q), w.indptr[:-1]) + eps
            )

            h_data = normalized_weight * q
            correction = np.add.reduceat(h_data, w.indptr[:-1])
            h_data -= normalized_weight * correction[row]
            H = sparse.csr_matrix(
                (h_data, w.indices, w.indptr),
                shape=w.shape,
                copy=False,
            )
            s = np.asarray(H.sum(axis=1)).ravel()
            I[:, direction_index] = mass * (
                H @ y - s * y_bar
            )
            U[:, direction_index] = mass[:, None] * (
                H @ xc - s[:, None] * mc
            )

        sum_w2 = np.asarray(w.multiply(w).sum(axis=1)).ravel()
        return {
            "I": I,
            "U": U,
            "mass": mass,
            "mean": mean,
            "n_eff": np.square(mass) / sum_w2,
            "eta": eta,
        }


ADPSingleIndexHypoStat = ADP_single_index_hypo_stat
```

- [ ] **Step 4: Run the small cross-method check**

Run:

```bash
rtk python hypo/statistic/tester.py
```

Expected:

```text
correctness: ok
```

- [ ] **Step 5: Check rejected inputs**

Run:

```bash
rtk python -c "import numpy as np; from ADP import ADP_Config; from hypo.statistic.ADP_single_index_hypo_stat import ADP_single_index_hypo_stat as M; m=M(ADP_Config(n=1,d=1,N_J=1,N_phi=1)); m.Calculate_statistic(np.zeros((1,2)),np.ones((1,1,1)),X=np.ones((2,1)),Y=np.ones(2))"
```

Expected: `ValueError` mentioning positive mass.

### Task 3: Turn the correctness check into the benchmark tester

**Files:**

- Modify: `hypo/statistic/tester.py`
- Test: `hypo/statistic/tester.py --profile smoke`

- [ ] **Step 1: Replace the small check with the complete benchmark CLI**

Set `hypo/statistic/tester.py` to:

```python
from __future__ import annotations

import argparse
import csv
import gc
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ADP import ADP_Config
from hypo.statistic.ADP_single_index_hypo_stat import (
    ADP_single_index_hypo_stat,
)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    n: int
    d: int
    centers: int
    directions: int
    density: float


SMOKE_CASES = (
    BenchmarkCase("smoke", 256, 12, 16, 4, 0.10),
)
STANDARD_CASES = (
    BenchmarkCase("standard-sparse", 1000, 32, 64, 8, 0.05),
    BenchmarkCase("standard-medium", 1500, 64, 96, 8, 0.15),
    BenchmarkCase("standard-dense", 1000, 32, 64, 8, 0.75),
)
METHODS = ("direct", "matrix", "sparse")


def prepare_case(
    case: BenchmarkCase,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(case.n, case.d))
    beta = rng.normal(size=case.d)
    beta /= np.linalg.norm(beta)
    Y = np.sin(X @ beta) + 0.05 * rng.normal(size=case.n)
    center_indices = rng.choice(case.n, size=case.centers, replace=False)
    centers = X[center_indices]
    distance2 = (
        np.square(centers).sum(axis=1)[:, None]
        + np.square(X).sum(axis=1)[None, :]
        - 2.0 * centers @ X.T
    )
    np.maximum(distance2, 0.0, out=distance2)
    radius2 = max(
        float(np.quantile(distance2, case.density)),
        np.finfo(float).tiny,
    )
    scaled_distance = distance2 / radius2
    weights = np.maximum(1.0 - np.square(scaled_distance), 0.0)
    directions = rng.normal(
        size=(case.centers, case.directions, case.d)
    )
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    return X, Y, weights, directions


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(
        float(np.linalg.norm(expected.ravel())),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm((actual - expected).ravel()) / denominator)


def benchmark_method(
    model: ADP_single_index_hypo_stat,
    method: str,
    X: np.ndarray,
    Y: np.ndarray,
    weights: np.ndarray,
    directions: np.ndarray,
    reference: dict[str, np.ndarray],
    *,
    repetitions: int,
    batch_size: int,
    shift: float,
) -> dict[str, object]:
    method_weights = (
        sparse.csr_matrix(weights) if method == "sparse" else weights
    )

    def calculate(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        return model.Calculate_statistic(
            method_weights,
            directions,
            method=method,
            X=x,
            Y=y,
            batch_size=batch_size,
        )

    warm = calculate(X, Y)
    del warm
    timings: list[float] = []
    for _ in range(repetitions):
        gc.collect()
        started = time.perf_counter()
        result = calculate(X, Y)
        timings.append(time.perf_counter() - started)
        del result

    gc.collect()
    tracemalloc.start()
    result = calculate(X, Y)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    shifted = calculate(X + shift, Y + shift)
    record = {
        "method": method,
        "median_sec": statistics.median(timings),
        "min_sec": min(timings),
        "peak_alloc_mib": peak_bytes / (1024.0**2),
        "relative_I": relative_error(result["I"], reference["I"]),
        "relative_U": relative_error(result["U"], reference["U"]),
        "shift_relative_I": relative_error(shifted["I"], result["I"]),
        "shift_relative_U": relative_error(shifted["U"], result["U"]),
        "finite": int(
            all(
                np.all(np.isfinite(result[name]))
                for name in ("I", "U", "mass", "mean", "n_eff", "eta")
            )
        ),
        "eta_max": float(np.max(result["eta"])),
        "n_eff_min": float(np.min(result["n_eff"])),
    }
    del result, shifted
    return record


def run_case(
    case: BenchmarkCase,
    *,
    repetitions: int,
    seed: int,
    batch_size: int,
    shift: float,
    tolerance: float,
) -> list[dict[str, object]]:
    X, Y, weights, directions = prepare_case(case, seed)
    model = ADP_single_index_hypo_stat(
        ADP_Config(n=1, d=1, N_J=1, N_phi=1)
    )
    reference = model.Calculate_statistic(
        weights,
        directions,
        method="direct",
        X=X.astype(np.float64, copy=False),
        Y=Y.astype(np.float64, copy=False),
    )
    actual_density = float(np.count_nonzero(weights) / weights.size)
    records = []
    for method in METHODS:
        record = benchmark_method(
            model,
            method,
            X,
            Y,
            weights,
            directions,
            reference,
            repetitions=repetitions,
            batch_size=batch_size,
            shift=shift,
        )
        record = {
            "case": case.name,
            "n": case.n,
            "d": case.d,
            "centers": case.centers,
            "directions": case.directions,
            "target_density": case.density,
            "actual_density": actual_density,
            "seed": seed,
            "repetitions": repetitions,
            "batch_size": batch_size,
            "shift": shift,
            **record,
        }
        if record["method"] != "direct" and max(
            float(record["relative_I"]),
            float(record["relative_U"]),
        ) > tolerance:
            raise AssertionError(
                f"{case.name}/{method}: расхождение больше {tolerance:g}"
            )
        records.append(record)

    direct = records[0]
    for record in records:
        record["time_speedup_vs_direct"] = (
            float(direct["median_sec"]) / float(record["median_sec"])
        )
        record["memory_ratio_vs_direct"] = (
            float(record["peak_alloc_mib"])
            / float(direct["peak_alloc_mib"])
        )
    return records


def write_csv(records: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сравнение трёх реализаций статистик ADP."
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "standard"),
        default="smoke",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shift", type=float, default=1e8)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hypo/statistic/benchmark_results.csv"),
    )
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error("--repetitions должен быть положительным")
    if args.batch_size < 1:
        parser.error("--batch-size должен быть положительным")
    if args.tolerance <= 0:
        parser.error("--tolerance должен быть положительным")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = SMOKE_CASES if args.profile == "smoke" else STANDARD_CASES
    records = [
        record
        for case_number, case in enumerate(cases)
        for record in run_case(
            case,
            repetitions=args.repetitions,
            seed=args.seed + case_number,
            batch_size=args.batch_size,
            shift=args.shift,
            tolerance=args.tolerance,
        )
    ]
    write_csv(records, args.output)
    for record in records:
        print(
            f"{record['case']:>16} {record['method']:>6} "
            f"time={float(record['median_sec']):.6f}s "
            f"memory={float(record['peak_alloc_mib']):.2f}MiB "
            f"dI={float(record['relative_I']):.2e} "
            f"dU={float(record['relative_U']):.2e} "
            f"shiftU={float(record['shift_relative_U']):.2e}"
        )
    print(f"csv: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the smoke benchmark**

Run:

```bash
rtk python hypo/statistic/tester.py --profile smoke --repetitions 2 --output /tmp/adp-statistics-smoke.csv
```

Expected: three printed rows, one for each method, followed by:

```text
csv: /tmp/adp-statistics-smoke.csv
```

- [ ] **Step 3: Inspect the persisted smoke schema and correctness values**

Run:

```bash
rtk python -c "import csv; rows=list(csv.DictReader(open('/tmp/adp-statistics-smoke.csv', encoding='utf-8'))); assert len(rows)==3; assert {r['method'] for r in rows}=={'direct','matrix','sparse'}; assert all(int(r['finite'])==1 for r in rows); assert max(float(r['relative_U']) for r in rows)<1e-10; print('csv: ok')"
```

Expected:

```text
csv: ok
```

### Task 4: Implement the common hypothesis-tester CLI

**Files:**

- Modify: `hypo/tester.py`
- Test: delegated `statistic --help`

- [ ] **Step 1: Confirm no tester can currently be selected**

Run:

```bash
rtk python hypo/tester.py statistic --help
```

Expected: no useful output because the file is empty.

- [ ] **Step 2: Implement discovery and argument forwarding**

Set `hypo/tester.py` to:

```python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HYPO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = HYPO_ROOT.parent


def discover_testers() -> dict[str, Path]:
    return {
        path.parent.name: path
        for path in sorted(HYPO_ROOT.glob("*/tester.py"))
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    testers = discover_testers()
    parser = argparse.ArgumentParser(
        description="Запуск тестеров гипотез из hypo/<name>/tester.py."
    )
    parser.add_argument("--list", action="store_true")

    if argv and not argv[0].startswith("-"):
        name, forwarded = argv[0], argv[1:]
        if name not in testers:
            parser.error(
                f"неизвестный tester {name!r}; доступны: "
                + ", ".join(testers)
            )
        return subprocess.run(
            [sys.executable, str(testers[name]), *forwarded],
            cwd=REPO_ROOT,
            check=False,
        ).returncode

    args = parser.parse_args(argv)
    if args.list:
        for name in testers:
            print(name)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Verify discovery**

Run:

```bash
rtk python hypo/tester.py --list
```

Expected:

```text
statistic
```

- [ ] **Step 4: Verify delegated help**

Run:

```bash
rtk python hypo/tester.py statistic --help
```

Expected: child help containing `--profile`, `--repetitions`,
`--batch-size`, and `--output`.

- [ ] **Step 5: Verify the end-to-end smoke path through the common CLI**

Run:

```bash
rtk python hypo/tester.py statistic --profile smoke --repetitions 2 --output /tmp/adp-statistics-cli-smoke.csv
```

Expected: three benchmark rows and a zero exit status.

### Task 5: Fresh verification and standard comparison

**Files:**

- Verify: `ADP/__init__.py`
- Verify: `hypo/statistic/ADP_single_index_hypo_stat.py`
- Verify: `hypo/statistic/tester.py`
- Verify: `hypo/tester.py`
- Create: `hypo/statistic/benchmark_results.csv`

- [ ] **Step 1: Compile every changed Python file**

Run:

```bash
rtk python -m py_compile ADP/__init__.py hypo/statistic/ADP_single_index_hypo_stat.py hypo/statistic/tester.py hypo/tester.py
```

Expected: exit status zero and no output.

- [ ] **Step 2: Run the fresh smoke comparison**

Run:

```bash
rtk python hypo/tester.py statistic --profile smoke --repetitions 3 --output /tmp/adp-statistics-final-smoke.csv
```

Expected: all methods are finite and matrix/sparse cross-method errors remain
below `1e-10`.

- [ ] **Step 3: Run and persist the standard comparison**

Run:

```bash
rtk python hypo/tester.py statistic --profile standard --repetitions 5 --output hypo/statistic/benchmark_results.csv
```

Expected: nine rows covering three cases and three methods.

- [ ] **Step 4: Summarize the standard CSV**

Run:

```bash
rtk python -c "import csv; rows=list(csv.DictReader(open('hypo/statistic/benchmark_results.csv', encoding='utf-8'))); assert len(rows)==9; print('\\n'.join(f\"{r['case']} {r['method']}: {float(r['median_sec']):.6f}s, {float(r['peak_alloc_mib']):.2f}MiB, speedup={float(r['time_speedup_vs_direct']):.2f}x, memory={float(r['memory_ratio_vs_direct']):.2f}x, shiftU={float(r['shift_relative_U']):.2e}\" for r in rows))"
```

Expected: nine concise comparison lines with finite values.

- [ ] **Step 5: Check whitespace and inspect only task-owned changes**

Run:

```bash
rtk git diff --check
rtk git status --short
```

Expected: `git diff --check` succeeds. Status shows the pre-existing user
changes plus the task-owned ADP export, hypothesis implementation, testers,
and benchmark CSV. Do not stage the pre-existing user files.

- [ ] **Step 6: Report evidence without creating an implementation commit**

Report:

- exact smoke and standard commands;
- fastest method per density case;
- measured peak allocation ratios;
- direct/matrix/sparse relative errors;
- large-shift errors;
- the persisted CSV path.

An implementation commit is deliberately excluded because
`ADP/ADP_single_index.py`, which the new package export depends on, is an
untracked user-owned file. Committing only the export would leave a broken
fresh checkout, while staging that source file would absorb unrelated user
work.
