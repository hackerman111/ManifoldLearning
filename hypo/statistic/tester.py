from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import platform
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)
for variable in THREAD_ENV_VARS:
    os.environ.setdefault(variable, "1")

import numpy as np
import scipy
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hypo.statistic.ADP_single_index_hypo_stat import ADP_single_index_hypo_stat


LOGGER = logging.getLogger("adp.statistics")


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    n: int
    d: int
    centers: int
    directions: int
    density: float


SMOKE_CASES = (BenchmarkCase("smoke", 256, 12, 16, 4, 0.10),)
STANDARD_CASES = (
    BenchmarkCase("standard-sparse", 2000, 32, 64, 8, 0.05),
    BenchmarkCase("standard-medium", 2500, 1000, 96, 8, 0.15),
    BenchmarkCase("standard-dense", 2000, 32, 64, 8, 0.75),
)
SCALING_CASES = tuple(
    BenchmarkCase(
        f"scaling-d{d}-r{n_over_d}",
        d * n_over_d,
        d,
        64,
        10,
        0.15,
    )
    for d in (50, 100, 1000)
    for n_over_d in (2, 5, 10)
)
STABILITY_CASE = BenchmarkCase("random-stability", 500, 100, 64, 10, 0.15)
METHODS = ("direct", "matrix", "sparse")
RESULT_FIELDS = ("I", "U", "mass", "mean", "n_eff", "eta")
STABILITY_FIELDS = ("I", "U", "mass", "mean", "n_eff")


def method_order(order_offset: int) -> tuple[str, ...]:
    offset = order_offset % len(METHODS)
    return METHODS[offset:] + METHODS[:offset]


def cpu_model() -> str:
    machine = platform.machine() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip() or machine
    except OSError:
        pass
    return machine


def prepare_case(
    case: BenchmarkCase, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    direct_tensor_mib = (
        case.centers * case.n * case.d * np.dtype(float).itemsize / 1024**2
    )
    LOGGER.info(
        "Подготовка %s: d=%d, n=%d, n/d=%g, J=%d, P=%d, "
        "плотность=%.0f%%, direct-тензор≈%.2f MiB",
        case.name,
        case.d,
        case.n,
        case.n / case.d,
        case.centers,
        case.directions,
        100 * case.density,
        direct_tensor_mib,
    )
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(case.n, case.d))
    beta = rng.normal(size=case.d)
    beta /= np.linalg.norm(beta)
    Y = np.sin(X @ beta) + 0.05 * rng.normal(size=case.n)
    centers = X[rng.choice(case.n, size=case.centers, replace=False)]
    distance2 = (
        np.square(centers).sum(axis=1)[:, None]
        + np.square(X).sum(axis=1)[None, :]
        - 2.0 * centers @ X.T
    )
    np.maximum(distance2, 0.0, out=distance2)
    radius2 = max(float(np.quantile(distance2, case.density)), np.finfo(float).tiny)
    weights = np.maximum(1.0 - np.square(distance2 / radius2), 0.0)
    directions = rng.normal(size=(case.centers, case.directions, case.d))
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    LOGGER.info(
        "Данные %s готовы: фактическая плотность весов %.2f%%",
        case.name,
        100 * np.count_nonzero(weights) / weights.size,
    )
    return X, Y, weights, directions


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        return float("inf")
    scale = max(
        float(np.max(np.abs(actual), initial=0.0)),
        float(np.max(np.abs(expected), initial=0.0)),
        np.finfo(float).tiny,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled_actual = actual.ravel() / scale
        scaled_expected = expected.ravel() / scale
        error = np.linalg.norm(scaled_actual - scaled_expected) / max(
            float(np.linalg.norm(scaled_expected)), np.finfo(float).tiny
        )
    return float(error) if np.isfinite(error) else float("inf")


def correctness_check() -> None:
    np.testing.assert_allclose(
        relative_error(np.array([1.1e200]), np.array([1e200])),
        0.1,
        rtol=1e-14,
    )
    assert np.isinf(relative_error(np.array([np.inf]), np.array([1.0])))
    standard_orders = [method_order(offset) for offset in range(len(STANDARD_CASES))]
    for method in METHODS:
        assert sorted(order.index(method) for order in standard_orders) == [0, 1, 2]
    scaling_orders = [method_order(offset) for offset in range(len(SCALING_CASES))]
    for method in METHODS:
        assert sorted(order.index(method) for order in scaling_orders) == sorted(
            [0, 1, 2] * 3
        )

    rng = np.random.default_rng(20260731)
    X = rng.normal(size=(24, 5))
    Y = rng.normal(size=24)
    weights = rng.uniform(0.1, 1.0, size=(4, 24))
    weights[rng.random(weights.shape) < 0.35] = 0.0
    directions = rng.normal(size=(4, 3, 5))
    model = object.__new__(ADP_single_index_hypo_stat)
    direct = model.Calculate_statistic(weights, directions, method="direct", X=X, Y=Y)
    for method in ("matrix", "sparse"):
        method_weights = sparse.csr_matrix(weights) if method == "sparse" else weights
        result = model.Calculate_statistic(
            method_weights,
            directions,
            method=method,
            X=X,
            Y=Y,
            batch_size=2,
        )
        for field in RESULT_FIELDS:
            np.testing.assert_allclose(
                result[field], direct[field], rtol=1e-12, atol=1e-12
            )

    scale_weights = np.array([[1e200] * 4, [1e-200] * 4])
    scale_X = rng.normal(size=(4, 5))
    scale_Y = rng.normal(size=4)
    scale_directions = rng.normal(size=(2, 1, 5))
    for method in METHODS:
        method_weights = (
            sparse.csr_matrix(scale_weights) if method == "sparse" else scale_weights
        )
        result = model.Calculate_statistic(
            method_weights,
            scale_directions,
            method=method,
            X=scale_X,
            Y=scale_Y,
            batch_size=1,
        )
        np.testing.assert_allclose(result["n_eff"], 4.0, rtol=1e-15)


def benchmark_method(
    model: ADP_single_index_hypo_stat,
    method: str,
    X: np.ndarray,
    Y: np.ndarray,
    method_weights: object,
    directions: np.ndarray,
    reference: dict[str, np.ndarray],
    *,
    repetitions: int,
    batch_size: int,
    shift: float,
    warmed: bool = False,
) -> dict[str, object]:
    with np.errstate(over="ignore", invalid="ignore"):
        shifted_X = X + shift
        shifted_Y = Y + shift

    def calculate(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        return model.Calculate_statistic(
            method_weights,
            directions,
            method=method,
            X=x,
            Y=y,
            batch_size=batch_size,
        )

    LOGGER.info(
        "Метод %s: %s, затем %d измерений времени",
        method,
        "reference уже выполнил прогрев" if warmed else "прогрев",
        repetitions,
    )
    if not warmed:
        warmup = calculate(X, Y)
        del warmup
    timings = []
    for _ in range(repetitions):
        gc.collect()
        started = time.perf_counter()
        result = calculate(X, Y)
        timings.append(time.perf_counter() - started)
        del result

    gc.collect()
    tracemalloc.start()
    try:
        result = calculate(X, Y)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        shifted = calculate(shifted_X, shifted_Y)
    relative_I = relative_error(result["I"], reference["I"])
    relative_U = relative_error(result["U"], reference["U"])
    shift_relative_I = relative_error(shifted["I"], result["I"])
    shift_relative_U = relative_error(shifted["U"], result["U"])
    relative_values = (
        relative_I,
        relative_U,
        shift_relative_I,
        shift_relative_U,
    )
    record = {
        "method": method,
        "median_sec": statistics.median(timings),
        "min_sec": min(timings),
        "peak_alloc_mib": peak_bytes / (1024.0**2),
        "relative_I": relative_I,
        "relative_U": relative_U,
        "shift_relative_I": shift_relative_I,
        "shift_relative_U": shift_relative_U,
        "finite": int(
            all(
                np.all(np.isfinite(output[field]))
                for output in (result, shifted)
                for field in RESULT_FIELDS
            )
            and all(np.isfinite(value) for value in relative_values)
        ),
        "eta_max": float(np.max(result["eta"])),
        "n_eff_min": float(np.min(result["n_eff"])),
    }
    LOGGER.info(
        "Метод %s завершён: медиана %.6f s, пик аллокаций %.2f MiB, "
        "ошибка I %.2e, ошибка U %.2e",
        method,
        record["median_sec"],
        record["peak_alloc_mib"],
        record["relative_I"],
        record["relative_U"],
    )
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
    order_offset: int,
) -> list[dict[str, object]]:
    X, Y, weights, directions = prepare_case(case, seed)
    csr_weights = sparse.csr_matrix(weights)
    method_weights = {
        "direct": weights,
        "matrix": weights,
        "sparse": csr_weights,
    }
    model = object.__new__(ADP_single_index_hypo_stat)
    reference = model.Calculate_statistic(
        weights,
        directions,
        method="direct",
        X=X.astype(np.float64, copy=False),
        Y=Y.astype(np.float64, copy=False),
    )
    actual_density = float(np.count_nonzero(weights) / weights.size)
    execution_order = method_order(order_offset)
    execution_order_text = ",".join(execution_order)
    LOGGER.info("Сценарий %s: порядок методов %s", case.name, execution_order_text)
    blas = (
        getattr(np.__config__, "CONFIG", {})
        .get("Build Dependencies", {})
        .get("blas", {})
        or {}
    )
    runtime_metadata = {
        "memory_metric": "tracemalloc_peak_allocations_excluding_prepared_inputs",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform() or "unknown",
        "machine": platform.machine() or "unknown",
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count() or "unknown",
        "blas_name": blas.get("name") or "unknown",
        "blas_version": blas.get("version") or "unknown",
        **{name: os.environ.get(name) or "unset" for name in THREAD_ENV_VARS},
    }
    records = []
    for order_index, method in enumerate(execution_order):
        result = benchmark_method(
            model,
            method,
            X,
            Y,
            method_weights[method],
            directions,
            reference,
            repetitions=repetitions,
            batch_size=batch_size,
            shift=shift,
            warmed=method == "direct",
        )
        record = {
            "case": case.name,
            "n": case.n,
            "d": case.d,
            "n_over_d": case.n / case.d,
            "centers": case.centers,
            "directions": case.directions,
            "target_density": case.density,
            "actual_density": actual_density,
            "seed": seed,
            "repetitions": repetitions,
            "batch_size": batch_size,
            "shift": shift,
            "order_index": order_index,
            "execution_order": execution_order_text,
            **runtime_metadata,
            **result,
        }
        if method != "direct":
            relative_values = (
                float(record["relative_I"]),
                float(record["relative_U"]),
            )
            if (
                not all(np.isfinite(value) for value in relative_values)
                or max(relative_values) > tolerance
            ):
                raise AssertionError(
                    f"{case.name}/{method}: relative error exceeds {tolerance:g}"
                )
        records.append(record)

    direct = next(record for record in records if record["method"] == "direct")
    for record in records:
        record["time_speedup_vs_direct"] = float(direct["median_sec"]) / float(
            record["median_sec"]
        )
        record["memory_ratio_vs_direct"] = float(record["peak_alloc_mib"]) / float(
            direct["peak_alloc_mib"]
        )
    return records


def run_random_stability(
    *,
    runs: int,
    seed: int,
    batch_size: int,
    tolerance: float,
) -> list[dict[str, object]]:
    model = object.__new__(ADP_single_index_hypo_stat)
    records = []
    for run in range(runs):
        run_seed = seed + run
        X, Y, weights, directions = prepare_case(STABILITY_CASE, run_seed)

        started = time.perf_counter()
        direct = model.Calculate_statistic(
            weights, directions, method="direct", X=X, Y=Y
        )
        direct_sec = time.perf_counter() - started

        started = time.perf_counter()
        matrix = model.Calculate_statistic(
            weights,
            directions,
            method="matrix",
            X=X,
            Y=Y,
            batch_size=batch_size,
        )
        matrix_sec = time.perf_counter() - started

        errors = {
            f"relative_{field}": relative_error(matrix[field], direct[field])
            for field in STABILITY_FIELDS
        }
        finite = all(
            np.all(np.isfinite(result[field]))
            for result in (direct, matrix)
            for field in RESULT_FIELDS
        ) and all(np.isfinite(error) for error in errors.values())
        max_error = max(errors.values())
        informative_eta = direct["n_eff"] > 1.0 + np.sqrt(np.finfo(float).eps)
        records.append(
            {
                "run": run + 1,
                "seed": run_seed,
                "n": STABILITY_CASE.n,
                "d": STABILITY_CASE.d,
                "n_over_d": STABILITY_CASE.n / STABILITY_CASE.d,
                "centers": STABILITY_CASE.centers,
                "directions": STABILITY_CASE.directions,
                "target_density": STABILITY_CASE.density,
                "actual_density": np.count_nonzero(weights) / weights.size,
                "batch_size": batch_size,
                "direct_sec": direct_sec,
                "matrix_sec": matrix_sec,
                **errors,
                "eta_direct_max": np.max(direct["eta"]),
                "eta_matrix_max": np.max(matrix["eta"]),
                "eta_direct_max_n_eff_gt_1": np.max(
                    direct["eta"][informative_eta], initial=0.0
                ),
                "eta_matrix_max_n_eff_gt_1": np.max(
                    matrix["eta"][informative_eta], initial=0.0
                ),
                "single_point_centers": np.count_nonzero(~informative_eta),
                "max_relative_error": max_error,
                "finite": int(finite),
                "within_tolerance": int(finite and max_error <= tolerance),
            }
        )
        if (run + 1) % 5 == 0 or run + 1 == runs:
            LOGGER.info(
                "Случайные данные: выполнено %d/%d, текущая максимальная "
                "относительная ошибка %.2e",
                run + 1,
                runs,
                max_error,
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
        description="Benchmark the direct, matrix, and sparse ADP statistics."
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "standard", "scaling", "stability"),
        default="smoke",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--random-runs", type=int, default=50)
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
        parser.error("--repetitions must be positive")
    if args.random_runs < 1:
        parser.error("--random-runs must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not np.isfinite(args.shift):
        parser.error("--shift must be finite")
    if not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    LOGGER.info(
        "Запуск профиля %s: повторений=%d, seed=%d, batch_size=%d",
        args.profile,
        args.repetitions,
        args.seed,
        args.batch_size,
    )
    if args.profile == "stability":
        records = run_random_stability(
            runs=args.random_runs,
            seed=args.seed,
            batch_size=args.batch_size,
            tolerance=args.tolerance,
        )
        write_csv(records, args.output)
        errors = np.array([record["max_relative_error"] for record in records])
        failed = sum(not record["within_tolerance"] for record in records)
        print(
            f"случайных_запусков={len(records)} "
            f"медианная_ошибка={np.median(errors):.2e} "
            f"p95_ошибки={np.quantile(errors, 0.95):.2e} "
            f"максимальная_ошибка={np.max(errors):.2e} "
            f"неуспешных={failed}"
        )
        print(f"csv: {args.output}")
        return int(failed != 0)
    if args.profile == "smoke":
        correctness_check()
        cases = SMOKE_CASES
    elif args.profile == "scaling":
        cases = SCALING_CASES
    else:
        cases = STANDARD_CASES
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
            order_offset=case_number,
        )
    ]
    write_csv(records, args.output)
    LOGGER.info("Записано %d строк в %s", len(records), args.output)
    for record in records:
        print(
            f"{record['case']:>16} {record['method']:>6} "
            f"время={float(record['median_sec']):.6f}s "
            f"пик_аллокаций={float(record['peak_alloc_mib']):.2f}MiB "
            f"ошибка_I={float(record['relative_I']):.2e} "
            f"ошибка_U={float(record['relative_U']):.2e} "
            f"ошибка_I_при_сдвиге={float(record['shift_relative_I']):.2e} "
            f"ошибка_U_при_сдвиге={float(record['shift_relative_U']):.2e}"
        )
    print(f"csv: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
