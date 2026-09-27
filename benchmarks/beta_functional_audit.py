# ruff: noqa: RUF002, RUF003
"""Изолированный reference и NumPy-аудит β-подзадачи; production не меняется."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import math
import os
import platform
import pstats
import resource
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import scipy

from ADP.solver import HYBRID, LSMR
from benchmarks.localization import environment


def compress_rows(U: np.ndarray, I: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """QR [U,I] без rank truncation; сохраняет норму любого локального residual.

    Применять только при P>d+1. Это reference представления; число строк
    исходной задачи требуется сохранить отдельно для старого SVD cutoff.
    """
    if U.shape[:2] != I.shape or U.shape[1] <= U.shape[2] + 1:
        raise ValueError("compression requires I=(J,P) and P>d+1")
    compressed = np.empty((U.shape[0], U.shape[2] + 1, U.shape[2] + 1))
    for j in range(len(U)):
        compressed[j] = np.linalg.qr(np.column_stack((U[j], I[j])), mode="r")
    return compressed[:, :, :-1], compressed[:, :, -1]


def design_without_temporary(
    U: np.ndarray, coefficients: np.ndarray, root: np.ndarray
) -> np.ndarray:
    """EXACT: писать broadcast product непосредственно в bounded design."""
    J, P, d = U.shape
    design = np.empty((J * P, coefficients.shape[1] * d))
    for a in range(coefficients.shape[1]):
        np.multiply(
            (root * coefficients[:, a])[:, None, None],
            U,
            out=design[:, a * d : (a + 1) * d].reshape(J, P, d),
        )
    return design


def stationarity_matmul(
    I: np.ndarray,
    U: np.ndarray,
    index: np.ndarray,
    coefficients: np.ndarray,
    mass: np.ndarray,
    loss: float,
    *,
    U_norm2: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Multi-only EXACT reference: сопряжённые matmul вместо contraction planner.

    Local numerator = B U_j.T r_j; projected остаётся для прежней norm scale.
    Все нормировки сертификата сохраняются.
    """
    residual = I - LSMR._predict(U, index, coefficients)
    pulled = (U.swapaxes(1, 2) @ residual[..., None]).squeeze(-1)
    gradient = -(mass[:, None] * coefficients).T @ pulled
    product = gradient @ index.T
    riemannian = gradient - (0.5 * (product + product.T)) @ index
    projected = U @ index.T
    local = -mass[:, None] * (pulled @ index.T)
    local_scale = np.maximum(
        1.0, np.linalg.norm(projected, axis=(1, 2)) * np.linalg.norm(I, axis=1)
    )
    local_score = float(np.max(np.linalg.norm(local, axis=1) / local_scale))
    if U_norm2 is None:
        U_norm2 = np.einsum("jpd,jpd->j", U, U, optimize=True)
    operator_norm2 = float(np.sum(mass * np.sum(coefficients**2, axis=1) * U_norm2))
    gradient_scale = max(1.0, math.sqrt(operator_norm2 * 2.0 * loss))
    orthogonality = float(np.linalg.norm(index @ index.T - np.eye(len(index))))
    return (
        float(np.linalg.norm(riemannian) / gradient_scale),
        local_score,
        orthogonality,
    )


def measure_pair(
    baseline: Callable, candidate: Callable, repeats: int
) -> dict[str, object]:
    """Парные медианы с чередованием порядка; allocation отдельно от wall."""
    baseline()
    candidate()
    samples: list[list[float]] = [[], []]
    for repeat in range(repeats):
        for k in (0, 1) if repeat % 2 == 0 else (1, 0):
            started = perf_counter()
            (baseline, candidate)[k]()
            samples[k].append(perf_counter() - started)
    peaks = []
    for method in (baseline, candidate):
        tracemalloc.start()
        method()
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    medians = [float(np.median(sample)) for sample in samples]
    return {
        "seconds": samples,
        "median_seconds": medians,
        "candidate_baseline_ratio": medians[1] / medians[0],
        "tracemalloc_peak_bytes_excluding_inputs": peaks,
    }


def profile(method: Callable) -> str:
    """Cumulative и self time без использования профиля как wall benchmark."""
    profiler = cProfile.Profile()
    profiler.runcall(method)
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs()
    stats.sort_stats("cumulative").print_stats(18)
    stats.sort_stats("tottime").print_stats(18)
    return stream.getvalue()


def audit_case(
    U: np.ndarray, I: np.ndarray, B: np.ndarray, mass: np.ndarray, repeats: int
) -> dict[str, object]:
    """Проверить kernels и фиксированную трёхшаговую HPAO trajectory."""
    C, _ = LSMR._local_refit(I, U, B)
    norm2 = np.einsum("jpd,jpd->j", U, U)
    loss = LSMR._loss(I, U, B, C, mass)

    def baseline():
        return LSMR._stationarity(I, U, B, C, mass, loss, U_norm2=norm2)

    def candidate():
        return stationarity_matmul(I, U, B, C, mass, loss, U_norm2=norm2)

    expected, actual = np.array(baseline()), np.array(candidate())
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
    row: dict[str, object] = {
        "shape_J_P_d_m": [*U.shape, len(B)],
        "stationarity_reference_absolute_error": float(np.max(abs(actual - expected))),
        "stationarity": measure_pair(baseline, candidate, repeats),
    }
    workspace = HYBRID.RidgeWorkspace(U, I, B, C, mass, max_unknowns=0)
    vector, data = B.ravel(), I.ravel()

    def actions_baseline():
        return workspace.matvec(vector), workspace.rmatvec(data)

    def actions_einsum():
        local = workspace._weighted_coefficients @ vector.reshape(B.shape)
        fitted = np.einsum("jpd,jd->jp", U, local, optimize=False).ravel()
        pulled = np.einsum("jpd,jp->jd", U, data.reshape(I.shape), optimize=False)
        return fitted, (workspace._weighted_coefficients.T @ pulled).ravel()

    for old, new in zip(actions_baseline(), actions_einsum(), strict=True):
        np.testing.assert_allclose(new, old, rtol=1e-10, atol=1e-12)
    row["actions_einsum_vs_matmul"] = measure_pair(
        actions_baseline, actions_einsum, repeats
    )
    root = np.sqrt(mass)
    if HYBRID.use_dense(I.size, B.size, 256, 64 * 1024**2):
        np.testing.assert_array_equal(
            HYBRID.design_matrix(U, C, root), design_without_temporary(U, C, root)
        )
        row["design"] = measure_pair(
            lambda: HYBRID.design_matrix(U, C, root),
            lambda: design_without_temporary(U, C, root),
            repeats,
        )

    def solve_baseline():
        return HYBRID.solve(B, U, I, mass=mass, lambda_prox=0.05, max_steps=3)

    def solve_candidate():
        with (
            patch.object(LSMR, "_stationarity", stationarity_matmul),
            patch.object(HYBRID, "design_matrix", design_without_temporary),
        ):
            return solve_baseline()

    ref, result = solve_baseline(), solve_candidate()
    np.testing.assert_allclose(result.index, ref.index, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(
        result.coefficients, ref.coefficients, rtol=1e-9, atol=1e-10
    )
    for key in ("accepted_steps", "rejected_trials", "linear_iterations_total"):
        if result.diagnostics[key] != ref.diagnostics[key]:
            raise AssertionError(f"trajectory differs: {key}")
    row["trajectory"] = {
        "measurement": measure_pair(solve_baseline, solve_candidate, repeats),
        "settings": {"lambda_prox": 0.05, "max_steps": 3, "tol": 1e-6},
        "diagnostics": ref.diagnostics,
        "max_index_absolute_error": float(np.max(abs(result.index - ref.index))),
        "max_coefficient_absolute_error": float(
            np.max(abs(result.coefficients - ref.coefficients))
        ),
    }
    row["profile_baseline"] = profile(solve_baseline)
    row["profile_candidate"] = profile(solve_candidate)
    # Проверка общего metric без хранения d×d: Frobenius inner product
    # <U_a.T U_a,U_b.T U_b> = ||U_a U_b.T||_F².
    active = np.flatnonzero(norm2 > 0)
    a, b = U[active[0]], U[active[1]]
    row["first_two_active_gram_cosine"] = float(
        np.sum((a @ b.T) ** 2) / (np.linalg.norm(a @ a.T) * np.linalg.norm(b @ b.T))
    )
    if U.shape[1] > U.shape[2] + 1:
        started = perf_counter()
        V, y = compress_rows(U, I)
        setup = perf_counter() - started
        inputs = (U, I, B, C, mass)
        compact_inputs = (V, y, B, C, mass)

        def correction(arrays):
            return HYBRID.RidgeWorkspace(*arrays).correction(1.0, 1e-6, None)

        ref_delta, compact_delta = correction(inputs), correction(compact_inputs)
        np.testing.assert_allclose(
            compact_delta[0], ref_delta[0], rtol=1e-9, atol=1e-11
        )
        row["row_compression"] = {
            "shape": list(V.shape),
            "setup_seconds": setup,
            "input_bytes": U.nbytes + I.nbytes,
            "compressed_bytes": V.nbytes + y.nbytes,
            "correction_absolute_error": float(
                np.max(abs(compact_delta[0] - ref_delta[0]))
            ),
            "normal_residual_ratios": [ref_delta[3], compact_delta[3]],
            "workspace_and_one_correction": measure_pair(
                lambda: correction(inputs), lambda: correction(compact_inputs), repeats
            ),
        }
    return row


def main() -> None:
    """Frozen данные для профиля, не новая selection/validation солвера."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--frozen-dir",
        type=Path,
        default=Path("benchmark_outputs/diagnostic/multi_solver_search_s1_20260924"),
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.output.exists():
        parser.error("positive repeats and a new output file are required")
    if any(
        os.environ.get(k) != "1"
        for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    ):
        parser.error("pin OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=OMP_NUM_THREADS=1")
    manifest = json.loads((args.frozen_dir / "manifest.json").read_text())
    rows = []
    for fit in manifest["fits"]:
        # Все шесть datasets, только первый snapshot для bounded profile.
        call = next(c for c in fit["calls"] if "file" in c)
        file = args.frozen_dir / call["file"]
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        if digest != call["sha256"]:
            raise RuntimeError(f"frozen hash mismatch: {file}")
        with np.load(file, allow_pickle=False) as arrays:
            U, I, B, mass = (arrays[k] for k in ("U", "I", "B", "mass"))
        row = audit_case(U, I, B, mass, args.repeats)
        row.update(file=str(file), sha256=digest)
        rows.append(row)
        print(json.dumps({"case": file.name, "completed": True}), flush=True)
    rng = np.random.default_rng(71000)
    for d, m in ((1000, 2), (1000, 10)):
        U, I = rng.normal(size=(100, 20, d)), rng.normal(size=(100, 20))
        B = np.linalg.qr(rng.normal(size=(d, m)))[0].T
        C = rng.normal(size=(100, m))
        mass = np.geomspace(0.01, 100, 100)
        loss = LSMR._loss(I, U, B, C, mass)
        args0 = (I, U, B, C, mass, loss)
        np.testing.assert_allclose(
            stationarity_matmul(*args0),
            LSMR._stationarity(*args0),
            rtol=1e-9,
            atol=1e-12,
        )
        rows.append(
            {
                "synthetic_seed": 71000,
                "shape_J_P_d_m": [100, 20, d, m],
                "stationarity": measure_pair(
                    lambda args0=args0: LSMR._stationarity(*args0),
                    lambda args0=args0: stationarity_matmul(*args0),
                    args.repeats,
                ),
            }
        )
    report = environment() | {
        "python_actual": platform.python_version(),
        "scipy": scipy.__version__,
        "dtype": "float64",
        "repeats": args.repeats,
        "results": rows,
        "memory_method": (
            "separate tracemalloc; process ru_maxrss high-water "
            "includes inputs and all cases"
        ),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        "source_sha256": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (
                "ADP/solver/LSMR.py",
                "ADP/solver/HYBRID/HYBRID.py",
                "ADP/solver/HYBRID/HYBRID_multi.py",
                "ADP/solver/HYBRID/HYBRID_manifold.py",
                __file__,
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
