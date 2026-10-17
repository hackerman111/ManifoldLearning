"""One fresh-process localization comparison; no fit loop or estimator edits."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, vstack
from threadpoolctl import threadpool_info, threadpool_limits

from ADP.core.ADP_Config import epanechnikov
from ADP.engine.common.calculus import calculate_alpha_k, pairwise_distance2
from ADP.engine.common.experimental_geometry import projected_components
from ADP.engine.common.weights import calculate_multi_weight

CASES = {
    "small": {"n": 500, "d": 30, "J": 500, "m": 2, "h": 1.0},
    "large": {"n": 10_000, "d": 100, "J": 500, "m": 2, "h": 0.4},
}
EIGENVALUES = np.array([1.0, 0.25])
ALPHA = 0.02
TARGET = 20


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _arrays(seed: int, n: int, d: int, J: int, m: int):
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n, d))
    X = np.empty_like(white)
    X[:, 0] = white[:, 0]
    scale = np.sqrt(1.0 - 0.5**2)
    for k in range(1, d):
        X[:, k] = 0.5 * X[:, k - 1] + scale * white[:, k]
    centers = X[np.linspace(0, n - 1, J, dtype=int)].copy()
    basis, _ = np.linalg.qr(rng.standard_normal((d, m)), mode="reduced")
    return X, centers, basis


def _weights(X, centers, basis, eigenvalues, h, alpha, distance2, arm):
    if arm == "projected":
        geometry = projected_components(X, centers, basis, eigenvalues, h, distance2)
        return geometry.weights(alpha), geometry
    blocks = [
        csr_matrix(block)
        for _, block in calculate_multi_weight(
            X,
            centers,
            basis,
            eigenvalues,
            h,
            alpha,
            epanechnikov,
            distance2=distance2,
        )
    ]
    return csr_matrix(vstack(blocks, format="csr")), None


def _alpha(X, centers, basis, eigenvalues, h, distance2, arm):
    if arm == "projected":
        geometry = projected_components(X, centers, basis, eigenvalues, h, distance2)
        return geometry.alpha(TARGET), geometry
    return (
        calculate_alpha_k(
            X,
            centers,
            basis,
            eigenvalues,
            h,
            TARGET,
            epanechnikov,
            distance2=distance2,
        ),
        None,
    )


def _sparse_stats(matrix: csr_matrix) -> dict:
    return {
        "nnz": int(matrix.nnz),
        "csr_bytes": int(
            matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        ),
        "dense_equivalent_bytes": int(np.prod(matrix.shape) * np.dtype(float).itemsize),
    }


def run(case: str, seed: int, mode: str, arm: str, output: Path) -> None:
    spec = CASES[case]
    n, d, J, m, h = (spec[key] for key in ("n", "d", "J", "m", "h"))
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        if not pools or any(pool.get("num_threads") != 1 for pool in pools):
            raise RuntimeError(f"one-thread verification failed: {pools}")
        start = time.perf_counter()
        X, centers, basis = _arrays(seed, n, d, J, m)
        input_seconds = time.perf_counter() - start
        start = time.perf_counter()
        distance2 = pairwise_distance2(X, centers)
        distance_seconds = time.perf_counter() - start
        hashes = {
            "X": _hash(X),
            "centers": _hash(centers),
            "basis": _hash(basis),
            "eigenvalues": _hash(EIGENVALUES),
            "distance2": _hash(distance2),
        }
        if mode == "weights":
            start = time.perf_counter()
            actual, geometry = _weights(
                X, centers, basis, EIGENVALUES, h, ALPHA, distance2, arm
            )
            localization_seconds = time.perf_counter() - start
            alpha = ALPHA
            alpha_seconds = 0.0
        else:
            start = time.perf_counter()
            alpha, geometry = _alpha(X, centers, basis, EIGENVALUES, h, distance2, arm)
            alpha_seconds = time.perf_counter() - start
            if alpha is None:
                actual = csr_matrix((J, n), dtype=float)
            else:
                start = time.perf_counter()
                if arm == "projected":
                    assert geometry is not None
                    actual = geometry.weights(alpha)
                else:
                    actual, _ = _weights(
                        X, centers, basis, EIGENVALUES, h, alpha, distance2, arm
                    )
                weight_seconds = time.perf_counter() - start
            localization_seconds = alpha_seconds + locals().get("weight_seconds", 0.0)
        peak_after_timing = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if mode == "weights":
            reference, _ = _weights(
                X,
                centers,
                basis,
                EIGENVALUES,
                h,
                ALPHA,
                distance2,
                "projected" if arm == "baseline" else "baseline",
            )
            alpha_error = 0.0
        else:
            reference_alpha, reference_geometry = _alpha(
                X,
                centers,
                basis,
                EIGENVALUES,
                h,
                distance2,
                "projected" if arm == "baseline" else "baseline",
            )
            alpha_error = (
                None
                if alpha is None or reference_alpha is None
                else abs(float(alpha) - float(reference_alpha))
            )
            if reference_alpha is None:
                reference = csr_matrix((J, n), dtype=float)
            elif reference_geometry is not None:
                reference = reference_geometry.weights(reference_alpha)
            else:
                reference, _ = _weights(
                    X,
                    centers,
                    basis,
                    EIGENVALUES,
                    h,
                    reference_alpha,
                    distance2,
                    "baseline",
                )
        delta = (actual - reference).tocsr()
        delta.eliminate_zeros()
        actual_support, reference_support = actual.copy(), reference.copy()
        actual_support.data[:] = 1.0
        reference_support.data[:] = 1.0
        support_delta = (actual_support - reference_support).tocsr()
        support_delta.eliminate_zeros()
        record = {
            "case": case,
            "seed": seed,
            "mode": mode,
            "arm": arm,
            "shape": {"n": n, "d": d, "J": J, "m": m, "h": h},
            "alpha_target": TARGET if mode == "alpha_weights" else None,
            "alpha": alpha,
            "alpha_error": alpha_error,
            "input_generation_sec": input_seconds,
            "distance2_build_sec": distance_seconds,
            "localization_plus_output_sec": localization_seconds,
            "weight_only_sec": localization_seconds - alpha_seconds,
            "alpha_search_sec": alpha_seconds,
            "max_weight_abs_error": float(
                np.max(np.abs(delta.data)) if delta.nnz else 0.0
            ),
            "weight_error_nonzero_count": int(delta.nnz),
            "support_mismatches": int(support_delta.nnz),
            "actual_storage": _sparse_stats(actual),
            "reference_storage": _sparse_stats(reference),
            "peak_rss_after_timing_kib": peak_after_timing,
            "peak_rss_after_reference_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
            "hashes": hashes,
            "threadpools": pools,
            "thread_env": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("weights", "alpha_weights"), required=True)
    parser.add_argument("--arm", choices=("baseline", "projected"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.case, args.seed, args.mode, args.arm, args.output)
