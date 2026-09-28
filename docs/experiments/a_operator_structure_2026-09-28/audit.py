"""Bounded read-only audit of the frozen HPAO correction operator."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from pathlib import Path

import numpy as np
import scipy

from ADP.solver import LSMR

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FROZEN = ROOT / "benchmark_outputs/diagnostic/multi_solver_search_s1_20260924"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_hashes(value: object) -> dict[str, str]:
    """Collect manifest hashes without depending on its nested report layout."""
    found: dict[str, str] = {}
    if isinstance(value, dict):
        if isinstance(value.get("file"), str) and isinstance(value.get("sha256"), str):
            found[value["file"]] = value["sha256"]
        for child in value.values():
            found.update(frozen_hashes(child))
    elif isinstance(value, list):
        for child in value:
            found.update(frozen_hashes(child))
    return found


def gram_from_local(U: np.ndarray, C: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Diagnostic-only md-by-md A.T A from its local Kronecker summands."""
    _, _, d = U.shape
    m = C.shape[1]
    gram = np.zeros((m, d, m, d), dtype=np.float64)
    for u, c, weight in zip(U, C, mass, strict=True):
        local = u.T @ u
        gram += weight * np.einsum("a,b,de->adbe", c, c, local)
    flat = gram.reshape(m * d, m * d)
    return 0.5 * (flat + flat.T)


def structure_residual(matrix: np.ndarray, *, cyclic: bool) -> float:
    """Relative Frobenius distance to the closest circulant or Toeplitz matrix."""
    n = len(matrix)
    row, col = np.indices(matrix.shape)
    lag = (row - col) % n if cyclic else row - col + n - 1
    count = np.bincount(lag.ravel())
    sums = np.bincount(lag.ravel(), weights=matrix.ravel())
    average = sums / count
    return float(np.linalg.norm(matrix - average[lag]) / np.linalg.norm(matrix))


def tiny_check() -> dict[str, float]:
    """Independent explicit-design comparison on a deterministic tiny problem."""
    rng = np.random.default_rng(917)
    U = rng.normal(size=(3, 4, 5))
    C = rng.normal(size=(3, 2))
    mass = rng.uniform(0.2, 2.0, size=3)
    root = np.sqrt(mass)
    explicit = np.vstack(
        [root[j] * np.hstack([C[j, a] * U[j] for a in range(2)]) for j in range(3)]
    )
    operator = LSMR._linear_operator(U, C, root, (2, 5))
    x = rng.normal(size=10)
    y = rng.normal(size=12)
    errors = {
        "forward": float(np.linalg.norm(operator @ x - explicit @ x)),
        "adjoint": float(np.linalg.norm(operator.rmatvec(y) - explicit.T @ y)),
        "gram": float(
            np.linalg.norm(gram_from_local(U, C, mass) - explicit.T @ explicit)
        ),
    }
    assert max(errors.values()) < 1e-11, errors
    return errors


def one(path: Path, expected_hash: str) -> dict[str, object]:
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(f"frozen input hash changed: {path.name}")
    with np.load(path, allow_pickle=False) as data:
        U, I, B, mass = (data[name] for name in ("U", "I", "B", "mass"))
    C, local_ranks = LSMR._local_refit(I, U, B)
    root = np.sqrt(mass)
    operator = LSMR._linear_operator(U, C, root, B.shape)
    gram = gram_from_local(U, C, mass)
    eigenvalues, vectors = np.linalg.eigh(gram)
    eigmax = float(eigenvalues[-1])
    eigmin = float(eigenvalues[0])
    singular_min = math.sqrt(max(0.0, eigmin))
    singular_max = math.sqrt(max(0.0, eigmax))
    relative_singular_min = singular_min / singular_max if singular_max else 0.0

    # A full-rank conclusion requires a positive eigenvalue well above the
    # absolute Gram-roundoff scale; this is not an exact-rank oracle.
    gram_roundoff = 256 * np.finfo(float).eps * len(eigenvalues) * eigmax
    certified_by_gram = bool(eigmin > gram_roundoff)
    rank_1e8 = int(np.count_nonzero(eigenvalues > 1e-16 * eigmax))
    rank_1e6 = int(np.count_nonzero(eigenvalues > 1e-12 * eigmax))
    separated_1e8 = 1 + int(np.count_nonzero(np.diff(eigenvalues) > 1e-8 * eigmax))
    separated_1e6 = 1 + int(np.count_nonzero(np.diff(eigenvalues) > 1e-6 * eigmax))

    residual = (root[:, None] * (I - LSMR._predict(U, B, C))).ravel()
    rhs = operator.rmatvec(residual)
    rhs_eigen = vectors.T @ rhs
    rhs_support_1e8 = int(
        np.count_nonzero(np.abs(rhs_eigen) > 1e-8 * np.linalg.norm(rhs))
    )
    trial = np.random.default_rng(14).normal(size=B.size)
    gram_action_error = float(
        np.linalg.norm(gram @ trial - operator.rmatvec(operator @ trial))
        / max(1.0, np.linalg.norm(gram @ trial))
    )
    assert gram_action_error < 1e-10, (path.name, gram_action_error)

    # Two high-energy local metrics test for a common orthogonal eigenbasis.
    energy = mass * np.einsum("jpd,jpd->j", U, U)
    j1, j2 = np.argsort(energy)[-2:]
    h1, h2 = U[j1].T @ U[j1], U[j2].T @ U[j2]
    commutator = float(
        np.linalg.norm(h1 @ h2 - h2 @ h1) / (np.linalg.norm(h1) * np.linalg.norm(h2))
    )

    correction, stop, iterations, normal_ratio, _ = LSMR._global_correction(
        I, U, B, C, mass, 0.05, 1e-6, None
    )
    direct = np.linalg.solve(gram + 0.05 * np.eye(B.size), rhs)
    direct_relative_error = float(
        np.linalg.norm(correction - direct) / max(1.0, np.linalg.norm(direct))
    )
    explicit_singular_ratio = None
    if path.name == "n1000-seed1000-outer0.npz":
        # Independent SVD cross-check for one d100 case, bounded to 32 MB.
        explicit = np.vstack(
            [
                root[j] * np.hstack([C[j, a] * U[j] for a in range(len(B))])
                for j in range(len(U))
            ]
        )
        singular = np.linalg.svd(explicit, compute_uv=False)
        explicit_singular_ratio = float(singular[-1] / singular[0])
        assert abs(explicit_singular_ratio - relative_singular_min) < 1e-10

    return {
        "file": path.name,
        "sha256": actual_hash,
        "shape": {"J": len(U), "P": U.shape[1], "d": U.shape[2], "m": len(B)},
        "coefficient_rank": int(np.linalg.matrix_rank(C)),
        "local_projected_rank_min": int(local_ranks.min()),
        "rank_at_relative_singular_1e8": rank_1e8,
        "rank_at_relative_singular_1e6": rank_1e6,
        "full_column_rank_resolved_by_gram": certified_by_gram,
        "gram_roundoff_bound": float(gram_roundoff),
        "gram_min_eigenvalue": eigmin,
        "gram_max_eigenvalue": eigmax,
        "relative_min_singular": relative_singular_min,
        "independent_explicit_svd_relative_min_singular": explicit_singular_ratio,
        "normal_condition_ridge_0p05": float((eigmax + 0.05) / (eigmin + 0.05)),
        "gram_circulant_relative_residual": structure_residual(gram, cyclic=True),
        "gram_toeplitz_relative_residual": structure_residual(gram, cyclic=False),
        "local_metric_commutator_relative": commutator,
        "resolved_eigenvalue_clusters_1e8": separated_1e8,
        "resolved_eigenvalue_clusters_1e6": separated_1e6,
        "rhs_eigencomponents_1e8": rhs_support_1e8,
        "gram_action_relative_error": gram_action_error,
        "lsmr_stop": int(stop),
        "lsmr_iterations": int(iterations),
        "lsmr_normal_residual_ratio": float(normal_ratio),
        "lsmr_vs_normal_solve_relative_error": direct_relative_error,
    }


def main() -> None:
    manifest_path = FROZEN / "manifest.json"
    hashes = frozen_hashes(json.loads(manifest_path.read_text()))
    files = [
        FROZEN / f"{point}-seed{seed}-outer0.npz"
        for point in ("d10", "n1000")
        for seed in (1000, 1001, 1002)
    ]
    result = {
        "metadata": {
            "purpose": "operator structure and rank; no production change",
            "dtype": "float64",
            "lambda_prox": 0.05,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "threads": {
                key: os.environ.get(key)
                for key in (
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_NUM_THREADS",
                )
            },
            "manifest_sha256": sha256(manifest_path),
            "live_solver_sha256": sha256(ROOT / "ADP/solver/LSMR.py"),
            "live_multi_operator_sha256": sha256(
                ROOT / "ADP/solver/_multi_operator.py"
            ),
        },
        "tiny_check": tiny_check(),
        "cases": [one(path, hashes[path.name]) for path in files],
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    for case in result["cases"]:
        print(
            case["file"],
            "rank",
            case["rank_at_relative_singular_1e8"],
            "min/max sigma",
            f"{case['relative_min_singular']:.3g}",
            "circulant residual",
            f"{case['gram_circulant_relative_residual']:.3g}",
            "minpoly clusters",
            case["resolved_eigenvalue_clusters_1e8"],
            "LSMR",
            case["lsmr_iterations"],
        )


if __name__ == "__main__":
    main()
