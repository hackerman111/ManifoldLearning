"""Медленный SVD-reference для градиента усечённого reduced objective."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class ReducedReference:
    objective: float
    gradient: np.ndarray
    coefficients: np.ndarray
    ranks: np.ndarray
    value_defect: float
    gradient_defect: float
    rank_loss: int


def _gradient_A(
    left: np.ndarray,
    singular: np.ndarray,
    right: np.ndarray,
    I: np.ndarray,
    mass: float,
    rank: int,
) -> np.ndarray:
    """Формула (3) из proof note, матрица P на m; скалярные циклы."""
    P, m = left.shape
    gradient = np.zeros((P, m))
    alpha = left.T @ I
    q0 = I - left @ alpha
    for i in range(rank):
        ai = singular[i] * right[i]
        lam_i = singular[i] ** 2
        gradient -= mass * alpha[i] / lam_i * np.outer(q0, ai)
        for k in range(rank, m):
            ak = singular[k] * right[k]
            gradient -= (
                mass
                * alpha[i]
                * alpha[k]
                / (lam_i - singular[k] ** 2)
                * (np.outer(left[:, k], ai) + np.outer(left[:, i], ak))
            )
    return gradient


def evaluate_reduced(
    B: np.ndarray, U: np.ndarray, I: np.ndarray, mass: np.ndarray
) -> ReducedReference:
    """Вычислить f_tau, точный gradient и ошибки rank-truncation.

    Это auditable reference для proof gate. Отказ означает выход
    из проверяемой гладкой области.
    """
    B = np.asarray(B, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    I = np.asarray(I, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    if (
        B.ndim != 2
        or U.ndim != 3
        or I.shape != U.shape[:2]
        or mass.shape != (U.shape[0],)
        or B.shape[1] != U.shape[2]
        or not 1 <= B.shape[0] <= min(U.shape[1], U.shape[2])
    ):
        raise ValueError("incompatible reduced-objective shapes")
    if not all(np.all(np.isfinite(a)) for a in (B, U, I, mass)):
        raise ValueError("nonfinite reduced-objective input")
    if np.any(mass <= 0):
        raise ValueError("mass must be positive")

    J, P, d = U.shape
    m = B.shape[0]
    tau = np.finfo(float).eps * max(P, m)
    gradient = np.zeros((m, d))
    exact_gradient = np.zeros((m, d))
    coefficients = np.zeros((J, m))
    ranks = np.empty(J, dtype=np.intp)
    value = defect = operator_norm2 = 0.0
    rank_loss = 0
    for j in range(J):
        projected = U[j] @ B.T
        left, singular, right = np.linalg.svd(projected, full_matrices=False)
        alpha = left.T @ I[j]
        rank = int(np.count_nonzero(singular > tau * singular[0]))
        ranks[j] = rank
        positive = int(np.count_nonzero(singular > 0))
        if rank == 0:
            if np.any(U[j] != 0):
                raise RuntimeError("rank-zero projected system has nonzero U")
        else:
            if singular[rank - 1] < 64 * np.finfo(float).eps * max(
                1.0, np.linalg.norm(U[j])
            ):
                raise RuntimeError("retained singular value below absolute guard")
            if singular[rank - 1] / singular[0] < 2 * tau:
                raise RuntimeError("retained singular value near cutoff")
            if rank < m and singular[rank] / singular[0] > 0.5 * tau:
                raise RuntimeError("discarded singular value near cutoff")

        residual = I[j] - left[:, :rank] @ alpha[:rank]
        value += 0.5 * mass[j] * float(residual @ residual)
        defect += 0.5 * mass[j] * float(alpha[rank:positive] @ alpha[rank:positive])
        if rank:
            coefficients[j] = right[:rank].T @ (alpha[:rank] / singular[:rank])
            operator_norm2 += (
                mass[j]
                * float(coefficients[j] @ coefficients[j])
                * float(np.sum(U[j] * U[j]))
            )
        gradient += _gradient_A(left, singular, right, I[j], mass[j], rank).T @ U[j]
        exact_gradient += (
            _gradient_A(left, singular, right, I[j], mass[j], positive).T @ U[j]
        )
        rank_loss += rank < m

    def horizontal(value: np.ndarray) -> np.ndarray:
        return value - (value @ B.T) @ B

    scale = max(1.0, math.sqrt(operator_norm2 * 2.0 * value))
    gradient_defect = float(
        np.linalg.norm(horizontal(gradient - exact_gradient)) / scale
    )
    if not all(np.isfinite(x) for x in (value, defect, gradient_defect)):
        raise RuntimeError("nonfinite reduced-objective result")
    return ReducedReference(
        value, gradient, coefficients, ranks, defect, gradient_defect, rank_loss
    )


def _retract(B: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    raw = B + tangent
    eigenvalues, eigenvectors = np.linalg.eigh(raw @ raw.T)
    return ((eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T) @ raw


def _rank_pattern(B: np.ndarray, U: np.ndarray) -> np.ndarray:
    singular = np.linalg.svd(U @ B.T, compute_uv=False)
    cutoff = np.finfo(float).eps * max(U.shape[1], B.shape[0]) * singular[:, :1]
    return np.count_nonzero(singular > cutoff, axis=1)


def main(argv: list[str] | None = None) -> int:
    """Проверить формулу (3) на 12 frozen задачах до рабочего solver."""
    from .multi_solver_search import _sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    if options.output.exists():
        raise FileExistsError(options.output)
    manifest_path = options.frozen_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    results = []
    for fit in manifest["fits"]:
        for call in fit["calls"]:
            if "file" not in call:
                continue
            path = options.frozen_dir / call["file"]
            if _sha256(path) != call["sha256"]:
                raise RuntimeError(f"frozen task hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as arrays:
                B, U, I, mass = (arrays[key] for key in ("B", "U", "I", "mass"))
                rng = np.random.default_rng(fit["seed"] + call["outer"] + 17)
                tangent = rng.normal(size=B.shape)
                tangent -= (tangent @ B.T) @ B
                tangent /= np.linalg.norm(tangent)
                base = evaluate_reduced(B, U, I, mass)
                pattern = _rank_pattern(B, U)
                directional = float(np.sum(base.gradient * tangent))
                errors = {}
                stable = True
                for step in (1e-3, 1e-4, 1e-5):
                    plus_B = _retract(B, step * tangent)
                    minus_B = _retract(B, -step * tangent)
                    stable &= bool(
                        np.array_equal(pattern, _rank_pattern(plus_B, U))
                        and np.array_equal(pattern, _rank_pattern(minus_B, U))
                    )
                    plus = evaluate_reduced(plus_B, U, I, mass).objective
                    minus = evaluate_reduced(minus_B, U, I, mass).objective
                    errors[str(step)] = abs(
                        (plus - minus) / (2 * step) - directional
                    ) / max(1.0, abs(directional))
                results.append(
                    {
                        "point": fit["point"],
                        "seed": fit["seed"],
                        "outer": call["outer"],
                        "file": call["file"],
                        "sha256": call["sha256"],
                        "rank_pattern_stable": stable,
                        "rank_loss": base.rank_loss,
                        "objective": base.objective,
                        "value_defect": base.value_defect,
                        "gradient_defect": base.gradient_defect,
                        "relative_fd_errors": errors,
                    }
                )
    if len(results) != 12:
        raise ValueError("expected 12 frozen tasks")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(
            {"frozen_manifest_sha256": _sha256(manifest_path), "tasks": results},
            indent=2,
        )
        + "\n"
    )
    print(f"checked {len(results)} frozen tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
