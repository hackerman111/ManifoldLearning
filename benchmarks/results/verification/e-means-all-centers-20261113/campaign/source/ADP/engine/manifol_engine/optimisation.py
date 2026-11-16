"""Шаги оптимизации локальных manifold-проекторов."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, linalg as sparse_linalg

from ...core.manifold import ADP_Manifold_utils as model_utils
from ...solver.HYBRID import solve_manifold
from .utils import orient_rows


def penalty_action(
    B: np.ndarray,
    source_projectors: np.ndarray,
    normalized_weights: np.ndarray,
) -> np.ndarray:
    """Применить штраф согласованности локальных проекторов к ``B``."""
    if B.shape[0] == 1:
        # EXACT: E = sum_j w_j p_j.T p_j для rank-one проекторов.
        rows = source_projectors[:, 0, :]
        projected = (normalized_weights * (rows @ B[0])) @ rows
        return B - projected[None, :]
    coordinates = B @ source_projectors.swapaxes(1, 2)  # (K, m, m)
    coordinates *= normalized_weights[:, None, None]
    # Только (m,K*m), без промежуточных (K,m,d) или (d,d).
    projected = coordinates.transpose(1, 0, 2).reshape(len(B), -1) @ (
        source_projectors.reshape(-1, B.shape[1])
    )
    return B - projected


def projector_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Измерить Frobenius-расстояние между двумя ортонормированными базисами."""
    overlap = right @ left.T
    distance2 = 2.0 * left.shape[0] - 2.0 * float(np.vdot(overlap, overlap).real)
    return math.sqrt(max(0.0, distance2))


def one_step(
    model: Any,
    I: np.ndarray,
    U: np.ndarray,
    mass: np.ndarray,
    graph: csr_matrix,
    projectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Выполнить один синхронный alternating step для всех targets."""
    J, m, _ = projectors.shape
    updated = np.empty_like(projectors)
    eigenvalues = np.empty((J, m))
    changes = np.empty(J)
    objectives = np.empty(J)
    penalties = np.empty(J)
    iterations_total = 0
    residual_max = 0.0
    dense_solves = 0
    pcg_solves = 0
    fallback_solves = 0

    for l in range(J):
        begin, end = graph.indptr[l : l + 2]
        sources = graph.indices[begin:end]
        weights = graph.data[begin:end]
        source_U = U[sources]
        source_I = I[sources]
        slopes = model._local_slopes(source_I, source_U, projectors[l], l)
        if model.solver == "hybrid":
            linear = solve_manifold(
                source_U,
                source_I,
                mass[sources],
                weights,
                projectors[sources],
                slopes,
                projectors[l],
                ridge=model.lambda_manifold,
                tol=model.cg_tol,
                maxiter=model.cg_maxiter,
                dense_max_unknowns=model.dense_max_unknowns,
                dense_max_bytes=model.dense_max_bytes,
            )
            B = linear.solution.reshape(projectors[l].shape)
            iterations, residual = linear.iterations, linear.relative_residual
            dense_solves += int(linear.backend == "dense-svd")
            pcg_solves += int(linear.backend == "block-pcg")
            fallback_solves += int(linear.backend.startswith("block-pcg->"))
        else:
            operator, preconditioner, rhs = model._build_B_system(
                source_U,
                source_I,
                mass[sources],
                weights,
                projectors[sources],
                slopes,
            )
            B, iterations, residual = model._solve_B(
                operator, preconditioner, rhs, projectors[l]
            )
        gamma = mass[sources] * weights
        updated[l], eigenvalues[l] = model._recover_projector(B, slopes, gamma, l)
        changes[l] = projector_distance(projectors[l], updated[l])
        objectives[l], penalties[l] = model._objective(
            B,
            source_U,
            source_I,
            gamma,
            slopes,
            projectors[sources],
            weights,
        )
        iterations_total += iterations
        residual_max = max(residual_max, residual)

    return (
        updated,
        eigenvalues,
        {
            "projector_change_median": float(np.median(changes)),
            "projector_change_max": float(np.max(changes)),
            "objective": float(objectives.sum()),
            "manifold_penalty": float(penalties.sum()),
            "cg_iterations": iterations_total if model.solver == "cg" else 0,
            "cg_relative_residual_max": residual_max if model.solver == "cg" else 0.0,
            "linear_iterations": iterations_total,
            "linear_relative_residual_max": residual_max,
            "dense_solves": dense_solves,
            "pcg_solves": pcg_solves,
            "fallback_solves": fallback_solves,
        },
    )


def local_slopes(
    I: np.ndarray, U: np.ndarray, projector: np.ndarray, target: int
) -> np.ndarray:
    """Оценить локальные slopes least-squares в текущем EDR-базисе."""
    slopes = np.empty((len(I), projector.shape[0]))
    for row, (I_j, U_j) in enumerate(zip(I, U, strict=True)):
        design = U_j @ projector.T
        slope, _, rank, _ = np.linalg.lstsq(design, I_j, rcond=None)
        model_utils.require_full_rank(
            rank,
            projector.shape[0],
            f"rank-deficient local slope for target {target}: "
            f"rank={rank}, required={projector.shape[0]}; "
            "increase N_phi or N_loc",
        )
        if not np.all(np.isfinite(slope)):
            model_utils.require_finite_directions(np.asarray([slope]))
        slopes[row] = slope
    return slopes


def build_B_system(
    model: Any,
    U: np.ndarray,
    I: np.ndarray,
    mass: np.ndarray,
    weights: np.ndarray,
    source_projectors: np.ndarray,
    slopes: np.ndarray,
) -> tuple[Any, Any, np.ndarray]:
    """Построить matrix-free normal operator для ``B`` формы ``(m,d)``."""
    m = slopes.shape[1]
    d = U.shape[2]
    gamma = mass * weights
    normalized_weights = weights / weights.sum()
    weighted_slopes = gamma[:, None] * slopes
    adjoint_U = U.swapaxes(1, 2)

    def matvec(vector: np.ndarray) -> np.ndarray:
        B = vector.reshape(m, d)
        local_vectors = slopes @ B  # (K, d)
        # EXACT: batched GEMV без повторного поиска einsum-path в каждом CG.
        projected = U @ local_vectors[..., None]  # (K, P, 1)
        pulled_back = (adjoint_U @ projected).squeeze(-1)  # (K, d)
        result = weighted_slopes.T @ pulled_back
        result += model.lambda_manifold * model._penalty_action(
            B, source_projectors, normalized_weights
        )
        return result.ravel()

    data_diagonal = np.einsum(
        "j,ja,jpd->ad",
        gamma,
        np.square(slopes),
        np.square(U),
        optimize=True,
    )
    projected_diagonal = np.einsum(
        "j,jrd,jrd->d",
        normalized_weights,
        source_projectors,
        source_projectors,
        optimize=True,
    )
    penalty_diagonal = 1.0 - projected_diagonal
    np.maximum(penalty_diagonal, 0.0, out=penalty_diagonal)
    diagonal = data_diagonal + model.lambda_manifold * penalty_diagonal[None, :]
    model_utils.require_operator_diagonal(diagonal)

    def precondition(vector: np.ndarray) -> np.ndarray:
        return vector / diagonal.ravel()

    operator_type: Any = sparse_linalg.LinearOperator
    operator = operator_type((m * d, m * d), matvec=matvec, rmatvec=matvec, dtype=float)
    preconditioner = operator_type(
        operator.shape,
        matvec=precondition,
        rmatvec=precondition,
        dtype=float,
    )
    pulled_I = np.einsum("jpd,jp->jd", U, I, optimize=True)
    rhs = np.einsum("j,ja,jd->ad", gamma, slopes, pulled_I, optimize=True).ravel()
    return operator, preconditioner, rhs


def solve_B(
    model: Any,
    operator: Any,
    preconditioner: Any,
    rhs: np.ndarray,
    initial: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Решить SPD-систему для ``B`` предобусловленным matrix-free CG."""
    iterations = 0

    def record(_: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    solution, info = sparse_linalg.cg(
        operator,
        rhs,
        x0=initial.ravel(),
        M=preconditioner,
        rtol=model.cg_tol,
        atol=0.0,
        maxiter=model.cg_maxiter or max(50, min(1000, 5 * int(np.prod(initial.shape)))),
        callback=record,
    )
    solution = np.asarray(solution, dtype=float)
    residual = operator @ solution - rhs
    relative_residual = float(np.linalg.norm(residual)) / max(
        float(np.linalg.norm(rhs)), np.finfo(float).tiny
    )
    limit = max(10.0 * model.cg_tol, 256.0 * np.finfo(float).eps)
    model_utils.require_cg_solution(
        info, iterations, relative_residual, solution, limit
    )
    return solution.reshape(initial.shape), iterations, relative_residual


def recover_projector(
    B: np.ndarray,
    slopes: np.ndarray,
    gamma: np.ndarray,
    target: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Извлечь ``P/Lambda`` из rank-``m`` фактора без ``d x d``."""
    if B.shape[0] == 1:
        # Единственное правое сингулярное направление sqrt(M)B равно B/||B||.
        value = float(np.dot(gamma, np.square(slopes[:, 0])))
        tolerance = 256.0 * np.finfo(float).eps * max(1.0, abs(value))
        model_utils.require_slope_matrix(np.asarray([value]), tolerance, target)
        norm = float(np.linalg.norm(B))
        singular = np.asarray([math.sqrt(max(0.0, value)) * norm])
        model_utils.require_identified(singular, B.shape, 1, target)
        projector = orient_rows(B / norm)
        spectrum = np.ones(1)
        orthogonality = np.linalg.norm(projector @ projector.T - np.eye(1), ord="fro")
        model_utils.require_projector(orthogonality, spectrum, target)
        return projector, spectrum
    M = np.einsum("j,ja,jb->ab", gamma, slopes, slopes, optimize=True)
    values, vectors = np.linalg.eigh(M)
    tolerance = 256.0 * np.finfo(float).eps * max(1.0, float(np.linalg.norm(M, ord=2)))
    model_utils.require_slope_matrix(values, tolerance, target)
    np.maximum(values, 0.0, out=values)
    root = (vectors * np.sqrt(values)[None, :]) @ vectors.T
    factor = root @ B
    _, singular_values, right_vectors = np.linalg.svd(factor, full_matrices=False)
    model_utils.require_identified(singular_values, factor.shape, B.shape[0], target)
    projector = orient_rows(right_vectors.copy())
    spectrum = np.square(singular_values)
    spectrum /= spectrum[0]
    orthogonality = np.linalg.norm(
        projector @ projector.T - np.eye(B.shape[0]), ord="fro"
    )
    model_utils.require_projector(orthogonality, spectrum, target)
    return projector, spectrum


def objective(
    model: Any,
    B: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    gamma: np.ndarray,
    slopes: np.ndarray,
    source_projectors: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Вычислить data-loss и manifold penalty текущего локального шага."""
    predicted = np.einsum("jpd,jd->jp", U, slopes @ B, optimize=True)
    data_term = float(
        np.einsum("j,jp,jp->", gamma, I - predicted, I - predicted, optimize=True)
    )
    penalty_action_value = model._penalty_action(
        B, source_projectors, weights / weights.sum()
    )
    penalty = model.lambda_manifold * float(np.vdot(B, penalty_action_value).real)
    tolerance = 256.0 * np.finfo(float).eps * max(1.0, float(np.vdot(B, B)))
    model_utils.require_nonnegative_objective(data_term, penalty, tolerance)
    penalty = max(0.0, penalty)
    return data_term + penalty, penalty
