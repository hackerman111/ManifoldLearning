"""Explicit rank-constrained multi-index step from ``SVD.tex`` (CPU)."""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr

from ._multi_operator import forward
from .LSMR import HPAOResult, _local_refit, _loss, _normalize_index, _validate_inputs


def solve_fixed_coefficients(
    P: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    g: np.ndarray,
    mass: np.ndarray,
    *,
    rank: int,
    lambda_penalty: float,
    inner_tol: float = 1e-6,
    inner_maxiter: int = 20,
    rank_tol: float = 1e-8,
    lsmr_maxiter: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Greedy rank-r minimization of the fixed-g objective in ``SVD.tex``.

    Returns the rank-deficient matrix B, not an m-dimensional EDR basis.
    Each v subproblem uses augmented LSMR; the m-by-m a system and k-by-k
    conditional singular-value system use dense rank-revealing solves.
    """
    P, U, I, mass = _validate_inputs(P, U, I, mass)
    if P.ndim != 2:
        raise ValueError("P must have shape (m, d)")
    m, d = P.shape
    if not np.allclose(P @ P.T, np.eye(m), rtol=1e-8, atol=1e-10):
        raise ValueError("P must have orthonormal rows")
    raw_g = np.asarray(g)
    if np.iscomplexobj(raw_g) or not np.issubdtype(raw_g.dtype, np.number):
        raise TypeError("g must be a real numeric array")
    g = np.asarray(g, dtype=float)
    if g.shape != (len(U), m) or not np.all(np.isfinite(g)):
        raise ValueError("g must have finite shape (J, m)")
    for name, value in (("rank", rank), ("inner_maxiter", inner_maxiter)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
    if not 1 <= rank < m or inner_maxiter < 1:
        raise ValueError("require 1 <= rank < m and inner_maxiter >= 1")
    if lsmr_maxiter is not None and (
        isinstance(lsmr_maxiter, bool)
        or not isinstance(lsmr_maxiter, (int, np.integer))
        or lsmr_maxiter < 1
    ):
        raise ValueError("lsmr_maxiter must be a positive integer or None")
    if not np.isfinite(lambda_penalty) or lambda_penalty < 0:
        raise ValueError("lambda_penalty must be finite and nonnegative")
    if not np.isfinite(inner_tol) or inner_tol <= 0:
        raise ValueError("inner_tol must be finite and positive")
    if not np.isfinite(rank_tol) or rank_tol < 0:
        raise ValueError("rank_tol must be finite and nonnegative")

    A = np.empty((m, 0))
    V = np.empty((d, 0))
    singular = np.empty(0)
    B = np.zeros_like(P)
    root_mass = np.sqrt(mass)
    objective = _objective(B, P, U, I, g, mass, lambda_penalty)
    if not np.isfinite(objective):
        raise ValueError("initial rank objective must be finite")
    history = [objective]
    certificates: list[float] = []
    inner_iterations: list[int] = []
    inner_converged: list[bool] = []
    stop_reason = "rank_limit"

    for k in range(1, rank + 1):
        residual = I - forward(U, B, g)
        pulled = (U.swapaxes(1, 2) @ residual[..., None]).squeeze(-1)
        Q = g.T @ (mass[:, None] * pulled) + lambda_penalty * (P - B)
        initial = Q - (Q @ V) @ V.T
        _, s0, right0 = np.linalg.svd(initial, full_matrices=False)
        if not len(s0) or s0[0] == 0:
            stop_reason = "zero_gradient"
            break
        v = right0[0]
        a = np.zeros(m)
        converged = False
        iterations = 0
        for _ in range(inner_maxiter):
            iterations += 1
            Uv = (U @ v[..., None]).squeeze(-1)  # (J, p)
            weights = mass * np.einsum("jp,jp->j", Uv, Uv)
            G = g.T @ (weights[:, None] * g) + lambda_penalty * np.eye(m)
            qv = Q @ v
            candidate_a = np.linalg.lstsq(G, qv, rcond=None)[0]
            norm_a = np.linalg.norm(candidate_a)
            if norm_a == 0 or not np.isfinite(norm_a):
                break
            a = candidate_a / norm_a
            previous_v = v
            v, certificate = _v_step(
                U,
                residual,
                P - B,
                g @ a,
                a,
                root_mass,
                lambda_penalty,
                inner_tol,
                lsmr_maxiter,
            )
            certificates.append(certificate)
            if 1.0 - abs(float(v @ previous_v)) < inner_tol:
                converged = True
                break
        if not np.any(a):
            stop_reason = "zero_gradient"
            break
        # Finish with the exact a minimizer for the last v, even at the cap.
        Uv = (U @ v[..., None]).squeeze(-1)
        weights = mass * np.einsum("jp,jp->j", Uv, Uv)
        G = g.T @ (weights[:, None] * g) + lambda_penalty * np.eye(m)
        candidate_a = np.linalg.lstsq(G, Q @ v, rcond=None)[0]
        norm_a = np.linalg.norm(candidate_a)
        if norm_a == 0 or not np.isfinite(norm_a):
            stop_reason = "zero_gradient"
            break
        a = candidate_a / norm_a
        inner_iterations.append(iterations)
        inner_converged.append(converged)
        alpha = g @ a
        denominator = float(np.sum(mass * alpha**2 * np.sum(Uv**2, axis=1)))
        denominator += lambda_penalty
        numerator = float(a @ Q @ v)
        if denominator <= 0 or not np.isfinite(denominator):
            stop_reason = "zero_curvature"
            break
        gain = numerator**2 / denominator
        if gain / max(1.0, objective) < rank_tol:
            stop_reason = "rank_tolerance"
            break

        # Compact QR/SVD of the k-column factors; no d-by-d matrix is formed.
        left = np.column_stack((A * singular, (numerator / denominator) * a))
        right = np.column_stack((V, v))
        ql, rl = np.linalg.qr(left, mode="reduced")
        qr, rr = np.linalg.qr(right, mode="reduced")
        small_a, values, small_vt = np.linalg.svd(rl @ rr.T, full_matrices=False)
        cutoff = np.finfo(float).eps * max(m, d) * values[0]
        if values[k - 1] <= cutoff:
            stop_reason = "no_rank_growth"
            break
        candidate_A = ql @ small_a[:, :k]
        candidate_V = qr @ small_vt[:k].T
        projected = U @ candidate_V  # (J, p, k)
        design = projected * (g @ candidate_A)[:, None, :]
        M = np.einsum("j,jpk,jpl->kl", mass, design, design, optimize=True)
        M += lambda_penalty * np.eye(k)
        b = np.einsum("j,jp,jpk->k", mass, I, design, optimize=True)
        b += lambda_penalty * np.einsum(
            "mk,mk->k", candidate_A, P @ candidate_V, optimize=True
        )
        candidate_s = np.linalg.lstsq(M, b, rcond=None)[0]
        candidate_B = (candidate_A * candidate_s) @ candidate_V.T
        candidate_objective = _objective(candidate_B, P, U, I, g, mass, lambda_penalty)
        if not np.isfinite(candidate_objective):
            raise RuntimeError("conditional singular-value solve is non-finite")
        if candidate_objective > objective + 64 * np.finfo(float).eps * max(
            1, objective
        ):
            raise RuntimeError("conditional singular-value solve increased objective")
        A, V, singular, B, objective = (
            candidate_A,
            candidate_V,
            candidate_s,
            candidate_B,
            candidate_objective,
        )
        history.append(objective)
        if np.linalg.matrix_rank(B) < k:
            stop_reason = "no_rank_growth"
            break

    return B, {
        "effective_rank": int(np.linalg.matrix_rank(B)),
        "rank_stop_reason": stop_reason,
        "rank_objective_history": tuple(history),
        "v_normal_residual_max": max(certificates, default=0.0),
        "inner_iterations": tuple(inner_iterations),
        "inner_converged": tuple(inner_converged),
        "rank_scales": tuple(float(x) for x in singular),
    }


def _v_step(U, residual, proximity, alpha, a, root_mass, ridge, tol, maxiter):
    """Solve H_a v=q_a with an augmented matrix-free least-squares operator."""
    d = U.shape[2]
    scale = root_mass * alpha
    root_ridge = math.sqrt(ridge)
    operator = _v_operator(U, scale, root_ridge)
    target = np.concatenate(
        ((root_mass[:, None] * residual).ravel(), root_ridge * (proximity.T @ a))
    )
    krylov_tol = min(1e-10, tol * 0.1)
    result = lsmr(
        operator,
        target,
        atol=krylov_tol,
        btol=krylov_tol,
        maxiter=maxiter or max(50, 5 * d),
    )
    solution = result[0]
    normal_rhs = operator.rmatvec(target)
    normal_residual = operator.rmatvec(target - operator @ solution)
    certificate = float(
        np.linalg.norm(normal_residual) / max(1.0, np.linalg.norm(normal_rhs))
    )
    if not np.all(np.isfinite(solution)) or certificate > max(1e-7, 10 * tol):
        raise RuntimeError("truncated-SVD v solve failed its normal-residual check")
    norm = np.linalg.norm(solution)
    if norm == 0:
        raise RuntimeError("truncated-SVD v solve returned a zero direction")
    return solution / norm, certificate


def _v_operator(U, scale, root_ridge):
    """Augmented design and its adjoint for a fixed left rank-one factor."""
    J, p, d = U.shape

    def matvec(x):
        return np.concatenate(((scale[:, None] * (U @ x)).ravel(), root_ridge * x))

    def rmatvec(y):
        pulled = (
            U.swapaxes(1, 2) @ (scale[:, None] * y[: J * p].reshape(J, p))[..., None]
        ).squeeze(-1)
        return pulled.sum(axis=0) + root_ridge * y[J * p :]

    return LinearOperator((J * p + d, d), matvec=matvec, rmatvec=rmatvec, dtype=float)


def _objective(B, P, U, I, g, mass, ridge):
    residual = I - forward(U, B, g)
    return float(np.sum(mass[:, None] * residual**2) + ridge * np.sum((B - P) ** 2))


def _complete_basis(B: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Keep recovered directions, then complete from the prior m-space."""
    left, singular, right = np.linalg.svd(B, full_matrices=False)
    del left
    threshold = np.finfo(float).eps * max(B.shape) * singular[0] if len(singular) else 0
    rank = int(np.count_nonzero(singular > threshold))
    if rank == 0:
        return P.copy()
    active = right[:rank]
    residual = P - (P @ active.T) @ active
    _, values, complement = np.linalg.svd(residual, full_matrices=False)
    if values[P.shape[0] - rank - 1] <= np.finfo(float).eps * max(P.shape) * values[0]:
        raise RuntimeError("prior basis cannot complete the rank-constrained step")
    return np.vstack((active, complement[: P.shape[0] - rank]))


def solve(
    index_init: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    *,
    mass: np.ndarray | None = None,
    lambda_prox: float = 1.0,
    rank: int,
    inner_tol: float = 1e-6,
    inner_maxiter: int = 20,
    rank_tol: float = 1e-8,
    lsmr_maxiter: int | None = None,
) -> HPAOResult:
    """Opt-in ADP solver: fixed-g rank-r step followed by prior completion.

    Here ``lambda_prox`` is the manuscript's ``lambda * ||B-P||²`` penalty,
    unlike the correction penalty in the default LSMR solver.
    """
    if hasattr(U, "__cuda_array_interface__"):
        raise NotImplementedError("truncated-SVD solver requires CPU statistics")
    P, U, I, mass = _validate_inputs(index_init, U, I, mass)
    if P.ndim != 2:
        raise ValueError("truncated-SVD solver requires a multi-index matrix")
    P = _normalize_index(P)
    g, _ = _local_refit(I, U, P)
    B, diagnostics = solve_fixed_coefficients(
        P,
        U,
        I,
        g,
        mass,
        rank=rank,
        lambda_penalty=lambda_prox,
        inner_tol=inner_tol,
        inner_maxiter=inner_maxiter,
        rank_tol=rank_tol,
        lsmr_maxiter=lsmr_maxiter,
    )
    basis = _complete_basis(B, P)
    coefficients, local_ranks = _local_refit(I, U, basis)
    diagnostics.update(
        linear_solver="truncated_svd",
        loss=_loss(I, U, basis, coefficients, mass),
        local_rank_loss=int(np.count_nonzero(local_ranks < len(P))),
        completion="prior_projected",
    )
    return HPAOResult(basis, coefficients, diagnostics)
