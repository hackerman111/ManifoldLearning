"""Experimental simultaneous rank-r optimization of the live SVD objective.

CPU/float64, Frobenius ridge. No global optimality claim. Rank loss is allowed:
the feasible set is rank <= r, rather than the open exact-rank manifold.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from ADP.solver._multi_operator import forward
from ADP.solver.LSMR import (
    HPAOResult,
    _local_refit,
    _loss,
    _normalize_index,
    _validate_inputs,
)
from ADP.solver.SVD import _complete_basis, _factor_rank, _FlatU

_WORKSPACE_BYTES = 16 * 1024**2


def _core(A, V, W, prior, data, g, mass, ridge):
    """Stable full r²-variable LS; streaming augmented QR, no normal matrix."""
    r = A.shape[1]
    width = r * r
    alpha = g @ A
    # Budget simultaneous design/block/stack/QR scratch, plus the small R.
    chunk = max(1, _WORKSPACE_BYTES // (8 * (6 * (width + 1) + 2 * r + 2)))
    reduced = np.empty((0, width + 1))
    flat_W = W.reshape(-1, r)
    p = W.shape[1]
    flat_data = data.ravel()
    for start in range(0, len(flat_W), chunk):
        stop = min(start + chunk, len(flat_W))
        centers = np.arange(start, stop) // p
        # row-major vec(M): alpha_i * (U V)_k for every (i,k).
        design = (alpha[centers, :, None] * flat_W[start:stop, None, :]).reshape(
            -1, width
        )
        block = np.column_stack((design, flat_data[start:stop]))
        block *= np.sqrt(mass[centers, None])
        reduced = np.linalg.qr(np.vstack((reduced, block)), mode="r")
    if ridge:
        penalty = np.column_stack((np.eye(width), (A.T @ prior @ V).ravel()))
        reduced = np.linalg.qr(np.vstack((reduced, np.sqrt(ridge) * penalty)), mode="r")
    D, rhs = reduced[:, :width], reduced[:, width]
    solution = np.linalg.lstsq(D, rhs, rcond=None)[0]
    certificate = np.linalg.norm(D.T @ (D @ solution - rhs)) / max(
        np.linalg.norm(D.T @ rhs), np.finfo(float).tiny
    )
    if not np.all(np.isfinite(solution)) or not np.isfinite(certificate):
        raise RuntimeError("joint core solve is nonfinite")
    if certificate > 1e-8:
        raise RuntimeError("joint core solve failed normal-residual check")
    return solution.reshape(r, r), float(certificate)


def _evaluate(A, M, V, W, prior, data, g, mass, ridge, flat_U=None):
    X = A @ M @ V.T
    residual = data - np.einsum("jpk,jk->jp", W, (g @ A) @ M)
    objective = float(np.sum(mass[:, None] * residual**2))
    objective += ridge * float(np.sum((X - prior) ** 2))
    if not np.isfinite(objective):
        raise RuntimeError("joint objective is nonfinite")
    if flat_U is None:
        return objective
    pulled = flat_U.batched_rmatvec(residual)
    G = -2 * g.T @ (mass[:, None] * pulled) + 2 * ridge * (X - prior)
    GA, GV = G @ V @ M.T, G.T @ A @ M
    GA -= A @ (A.T @ GA)
    GV -= V @ (V.T @ GV)
    if not np.all(np.isfinite(GA)) or not np.all(np.isfinite(GV)):
        raise RuntimeError("joint gradient is nonfinite")
    return objective, GA, GV, G


def solve_fixed_coefficients(
    P,
    U,
    I,
    g,
    mass,
    *,
    rank,
    lambda_penalty,
    maxiter=100,
    tol=1e-6,
    low_rank_target="matrix",
):
    """Optimize all r directions together; return raw B and diagnostics.

    Initializes from top-r SVD of -grad(F(0))/2 in target coordinates.
    Reoptimizes full M after every QR trial; Armijo search on both spaces.
    ``tol`` controls the horizontal factor-gradient norm relative to its
    initial value. Other stops are explicitly not convergence certificates.
    """
    if hasattr(U, "__cuda_array_interface__"):
        raise NotImplementedError("joint rank solver requires CPU statistics")
    P, U, I, mass = _validate_inputs(P, U, I, mass)
    if P.ndim != 2 or not np.allclose(P @ P.T, np.eye(len(P)), atol=1e-10):
        raise ValueError("P must have orthonormal rows")
    if np.iscomplexobj(g) or not np.issubdtype(np.asarray(g).dtype, np.number):
        raise TypeError("g must be a real numeric array")
    g = np.asarray(g, dtype=float)
    if g.shape != (len(U), len(P)) or not np.all(np.isfinite(g)):
        raise ValueError("g must have finite shape (J,m)")
    if low_rank_target not in ("matrix", "correction"):
        raise ValueError("low_rank_target must be matrix or correction")
    for name, value in (("rank", rank), ("maxiter", maxiter)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
    if not 0 <= rank <= len(P) or maxiter < 1:
        raise ValueError("require 0 <= rank <= m and maxiter >= 1")
    if 3 * 8 * (rank * rank + 1) ** 2 > _WORKSPACE_BYTES:
        raise ValueError("rank is too large for the 16 MiB small-core workspace")
    if not np.isfinite(lambda_penalty) or lambda_penalty < 0:
        raise ValueError("lambda_penalty must be finite and nonnegative")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    ridge = float(lambda_penalty)
    base = P if low_rank_target == "correction" else np.zeros_like(P)
    prior = P - base
    data = I - forward(U, base, g) if low_rank_target == "correction" else I
    flat_U = _FlatU(U)
    initial = float(np.sum(mass[:, None] * data**2) + ridge * np.sum(prior**2))
    if not np.isfinite(initial):
        raise ValueError("initial joint objective must be finite")
    history = [initial]
    certificates = []
    backtracks = 0
    stop = "rank_zero"
    relative_gradient = 0.0
    tangent_gradient = 0.0
    X = np.zeros_like(P)
    A, V, singular = np.empty((len(P), 0)), np.empty((P.shape[1], 0)), np.empty(0)
    if rank:
        Q = g.T @ (mass[:, None] * flat_U.batched_rmatvec(data)) + ridge * prior
        if not np.all(np.isfinite(Q)):
            raise RuntimeError("joint initialization is nonfinite")
        left, values, right = np.linalg.svd(Q, full_matrices=False)
        if values[0] == 0:
            stop = "zero_gradient"
        else:
            A, V = left[:, :rank], right[:rank].T
            W = flat_U.matmat(V).reshape(*U.shape[:2], rank)
            M, cert = _core(A, V, W, prior, data, g, mass, ridge)
            certificates.append(cert)
            objective = cast(float, _evaluate(A, M, V, W, prior, data, g, mass, ridge))
            if objective > initial:
                raise RuntimeError("joint initialization increased objective")
            history.append(objective)
            initial_gradient = None
            step = 1.0
            stop = "iteration_limit"
            for _ in range(maxiter + 1):
                objective, GA, GV, G = _evaluate(
                    A, M, V, W, prior, data, g, mass, ridge, flat_U
                )
                objective, GA, GV, G = cast(
                    tuple[float, np.ndarray, np.ndarray, np.ndarray],
                    (objective, GA, GV, G),
                )
                norm2 = float(np.sum(GA**2) + np.sum(GV**2))
                if not np.isfinite(norm2):
                    raise RuntimeError("joint gradient norm is nonfinite")
                if initial_gradient is None:
                    initial_gradient = max(1.0, np.sqrt(norm2))
                relative_gradient = np.sqrt(norm2) / initial_gradient
                # Matrix tangent residual, including core: meaningful when
                # M has full rank. Factor gradients alone can hide rank loss.
                tangent = A @ (A.T @ G) + (G @ V) @ V.T
                tangent -= A @ (A.T @ G @ V) @ V.T
                tangent_gradient = float(np.linalg.norm(tangent))
                if relative_gradient <= tol:
                    stop = "factor_gradient_tolerance"
                    break
                if len(history) - 2 >= maxiter:
                    break
                if len(history) == 2:
                    step = 1 / max(1.0, np.sqrt(norm2))
                accepted = False
                new_A, new_V, new_W, new_M = A, V, W, M
                candidate = objective
                # ponytail: plain horizontal descent; add Riemannian CG only
                # if paired evidence justifies more optimizer machinery.
                for _trial in range(30):
                    new_A = np.linalg.qr(A - step * GA, mode="reduced")[0]
                    new_V = np.linalg.qr(V - step * GV, mode="reduced")[0]
                    new_W = flat_U.matmat(new_V).reshape(*U.shape[:2], rank)
                    new_M, cert = _core(
                        new_A, new_V, new_W, prior, data, g, mass, ridge
                    )
                    certificates.append(cert)
                    candidate = cast(
                        float,
                        _evaluate(
                            new_A, new_M, new_V, new_W, prior, data, g, mass, ridge
                        ),
                    )
                    if candidate <= objective - 1e-4 * step * norm2:
                        accepted = True
                        break
                    step *= 0.5
                    backtracks += 1
                if not accepted:
                    stop = "line_search_failed"
                    break
                # Canonicalize full M; rotate cached U V without another pass.
                R, singular, St = np.linalg.svd(new_M, full_matrices=False)
                A, V = new_A @ R, new_V @ St.T
                W = (new_W.reshape(-1, rank) @ St.T).reshape(*U.shape[:2], rank)
                M = np.diag(singular)
                history.append(candidate)
                step *= 2
            _, singular, _ = np.linalg.svd(M, full_matrices=False)
            X = A @ M @ V.T
    effective = _factor_rank(singular, *P.shape)
    diagnostics = {
        "optimizer": "joint_rank_r",
        "low_rank_target": low_rank_target,
        "effective_rank": effective,
        "rank_objective_history": tuple(history),
        "joint_stop_reason": stop,
        "joint_iterations": max(0, len(history) - 2),
        "joint_converged": stop == "factor_gradient_tolerance" and effective == rank,
        "factor_gradient_relative": float(relative_gradient),
        "matrix_tangent_gradient_norm": tangent_gradient,
        "core_normal_residual_max": max(certificates, default=0.0),
        "core_solves": len(certificates),
        "backtracks": backtracks,
        "u_vector_passes": flat_U.passes,
        "rank_scales": tuple(map(float, singular)),
        "rank_loss": effective < rank,
    }
    if low_rank_target == "correction":
        diagnostics.update(
            correction_rank=effective,
            correction_frobenius_norm=float(np.linalg.norm(X)),
        )
    return base + X, diagnostics


def solve(
    index_init,
    U,
    I,
    *,
    mass=None,
    lambda_prox=1.0,
    rank,
    maxiter=100,
    tol=1e-6,
    low_rank_target="matrix",
) -> HPAOResult:
    """ADP_solver callback with existing basis completion/QR and local refit."""
    if hasattr(U, "__cuda_array_interface__"):
        raise NotImplementedError("joint rank solver requires CPU statistics")
    P, U, I, mass = _validate_inputs(index_init, U, I, mass)
    if P.ndim != 2:
        raise ValueError("joint solver requires a multi-index matrix")
    P = _normalize_index(P)
    g, _ = _local_refit(I, U, P)
    raw, diagnostics = solve_fixed_coefficients(
        P,
        U,
        I,
        g,
        mass,
        rank=rank,
        lambda_penalty=lambda_prox,
        maxiter=maxiter,
        tol=tol,
        low_rank_target=low_rank_target,
    )
    if low_rank_target == "correction":
        if np.linalg.matrix_rank(raw) < len(P):
            raise RuntimeError("updated correction basis lost row rank")
        basis = np.linalg.qr(raw.T, mode="reduced")[0].T
        completion = "updated_basis_qr"
    else:
        _, singular, Vt = np.linalg.svd(raw, full_matrices=False)
        if _factor_rank(singular, *P.shape) == len(P):
            basis = Vt
        else:
            basis = _complete_basis(Vt.T, singular, P)
        completion = "prior_projected"
    coefficients, local_ranks = _local_refit(I, U, basis)
    diagnostics.update(
        linear_solver="experimental_joint_rank_r",
        loss=_loss(I, U, basis, coefficients, mass),
        local_rank_loss=int(np.count_nonzero(local_ranks < len(P))),
        completion=completion,
    )
    return HPAOResult(basis, coefficients, diagnostics)
