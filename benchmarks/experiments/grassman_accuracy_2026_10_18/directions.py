"""APPROXIMATE prototype: выбор rank-one плоскости по true-profile выигрышу.

Копия production solve от 2026-10-18; отличается только rank-one веткой.
Кандидаты — исходные SVD пары градиента, ограниченные rank/energy_tol.
Для каждого выполняется scalar search, затем true-profile Armijo.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy.optimize import OptimizeResult, minimize_scalar

from ADP.solver.grassman import (
    _core_jacobian,
    _gradient,
    _polar,
    _profile,
    _project,
)
from ADP.solver.LSMR import HPAOResult, _normalize_index, _validate_inputs

from .candidate import AngleProfile as _AngleProfile


def solve(
    index_init: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    *,
    mass: np.ndarray | None = None,
    lambda_prox: float = 0.05,
    max_steps: int = 5,
    tol: float = 1e-6,
    rank: int | None = None,
    energy_tol: float = 0.05,
    local_ridge: float = 0.0,
    max_angle: float = 0.5,
    workspace_bytes: int = 16 * 1024**2,
) -> HPAOResult:
    """Минимизировать sum mass*min_g(||I-U Y g||²+local_ridge||g||²).

    По умолчанию пять шагов: измеренный бюджет для outer warm start;
    отдельное frozen решение может требовать большего max_steps.
    index_init/result.index — (m,d), Y — (d,m). lambda_prox — положительное
    GN damping, не постоянный штраф в статистическом функционале.
    core_gn: полный q*m core Jacobian (II), polar retraction + Armijo.
    При tau=0 около rank boundary используется value-based rank-one вместо GN.
    При лимите workspace GN заменяется spectral; причина видна в diagnostics.
    Кэши O(J*p*(m+q)); GN workspace O(J*(p+m)*q*m), ограничен явно.
    Ограниченные шаги возвращают converged=False; глобальный минимум не обещан.
    """
    if hasattr(U, "__cuda_array_interface__"):
        raise NotImplementedError("Grassmann solver requires CPU statistics")
    B, U, I, mass = _validate_inputs(index_init, U, I, mass)
    assert mass is not None
    if B.ndim != 2:
        raise ValueError("Grassmann solver requires a multi-index matrix")
    for name, value in (
        ("max_steps", max_steps),
        ("workspace_bytes", workspace_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if rank is not None and (
        isinstance(rank, bool)
        or not isinstance(rank, (int, np.integer))
        or not 1 <= rank <= min(B.shape[0], B.shape[1] - B.shape[0])
    ):
        raise ValueError("rank must be between 1 and min(m,d-m)")
    for name, value in (
        ("lambda_prox", lambda_prox),
        ("tol", tol),
        ("max_angle", max_angle),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if max_angle >= np.pi / 2:
        raise ValueError("max_angle must be less than pi/2")
    if not np.isfinite(local_ridge) or local_ridge < 0:
        raise ValueError("local_ridge must be finite and nonnegative")
    if not np.isfinite(energy_tol) or not 0 <= energy_tol < 1:
        raise ValueError("energy_tol must be in [0,1)")
    U = np.ascontiguousarray(U)
    root_mass = np.sqrt(mass)[:, None]
    tangent_limit = np.tan(max_angle)
    Y = _normalize_index(B).T
    m, d = B.shape
    state = _profile(_project(U, Y), I, mass, local_ridge)
    history = [state.value]
    q_history: list[int] = []
    evaluations = gn_solves = fallback_rank = fallback_memory = 0
    accepted = 0
    angle_directions_tested = 0
    stop = "max_steps"
    scale = max(1.0, float(np.sum(mass[:, None] * I**2)))
    converged = False
    for _ in range(max_steps):
        G = _gradient(Y, U, state, mass)
        gradient_norm = float(np.linalg.norm(G))
        if gradient_norm <= tol * scale or m == d:
            converged = state.smooth or m == d
            stop = "stationary" if converged else "rank_boundary_stationary"
            break
        V, singular, At = np.linalg.svd(G, full_matrices=False)
        available = min(
            m,
            d - m,
            int(
                np.count_nonzero(
                    singular > np.finfo(float).eps * max(d, m) * singular[0]
                )
            ),
        )
        q = (
            min(rank, available)
            if rank is not None
            else min(
                available,
                int(
                    np.searchsorted(
                        np.cumsum(singular**2), (1 - energy_tol) * np.sum(singular**2)
                    )
                )
                + 1,
            )
        )
        current = "core_gn"
        if not state.smooth:
            current = "rank_one"
            fallback_rank += 1
        q_history.append(1 if current == "rank_one" else q)
        V, At = V[:, :q], At[:q]
        # Horizontalize once more before cached actions: roundoff only.
        V -= Y @ (Y.T @ V)
        V, _ = np.linalg.qr(V, mode="reduced")
        # QR can change signs/gauge; recover the actual gradient coordinates.
        reduced_gradient = V.T @ G
        W = _project(U, V)
        candidate = None
        if current == "core_gn":
            rows = len(I) * (I.shape[1] + (m if local_ridge else 0))
            # Jacobian, weighting copy and augmented lstsq copies are bounded.
            estimate = 8 * (4 * rows * q * m + (q * m) ** 2)
            if estimate > workspace_bytes:
                current = "spectral"
                fallback_memory += 1
            else:
                jac, residual = _core_jacobian(
                    state, W, mass, local_ridge, root_mass=root_mass
                )
                damping = lambda_prox
                # Меняется только малый ridge-блок, не копируем Jacobian на trial.
                coordinates = q * m
                design = np.empty((rows + coordinates, coordinates))
                design[:rows] = jac
                identity = np.eye(coordinates)
                rhs = np.concatenate((-residual, np.zeros(coordinates)))
                for _trial in range(8):
                    design[rows:] = np.sqrt(damping) * identity
                    K = np.linalg.lstsq(design, rhs, rcond=None)[0].reshape(q, m)
                    gn_solves += 1
                    length = np.linalg.norm(K, ord=2)
                    if length > tangent_limit:
                        K *= tangent_limit / length
                    slope = float(np.sum(reduced_gradient * K))
                    if slope < 0:
                        for backtrack in range(16):
                            Yn, Mn = _polar(Y, state.M, V, W, K * 0.5**backtrack)
                            trial = _profile(Mn, I, mass, local_ridge)
                            evaluations += 1
                            if (
                                trial.value
                                <= state.value + 1e-4 * slope * 0.5**backtrack
                            ):
                                candidate = Yn, trial
                                break
                    if candidate is not None:
                        break
                    damping *= 10
                if candidate is None:
                    current = "spectral"
        if current == "rank_one":
            for direction in range(q):
                a = At[direction]
                sign = -1.0 if float((V[:, direction] @ G) @ a) > 0 else 1.0
                v, w = sign * V[:, direction], sign * W[:, :, direction]
                curve = _AngleProfile(state.M, w, a, I, mass, local_ridge)
                angle_directions_tested += 1
                result = cast(
                    OptimizeResult,
                    minimize_scalar(
                        curve.value,
                        bounds=(0.0, max_angle),
                        method="bounded",
                        options={"xatol": max(1e-10, tol * 0.01)},
                    ),
                )
                t = float(result.x)
                slope = float(v @ G @ a)
                for backtrack in range(16):
                    angle = t * 0.5**backtrack
                    Yn = (
                        Y
                        + (Y @ a * (np.cos(angle) - 1) + v * np.sin(angle))[:, None] * a
                    )
                    trial = _profile(curve.design(angle), I, mass, local_ridge)
                    evaluations += 1
                    if trial.value <= state.value + 1e-4 * angle * slope:
                        if candidate is None or trial.value < candidate[1].value:
                            candidate = Yn, trial
                        break
                evaluations += curve.evaluations
        elif current == "spectral":
            # Recompute the tiny SVD after V's QR; keeps the same rank-q descent.
            L, s, R = np.linalg.svd(reduced_gradient, full_matrices=False)
            rotation_V, rotation_W = V @ L, W @ L
            angular_speed = s / s[0]
            slope = -float(np.sum(s * angular_speed))
            projected_Y, projected_M = Y @ R.T, state.M @ R.T
            for backtrack in range(20):
                t = max_angle * 0.5**backtrack
                angles = t * angular_speed
                delta = projected_Y * (np.cos(angles) - 1) - rotation_V * np.sin(angles)
                delta_M = projected_M * (np.cos(angles) - 1) - rotation_W * np.sin(
                    angles
                )
                Yn = Y + delta @ R
                trial = _profile(state.M + delta_M @ R, I, mass, local_ridge)
                evaluations += 1
                if trial.value <= state.value + 1e-4 * t * slope:
                    candidate = Yn, trial
                    break
        if candidate is None:
            stop = "line_search_failed"
            break
        Y, state = candidate
        accepted += 1
        history.append(state.value)
    # Final certificate uses the live operator rather than accumulated cached actions.
    state = _profile(_project(U, Y), I, mass, local_ridge)
    final_G = _gradient(Y, U, state, mass)
    gradient_norm = float(np.linalg.norm(final_G))
    if gradient_norm <= tol * scale:
        converged = state.smooth or m == d
        stop = "stationary" if converged else "rank_boundary_stationary"
    diagnostics: dict[str, object] = {
        "linear_solver": "grassman",
        "method": "core_gn_direction_search",
        "angle_directions_tested": angle_directions_tested,
        "angle_backend": "schur",
        "loss": 0.5 * float(np.sum(mass[:, None] * state.residual**2)),
        "profile_objective": state.value,
        "loss_history": tuple(v / 2 for v in history),
        "profile_history": tuple(history),
        "converged": converged,
        "stop_reason": stop,
        "accepted_steps": accepted,
        "iterations": accepted,
        "riemannian_gradient": gradient_norm / scale,
        "gradient_norm": gradient_norm,
        "local_rank_loss": int(np.count_nonzero(state.ranks < m)),
        "stationarity_applicable": state.smooth or m == d,
        "orthogonality_error": float(np.linalg.norm(Y.T @ Y - np.eye(m))),
        "step_ranks": tuple(q_history),
        "profile_evaluations": evaluations,
        "gn_solves": gn_solves,
        "rank_guard_fallbacks": fallback_rank,
        "workspace_fallbacks": fallback_memory,
        "local_ridge": local_ridge,
    }
    return HPAOResult(Y.T, state.coefficients, diagnostics)
