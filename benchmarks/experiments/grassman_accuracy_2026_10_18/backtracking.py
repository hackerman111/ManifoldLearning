"""APPROXIMATE prototype: true-profile Armijo вместо bounded angle search.

Исходное ядро production2026-10-18; отличается rank-one line search.
Прямой QR/SVD каждого trial, без Schur subtraction или scalar optimizer.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy.optimize import OptimizeResult, minimize_scalar

from ADP.solver.grassman import (
    _AngleProfile,
    _core_jacobian,
    _gradient,
    _polar,
    _profile,
    _project,
)
from ADP.solver.LSMR import HPAOResult, _normalize_index, _validate_inputs


def _intrinsic_model(U, I, mass, state, cached_ranks):
    """Approximate GN model only; full profiles/certificates remain original."""
    m = state.M.shape[2]
    deficient = state.ranks < m
    if not np.any(deficient):
        return None
    unknown = np.flatnonzero(deficient & (cached_ranks < 0))
    chunk = max(1, (4 * 1024**2) // (8 * U.shape[1] * U.shape[2]))
    for start in range(0, len(unknown), chunk):
        indices = unknown[start : start + chunk]
        singular = np.linalg.svd(U[indices], compute_uv=False)
        cutoff = np.finfo(float).eps * max(U.shape[1:]) * singular[:, :1]
        cached_ranks[indices] = np.count_nonzero(singular > cutoff, axis=1)
    if np.any(cached_ranks[deficient] != state.ranks[deficient]):
        return None
    regular = ~deficient
    if not np.any(regular):
        return None
    model = _profile(state.M[regular], I[regular], mass[regular], 0.0)
    return (regular, model) if model.smooth else None


def _conjugate_gradient(
    Y: np.ndarray,
    G: np.ndarray,
    previous_gradient: np.ndarray | None,
    previous_direction: np.ndarray | None,
) -> tuple[np.ndarray, float]:
    """PR+ with projection transport and an actual-gradient descent guard."""
    if previous_gradient is None or previous_direction is None:
        return G, 0.0
    denominator = float(np.sum(previous_gradient**2))
    if not np.isfinite(denominator) or denominator <= 0:
        return G, 0.0
    transported_G = previous_gradient - Y @ (Y.T @ previous_gradient)
    transported_D = previous_direction - Y @ (Y.T @ previous_direction)
    beta = max(0.0, float(np.sum(G * (G - transported_G))) / denominator)
    direction = -G + beta * transported_D
    margin = -1e-3 * np.linalg.norm(G) * np.linalg.norm(direction)
    if not np.isfinite(beta) or not float(np.sum(G * direction)) < margin:
        return G, 0.0
    return -direction, beta


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
    quadratic_trials: int = 0,
    smooth_max_steps: int | None = 5,
    intrinsic_gn: bool = False,
    intrinsic_warmup: bool = False,
    spectral_guard: bool = False,
    conjugate_guard: bool = False,
    per_center_profile: bool = False,
    normalized_damping: bool = False,
) -> HPAOResult:
    """Прежний профиль, прямой QR/SVD Armijo для rank-one fallback.

    Initial smooth profile получает min(max_steps,5), иначе max_steps.
    GN rank/workspace guards сохранены. Каждый rank-one trial проверяет
    полный профиль без Schur-вычитания. Сертификат вычисляется через live UY.
    index_init/result.index — (m,d); нет гарантии глобального минимума.
    """
    if hasattr(U, "__cuda_array_interface__"):
        raise NotImplementedError("Grassmann solver requires CPU statistics")
    if not isinstance(intrinsic_warmup, bool):
        raise TypeError("intrinsic_warmup must be a boolean")
    if intrinsic_warmup and not intrinsic_gn:
        raise ValueError("intrinsic_warmup requires intrinsic_gn")
    if not isinstance(spectral_guard, bool):
        raise TypeError("spectral_guard must be a boolean")
    if spectral_guard and intrinsic_gn:
        raise ValueError("spectral_guard and intrinsic_gn are separate variants")
    if not isinstance(conjugate_guard, bool):
        raise TypeError("conjugate_guard must be a boolean")
    if not isinstance(per_center_profile, bool):
        raise TypeError("per_center_profile must be a boolean")
    if per_center_profile and intrinsic_gn:
        raise ValueError("per_center_profile and intrinsic_gn are separate variants")
    if not isinstance(normalized_damping, bool):
        raise TypeError("normalized_damping must be a boolean")
    if normalized_damping and not intrinsic_gn:
        raise ValueError("normalized_damping requires intrinsic_gn")
    if conjugate_guard and not spectral_guard:
        raise ValueError("conjugate_guard requires spectral_guard")
    if smooth_max_steps is not None and (
        isinstance(smooth_max_steps, bool)
        or not isinstance(smooth_max_steps, (int, np.integer))
        or smooth_max_steps <= 0
    ):
        raise ValueError("smooth_max_steps must be a positive integer or None")
    if (
        isinstance(quadratic_trials, bool)
        or not isinstance(quadratic_trials, (int, np.integer))
        or not 0 <= quadratic_trials <= 3
    ):
        raise ValueError("quadratic_trials must be an integer in [0,3]")
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
    profile_fn = _profile
    if per_center_profile:
        from .selective import profile as profile_fn

    U = np.ascontiguousarray(U)
    root_mass = np.sqrt(mass)[:, None]
    tangent_limit = np.tan(max_angle)
    Y = _normalize_index(B).T
    m, d = B.shape
    state = profile_fn(_project(U, Y), I, mass, local_ridge)
    history = [state.value]
    q_history: list[int] = []
    evaluations = gn_solves = fallback_rank = fallback_memory = 0
    accepted = 0
    stop = "max_steps"
    scale = max(1.0, float(np.sum(mass[:, None] * I**2)))
    converged = False
    cached_ranks = np.full(len(I), -1, dtype=int) if intrinsic_gn else None
    intrinsic_proposals = 0
    normalized_proposals = damping_scale_fallbacks = 0
    warmup_steps = 0
    previous_gradient: np.ndarray | None = None
    previous_direction: np.ndarray | None = None
    previous_ranks: np.ndarray | None = None
    conjugate_betas: list[float] = []
    step_budget = (
        min(max_steps, smooth_max_steps)
        if state.smooth and smooth_max_steps is not None
        else max_steps
    )
    for _ in range(step_budget):
        G = _gradient(Y, U, state, mass)
        gradient_norm = float(np.linalg.norm(G))
        if gradient_norm <= tol * scale or m == d:
            converged = state.smooth or m == d
            stop = "stationary" if converged else "rank_boundary_stationary"
            break
        search_gradient = G
        if conjugate_guard and not state.smooth:
            same_ranks = previous_ranks is not None and np.array_equal(
                state.ranks, previous_ranks
            )
            search_gradient, beta = _conjugate_gradient(
                Y,
                G,
                previous_gradient if same_ranks else None,
                previous_direction if same_ranks else None,
            )
            conjugate_betas.append(beta)
        V, singular, At = np.linalg.svd(search_gradient, full_matrices=False)
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
        gn_state, gn_mask = state, slice(None)
        warmup = intrinsic_warmup and not state.smooth and accepted == 0
        if not state.smooth:
            model = (
                _intrinsic_model(U, I, mass, state, cached_ranks)
                if intrinsic_gn and not local_ridge and not warmup
                else None
            )
            if model is not None:
                gn_mask, gn_state = model
                intrinsic_proposals += 1
            else:
                current = "spectral" if spectral_guard else "rank_one"
                fallback_rank += 1
                if spectral_guard and rank is None:
                    q = available
        if current == "rank_one":
            q = 1
        q_history.append(q)
        V, At = V[:, :q], At[:q]
        # Horizontalize once more before cached actions: roundoff only.
        V -= Y @ (Y.T @ V)
        V, _ = np.linalg.qr(V, mode="reduced")
        # QR can change signs/gauge; recover the actual gradient coordinates.
        reduced_gradient = V.T @ G
        W = _project(U, V)
        candidate = None
        search_direction: np.ndarray | None = None
        if current == "core_gn":
            rows = len(gn_state.M) * (I.shape[1] + (m if local_ridge else 0))
            # Jacobian, weighting copy and augmented lstsq copies are bounded.
            estimate = 8 * (4 * rows * q * m + (q * m) ** 2)
            if estimate > workspace_bytes:
                current = "spectral"
                fallback_memory += 1
            else:
                jac, residual = _core_jacobian(
                    gn_state,
                    W[gn_mask],
                    mass[gn_mask],
                    local_ridge,
                    root_mass=root_mass[gn_mask],
                )
                damping = lambda_prox
                # Меняется только малый ridge-блок, не копируем Jacobian на trial.
                coordinates = q * m
                gn_trials = 8
                if normalized_damping and not state.smooth:
                    # Единицы J^T J без построения нормальной матрицы/копии J.
                    damping *= float(np.einsum("ij,ij->", jac, jac)) / coordinates
                    if np.isfinite(damping) and damping > 0:
                        normalized_proposals += 1
                    else:
                        damping_scale_fallbacks += 1
                        gn_trials = 0
                design = np.empty((rows + coordinates, coordinates))
                design[:rows] = jac
                identity = np.eye(coordinates)
                rhs = np.concatenate((-residual, np.zeros(coordinates)))
                for _trial in range(gn_trials):
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
                            trial = profile_fn(Mn, I, mass, local_ridge)
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
            a = At[0]
            # V from QR; choose sign by actual derivative, not QR convention.
            sign = -1.0 if float((V[:, 0] @ G) @ a) > 0 else 1.0
            v, w = sign * V[:, 0], sign * W[:, :, 0]
            u = (state.M @ a).reshape(I.shape)
            slope = float(v @ G @ a)
            angle = max_angle
            if warmup:
                warmup_steps += 1
                curve = _AngleProfile(state.M, w, a, I, mass, local_ridge)
                result = cast(
                    OptimizeResult,
                    minimize_scalar(
                        curve.value,
                        bounds=(0.0, max_angle),
                        method="bounded",
                        options={"xatol": max(1e-10, tol * 0.01)},
                    ),
                )
                for backtrack in range(16):
                    angle = float(result.x) * 0.5**backtrack
                    Yn = (
                        Y
                        + (Y @ a * (np.cos(angle) - 1) + v * np.sin(angle))[:, None] * a
                    )
                    trial = profile_fn(curve.design(angle), I, mass, local_ridge)
                    evaluations += 1
                    if trial.value <= state.value + 1e-4 * angle * slope:
                        candidate = Yn, trial
                        break
                evaluations += curve.evaluations
            for _proposal in range(0 if warmup else quadratic_trials):
                delta, sine = np.cos(angle) - 1, np.sin(angle)
                Yn = Y + (Y @ a * delta + v * sine)[:, None] * a
                Mn = state.M + (u * delta + w * sine)[:, :, None] * a
                trial = profile_fn(Mn, I, mass, local_ridge)
                evaluations += 1
                if trial.value <= state.value + 1e-4 * angle * slope and (
                    candidate is None or trial.value < candidate[1].value
                ):
                    candidate = Yn, trial
                curvature = 2 * (trial.value - state.value - angle * slope) / angle**2
                if not np.isfinite(curvature) or curvature <= 0:
                    break
                proposed = float(
                    np.clip(-slope / curvature, max_angle * 0.5**19, max_angle)
                )
                if abs(proposed - angle) <= 1e-6 * angle:
                    break
                angle = proposed
            for backtrack in range(20 if candidate is None and not warmup else 0):
                angle = max_angle * 0.5**backtrack
                delta = np.cos(angle) - 1
                sine = np.sin(angle)
                Yn = Y + (Y @ a * delta + v * sine)[:, None] * a
                Mn = state.M + (u * delta + w * sine)[:, :, None] * a
                trial = profile_fn(Mn, I, mass, local_ridge)
                evaluations += 1
                if trial.value <= state.value + 1e-4 * angle * slope:
                    candidate = Yn, trial
                    break
        elif current == "spectral":
            # Recompute the tiny SVD after V's QR; keeps the same rank-q descent.
            L, s, R = np.linalg.svd(
                V.T @ search_gradient if conjugate_guard else reduced_gradient,
                full_matrices=False,
            )
            rotation_V, rotation_W = V @ L, W @ L
            if conjugate_guard:
                search_direction = -((rotation_V * s) @ R)
            angular_speed = s / s[0]
            slope = (
                -float(np.sum(reduced_gradient * ((L * angular_speed) @ R)))
                if conjugate_guard
                else -float(np.sum(s * angular_speed))
            )
            projected_Y, projected_M = Y @ R.T, state.M @ R.T
            for backtrack in range(20 if slope < 0 else 0):
                t = max_angle * 0.5**backtrack
                angles = t * angular_speed
                delta = projected_Y * (np.cos(angles) - 1) - rotation_V * np.sin(angles)
                delta_M = projected_M * (np.cos(angles) - 1) - rotation_W * np.sin(
                    angles
                )
                Yn = Y + delta @ R
                trial = profile_fn(state.M + delta_M @ R, I, mass, local_ridge)
                evaluations += 1
                if trial.value <= state.value + 1e-4 * t * slope:
                    candidate = Yn, trial
                    break
        if candidate is None:
            stop = "line_search_failed"
            break
        if conjugate_guard:
            if not state.smooth and search_direction is not None:
                previous_gradient, previous_ranks = G, state.ranks
                previous_direction = search_direction
            else:
                previous_gradient = previous_direction = previous_ranks = None
        Y, state = candidate
        accepted += 1
        history.append(state.value)
    # Final certificate uses the live operator rather than accumulated cached actions.
    state = profile_fn(_project(U, Y), I, mass, local_ridge)
    final_G = _gradient(Y, U, state, mass)
    gradient_norm = float(np.linalg.norm(final_G))
    if gradient_norm <= tol * scale:
        converged = state.smooth or m == d
        stop = "stationary" if converged else "rank_boundary_stationary"
    diagnostics: dict[str, object] = {
        "linear_solver": "grassman",
        "method": "core_gn_quadratic" if quadratic_trials else "core_gn_armijo",
        "angle_backend": "refit_quadratic" if quadratic_trials else "refit_armijo",
        "effective_max_steps": step_budget,
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
    if intrinsic_gn:
        diagnostics.update(
            method="core_gn_intrinsic", intrinsic_gn_proposals=intrinsic_proposals
        )
    if intrinsic_warmup:
        diagnostics.update(
            method="core_gn_intrinsic_warmup",
            angle_backend="schur_then_refit_armijo",
            intrinsic_warmup_steps=warmup_steps,
        )
    if normalized_damping:
        diagnostics.update(
            method="core_gn_relative",
            normalized_gn_proposals=normalized_proposals,
            damping_scale_fallbacks=damping_scale_fallbacks,
        )
    if spectral_guard:
        diagnostics["method"] = "core_gn_spectral_guard"
    if conjugate_guard:
        diagnostics.update(
            method="core_gn_conjugate_guard",
            conjugate_betas=tuple(conjugate_betas),
            conjugate_steps=sum(beta > 0 for beta in conjugate_betas),
        )
    return HPAOResult(Y.T, state.coefficients, diagnostics)
