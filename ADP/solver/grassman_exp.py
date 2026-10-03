"""Экспериментальные Grassmann режимы, не прошедшие time/quality selection.

solve(method="flat_gn") — совместные q углов; solve_full_gn — полный tangent GN.
Режимы APPROXIMATE, цель и rank guards сохранены. Core baseline доступен для пар.
Evidence: experiments/{grassman_flat,glm_grassmann}_2026_10_02/REPORT.md.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from scipy.optimize import OptimizeResult, minimize_scalar
from scipy.sparse.linalg import LinearOperator, lsmr

from . import grassman as base
from .grassman import (
    _AngleProfile,
    _core_jacobian,
    _gradient,
    _polar,
    _Profile,
    _profile,
    _project,
)
from .LSMR import HPAOResult, _normalize_index, _validate_inputs


def _flat_design(
    M: np.ndarray,
    rotated_M: np.ndarray,
    W: np.ndarray,
    At: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Точное U Exp_Y(V diag(theta) A.T), только кэш (J,p,m+q)."""
    return M + (rotated_M * (np.cos(theta) - 1) + W * np.sin(theta)) @ At


def _flat_jacobian(
    state: _Profile,
    D: np.ndarray,
    At: np.ndarray,
    mass: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Полный profiled Jacobian по q углам, включая dg и ridge-блок.

    D=(J,p,q), partial_l M=D_l a_l.T, At=(q,m). Нужна безопасная
    full-rank область; производная minimum-norm на rank boundary не применяется.
    """
    if not state.smooth:
        raise ValueError(
            "full-rank profile derivative requires well-conditioned local systems"
        )
    J, p, m = state.M.shape
    q = At.shape[0]
    projected_g = state.coefficients @ At.T  # (J,q)
    dr_dot = (state.residual[:, None, :] @ D).squeeze(1)  # (J,q)
    rhs = At.T[None] * dr_dot[:, None, :]
    rhs -= (state.M.swapaxes(1, 2) @ D) * projected_g[:, None, :]
    dg = state.normal_solve(rhs)  # (J,m,q)
    jac = np.empty((J, p + (m if tau else 0), q))
    jac[:, :p] = -D * projected_g[:, None, :]
    jac[:, :p] -= state.M @ dg
    if tau:
        jac[:, p:] = -np.sqrt(tau) * dg
    residual = (
        state.residual
        if not tau
        else np.concatenate(
            (state.residual, -np.sqrt(tau) * state.coefficients), axis=1
        )
    )
    root = np.sqrt(mass)[:, None]
    jac *= root[:, :, None]
    return jac.reshape(-1, q), (root * residual).ravel()


def _flat_optimize(
    state: _Profile,
    W: np.ndarray,
    At: np.ndarray,
    I: np.ndarray,
    mass: np.ndarray,
    tau: float,
    *,
    damping: float,
    max_angle: float,
    max_steps: int,
    tol: float,
) -> tuple[np.ndarray, _Profile, int, int, int]:
    """Совместный bounded GN/Armijo; ни U, ни ambient Y в inner loop нет.

    Возвращает theta, профиль, число принятых inner шагов/evaluations/solves.
    Box ограничивает конечные углы выбранного flat, не меняя сам профиль.
    """
    q = At.shape[0]
    theta = np.zeros(q)
    base_M = state.M
    rotated_M = base_M @ At.T
    evaluations = solves = accepted = 0
    scale = max(1.0, float(np.sum(mass[:, None] * I**2)))
    for _ in range(max_steps):
        if not state.smooth:
            break
        D = -rotated_M * np.sin(theta) + W * np.cos(theta)
        jac, residual = _flat_jacobian(state, D, At, mass, tau)
        gradient = 2 * (jac.T @ residual)
        if np.linalg.norm(gradient) <= tol * scale:
            break
        rows = jac.shape[0]
        design = np.empty((rows + q, q))
        design[:rows] = jac
        rhs = np.concatenate((-residual, np.zeros(q)))
        identity = np.eye(q)
        candidate = None
        trial_damping = damping
        for _trial in range(8):
            design[rows:] = np.sqrt(trial_damping) * identity
            step = np.linalg.lstsq(design, rhs, rcond=None)[0]
            solves += 1
            # Совместный векторный шаг; clipping задаёт box на theta.
            step = np.clip(theta + step, -max_angle, max_angle) - theta
            slope = float(gradient @ step)
            if np.all(np.isfinite(step)) and slope < 0:
                for backtrack in range(16):
                    factor = 0.5**backtrack
                    angles = theta + factor * step
                    trial = _profile(
                        _flat_design(base_M, rotated_M, W, At, angles),
                        I,
                        mass,
                        tau,
                    )
                    evaluations += 1
                    if trial.value <= state.value + 1e-4 * factor * slope:
                        candidate = angles, trial
                        break
            if candidate is not None:
                break
            trial_damping *= 10
        if candidate is None:
            break
        theta, state = candidate
        accepted += 1
    return theta, state, accepted, evaluations, solves


def solve(
    index_init: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    *,
    mass: np.ndarray | None = None,
    lambda_prox: float = 0.05,
    max_steps: int = 5,
    tol: float = 1e-6,
    method: str = "flat_gn",
    angle_backend: str = "schur",
    rank: int | None = None,
    energy_tol: float = 0.05,
    local_ridge: float = 0.0,
    max_angle: float = 0.5,
    workspace_bytes: int = 16 * 1024**2,
    flat_max_steps: int = 8,
) -> HPAOResult:
    """Минимизировать sum mass*min_g(||I-U Y g||²+local_ridge||g||²).

    По умолчанию пять шагов: измеренный бюджет для outer warm start;
    отдельное frozen решение может требовать большего max_steps.
    index_init/result.index — (m,d), Y — (d,m). lambda_prox — положительное
    GN damping, не постоянный штраф в статистическом функционале.
    rank_one: Schur/обычный refit (абляция I); spectral: adaptive-q геодезика;
    core_gn: полный q*m core Jacobian (II), polar retraction + Armijo.
    flat_gn: до flat_max_steps совместных q-angle GN внутри одного cached flat,
    точная экспонента, full profiled Jacobian и Armijo на истинном профиле.
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
        ("flat_max_steps", flat_max_steps),
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
    if method not in {"rank_one", "spectral", "core_gn", "flat_gn"}:
        raise ValueError("method must be rank_one, spectral, core_gn or flat_gn")
    if angle_backend not in {"schur", "refit"}:
        raise ValueError("angle_backend must be schur or refit")
    U = np.ascontiguousarray(U)
    root_mass = np.sqrt(mass)[:, None]
    tangent_limit = np.tan(max_angle)
    Y = _normalize_index(B).T
    m, d = B.shape
    state = _profile(_project(U, Y), I, mass, local_ridge)
    history = [state.value]
    q_history: list[int] = []
    evaluations = gn_solves = fallback_rank = fallback_memory = 0
    flat_iterations = flat_fallbacks = 0
    flat_angles: list[tuple[float, ...]] = []
    accepted = 0
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
        current = method
        if current in {"core_gn", "flat_gn"} and not state.smooth:
            current = "rank_one"
            fallback_rank += 1
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
        if current == "flat_gn":
            rows = len(I) * (I.shape[1] + (m if local_ridge else 0))
            # D, dg/rhs и refit scratch + Jacobian/augmented lstsq copies.
            estimate = 8 * (
                4 * rows * q + 6 * len(I) * (I.shape[1] + m) * (m + q) + q**2
            )
            if estimate > workspace_bytes:
                current = "spectral"
                fallback_memory += 1
            else:
                L, _, At = np.linalg.svd(reduced_gradient, full_matrices=False)
                rotation_V, rotation_W = V @ L, W @ L
                theta, trial, inner_steps, neval, nsolve = _flat_optimize(
                    state,
                    rotation_W,
                    At,
                    I,
                    mass,
                    local_ridge,
                    damping=lambda_prox,
                    max_angle=max_angle,
                    max_steps=flat_max_steps,
                    tol=tol,
                )
                flat_iterations += inner_steps
                evaluations += neval
                gn_solves += nsolve
                if inner_steps and trial.value < state.value:
                    delta = (Y @ At.T) * (np.cos(theta) - 1)
                    delta += rotation_V * np.sin(theta)
                    candidate = Y + delta @ At, trial
                    flat_angles.append(tuple(float(t) for t in theta))
                else:
                    current = "spectral"
                    flat_fallbacks += 1
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
            a = At[0]
            # V from QR; choose sign by actual derivative, not QR convention.
            sign = -1.0 if float((V[:, 0] @ G) @ a) > 0 else 1.0
            v, w = sign * V[:, 0], sign * W[:, :, 0]
            curve = _AngleProfile(state.M, w, a, I, mass, local_ridge)

            def curve_value(t: float, curve: _AngleProfile = curve) -> float:
                if angle_backend == "schur":
                    return curve.value(t)
                curve.evaluations += 1
                return _profile(curve.design(t), I, mass, local_ridge).value

            result = cast(
                OptimizeResult,
                minimize_scalar(
                    curve_value,
                    bounds=(0.0, max_angle),
                    method="bounded",
                    options={"xatol": max(1e-10, tol * 0.01)},
                ),
            )
            t = float(result.x)
            slope = float(v @ G @ a)
            for backtrack in range(16):
                angle = t * 0.5**backtrack
                Yn = Y + (Y @ a * (np.cos(angle) - 1) + v * np.sin(angle))[:, None] * a
                trial = _profile(curve.design(angle), I, mass, local_ridge)
                evaluations += 1
                if trial.value <= state.value + 1e-4 * angle * slope:
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
        "method": method,
        "angle_backend": angle_backend,
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
    if method == "flat_gn":
        diagnostics.update(
            flat_max_steps=flat_max_steps,
            flat_inner_iterations=flat_iterations,
            flat_angles=tuple(flat_angles),
            flat_fallbacks=flat_fallbacks,
        )
    return HPAOResult(Y.T, state.coefficients, diagnostics)


def horizontal(Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return Z - Y @ (Y.T @ Z)


def packed(state: base._Profile, mass: np.ndarray, tau: float) -> np.ndarray:
    residual = state.residual
    if tau:
        residual = np.concatenate(
            (residual, -np.sqrt(tau) * state.coefficients), axis=1
        )
    return (np.sqrt(mass)[:, None] * residual).ravel()


class ProfileJacobian(LinearOperator):
    """Полный horizontal Jacobian, ambient координаты (d,m).

    normal_solve использует SVD/QR, не normal matrix. Рабочие массивы
    O(J*p*m+d*m); adjoint GEMM scratch чанкуется до примерно 4 MiB.
    """

    def __init__(
        self,
        Y: np.ndarray,
        U: np.ndarray,
        state: base._Profile,
        mass: np.ndarray,
        tau: float,
    ) -> None:
        if not state.smooth:
            raise ValueError("profile Jacobian requires safe full local rank")
        self.Y, self.U, self.state, self.tau = Y, U, state, tau
        self.root = np.sqrt(mass)[:, None]
        self.J, self.p, self.d = U.shape
        self.m = Y.shape[1]
        self.rows = self.p + (self.m if tau else 0)
        self.forward_calls = self.adjoint_calls = 0
        super().__init__(dtype=np.dtype(float), shape=(self.J * self.rows, Y.size))

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        self.forward_calls += 1
        state = self.state
        T = base._project(self.U, horizontal(self.Y, x.reshape(self.d, self.m)))
        Tg = (T @ state.coefficients[..., None]).squeeze(-1)
        rhs = (T.swapaxes(1, 2) @ state.residual[..., None]) - (
            state.M.swapaxes(1, 2) @ Tg[..., None]
        )
        dg = state.normal_solve(rhs).squeeze(-1)
        dr = -Tg - (state.M @ dg[..., None]).squeeze(-1)
        if self.tau:
            dr = np.concatenate((dr, -np.sqrt(self.tau) * dg), axis=1)
        result = (self.root * dr).ravel()
        if not np.all(np.isfinite(result)):
            raise RuntimeError("non-finite profiled Jacobian action")
        return result

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        self.adjoint_calls += 1
        state = self.state
        dual = self.root * x.reshape(self.J, self.rows)
        a = dual[:, : self.p]
        rhs = -(state.M.swapaxes(1, 2) @ a[..., None])
        if self.tau:
            rhs -= np.sqrt(self.tau) * dual[:, self.p :, None]
        h = state.normal_solve(rhs).squeeze(-1)
        adjusted = a + (state.M @ h[..., None]).squeeze(-1)
        Z = np.zeros((self.d, self.m))
        chunk = max(1, (4 * 1024**2) // (8 * self.p * self.m))
        for start in range(0, self.J, chunk):
            stop = min(self.J, start + chunk)
            data = (
                state.residual[start:stop, :, None] * h[start:stop, None, :]
                - adjusted[start:stop, :, None]
                * state.coefficients[start:stop, None, :]
            )
            Z += self.U[start:stop].reshape(-1, self.d).T @ data.reshape(-1, self.m)
        result = horizontal(self.Y, Z).ravel()
        if not np.all(np.isfinite(result)):
            raise RuntimeError("non-finite profiled Jacobian adjoint")
        return result


def polar(
    Y: np.ndarray, M: np.ndarray, Z: np.ndarray, T: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Polar(Y+alpha*Z), Z horizontal; cached U action сохраняется."""
    values, vectors = np.linalg.eigh(np.eye(Y.shape[1]) + alpha**2 * (Z.T @ Z))
    root = (vectors * (1 / np.sqrt(values))) @ vectors.T
    return (Y + alpha * Z) @ root, (M + alpha * T) @ root


def solve_full_gn(
    index_init: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    *,
    mass: np.ndarray | None = None,
    lambda_prox: float = 0.05,
    max_steps: int = 5,
    tol: float = 1e-6,
    local_ridge: float = 0.0,
    max_angle: float = 0.5,
    lsmr_maxiter: int = 30,
) -> HPAOResult:
    """Full GN, true-profile Armijo; при unsafe/неуспехе один baseline шаг.

    local_ridge — прежний estimator option. Krylov tol=1e-9, certificate
    <=1e-6; cap/отказы явно диагностируются. Никакой global guarantee.
    """
    if hasattr(U, "__cuda_array_interface__"):
        raise NotImplementedError("full GN prototype requires CPU statistics")
    B, U, I, mass = _validate_inputs(index_init, U, I, mass)
    assert mass is not None
    if B.ndim != 2:
        raise ValueError("full GN requires multi-index row-basis")
    for name, count in (("max_steps", max_steps), ("lsmr_maxiter", lsmr_maxiter)):
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if count <= 0:
            raise ValueError(f"{name} must be positive")
    for name, value in (
        ("lambda_prox", lambda_prox),
        ("tol", tol),
        ("max_angle", max_angle),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if max_angle >= np.pi / 2 or not np.isfinite(local_ridge) or local_ridge < 0:
        raise ValueError("require max_angle<pi/2 and finite local_ridge>=0")
    U = np.ascontiguousarray(U)
    Y = _normalize_index(B).T
    m = Y.shape[1]
    state = base._profile(base._project(U, Y), I, mass, local_ridge)
    scale = max(1.0, float(np.sum(mass[:, None] * I**2)))
    history = [state.value]
    inner_stops, inner_iterations, certificates = [], [], []
    fallbacks: list[str] = []
    forward_calls = adjoint_calls = evaluations = accepted = 0
    stop = "max_steps"
    for _ in range(max_steps):
        G = base._gradient(Y, U, state, mass)
        if np.linalg.norm(G) <= tol * scale or Y.shape[0] == m:
            stop = (
                "stationary"
                if state.smooth or Y.shape[0] == m
                else "rank_boundary_stationary"
            )
            break
        candidate = None
        fallback = "rank_guard"
        if state.smooth:
            operator = ProfileJacobian(Y, U, state, mass, local_ridge)
            residual = packed(state, mass, local_ridge)
            # SciPy's current type export resolves lsmr as a module in Pyright.
            krylov_solve = cast(Callable[..., tuple], lsmr)
            result = krylov_solve(
                operator,
                -residual,
                damp=np.sqrt(lambda_prox),
                atol=1e-9,
                btol=1e-9,
                maxiter=lsmr_maxiter,
            )
            x = result[0]
            rhs_norm = float(np.linalg.norm(operator.rmatvec(residual)))
            normal = operator.rmatvec(operator.matvec(x) + residual) + lambda_prox * x
            certificate = float(np.linalg.norm(normal)) / max(
                np.finfo(float).tiny, rhs_norm + lambda_prox * np.linalg.norm(x)
            )
            inner_stops.append(int(result[1]))
            inner_iterations.append(int(result[2]))
            certificates.append(certificate)
            fallback = "inner_certificate"
            if np.all(np.isfinite(x)) and certificate <= 1e-6:
                Z = horizontal(Y, x.reshape(Y.shape))
                length = np.linalg.norm(Z, ord=2)
                if length > np.tan(max_angle):
                    Z *= np.tan(max_angle) / length
                slope = float(np.sum(G * Z))
                fallback = "line_search"
                if slope < 0:
                    T = base._project(U, Z)
                    for backtrack in range(16):
                        alpha = 0.5**backtrack
                        Yn, Mn = polar(Y, state.M, Z, T, alpha)
                        trial = base._profile(Mn, I, mass, local_ridge)
                        evaluations += 1
                        if trial.value <= state.value + 1e-4 * alpha * slope:
                            candidate = Yn, trial
                            break
            forward_calls += operator.forward_calls
            adjoint_calls += operator.adjoint_calls
        if candidate is None:
            fallbacks.append(fallback)
            reference = base.solve(
                Y.T,
                U,
                I,
                mass=mass,
                lambda_prox=lambda_prox,
                max_steps=1,
                tol=tol,
                local_ridge=local_ridge,
                max_angle=max_angle,
            )
            if not reference.diagnostics["accepted_steps"]:
                stop = str(reference.diagnostics["stop_reason"])
                break
            Yn = reference.index.T
            candidate = Yn, base._profile(base._project(U, Yn), I, mass, local_ridge)
        Y, state = candidate
        accepted += 1
        history.append(state.value)
    state = base._profile(base._project(U, Y), I, mass, local_ridge)
    gradient_norm = float(np.linalg.norm(base._gradient(Y, U, state, mass)))
    applicable = state.smooth or Y.shape[0] == m
    converged = applicable and gradient_norm <= tol * scale
    if converged:
        stop = "stationary"
    return HPAOResult(
        Y.T,
        state.coefficients,
        {
            "linear_solver": "glm_full_gn",
            "loss": 0.5 * float(np.sum(mass[:, None] * state.residual**2)),
            "profile_objective": state.value,
            "profile_history": tuple(history),
            "loss_history": tuple(v / 2 for v in history),
            "converged": converged,
            "stop_reason": stop,
            "accepted_steps": accepted,
            "iterations": accepted,
            "riemannian_gradient": gradient_norm / scale,
            "gradient_norm": gradient_norm,
            "stationarity_applicable": applicable,
            "local_rank_loss": int(np.count_nonzero(state.ranks < m)),
            "orthogonality_error": float(np.linalg.norm(Y.T @ Y - np.eye(m))),
            "lsmr_stop": tuple(inner_stops),
            "lsmr_iterations": tuple(inner_iterations),
            "normal_residual": tuple(certificates),
            "fallbacks": tuple(fallbacks),
            "forward_calls": forward_calls,
            "adjoint_calls": adjoint_calls,
            "profile_evaluations": evaluations,
            "local_ridge": local_ridge,
        },
    )
