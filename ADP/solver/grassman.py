"""Профильный Grassmann multi-index: улучшения I/II EDR_unified_theory.tex.

APPROXIMATE: ограниченный optimizer прежней finite-sketch цели при tau=0.
local_ridge>0 — явный ESTIMATOR вариант. CPU, row-basis (m,d), HPAOResult.
Ни permanent chordal penalty, ни изменение статистик здесь не вводятся.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.optimize import OptimizeResult, minimize_scalar

from .LSMR import HPAOResult, _normalize_index, _validate_inputs


@dataclass(slots=True)
class _Profile:
    """Малые SVD-факторы; local cutoff соответствует lstsq(rcond=None)."""

    M: np.ndarray  # (J,p,m)
    coefficients: np.ndarray  # (J,m)
    residual: np.ndarray  # (J,p)
    right: np.ndarray
    inverse_squared: np.ndarray
    ranks: np.ndarray
    smooth: bool
    value: float  # sum mass * (residual**2 + tau*coefficients**2)

    def normal_solve(self, rhs: np.ndarray) -> np.ndarray:
        """Применить (M.T M + tau I)^-1 через локальные SVD-факторы."""
        coordinates = self.right @ rhs
        return self.right.swapaxes(1, 2) @ (
            self.inverse_squared[:, :, None] * coordinates
        )


def _profile(M: np.ndarray, I: np.ndarray, mass: np.ndarray, tau: float) -> _Profile:
    J, p, m = M.shape
    if tau:
        design = np.concatenate(
            (M, np.broadcast_to(np.sqrt(tau) * np.eye(m), (J, m, m))), axis=1
        )
        target = np.concatenate((I, np.zeros((J, m))), axis=1)
    else:
        design, target = M, I
    left, singular, right = np.linalg.svd(design, full_matrices=False)
    cutoff = np.finfo(float).eps * max(p, m)
    keep = singular > cutoff * singular[:, :1]
    inverse = np.divide(1.0, singular, out=np.zeros_like(singular), where=keep)
    coordinates = (left.swapaxes(1, 2) @ target[..., None]).squeeze(-1)
    coefficients = (right.swapaxes(1, 2) @ (coordinates * inverse)[..., None]).squeeze(
        -1
    )
    ranks = np.count_nonzero(keep, axis=1)
    # Match existing lstsq fallback in sensitive unregularized neighborhoods.
    sensitive = singular[:, -1] <= np.sqrt(np.finfo(float).eps) * singular[:, 0]
    if not tau:
        for j in np.flatnonzero(sensitive):
            coefficients[j], _, ranks[j], _ = np.linalg.lstsq(M[j], I[j], rcond=None)
    residual = I - (M @ coefficients[..., None]).squeeze(-1)
    value = float(
        np.sum(
            mass * (np.sum(residual**2, axis=1) + tau * np.sum(coefficients**2, axis=1))
        )
    )
    # GN uses full-rank derivative only, comfortably away from cutoff.
    absolute_guard = (
        64 * np.finfo(float).eps * np.maximum(1.0, np.linalg.norm(design, axis=(1, 2)))
    )
    smooth = bool(
        np.all(ranks == m)
        and not np.any(sensitive)
        and np.all(singular[:, -1] > absolute_guard)
    )
    if not np.isfinite(value) or not np.all(np.isfinite(coefficients)):
        raise RuntimeError("non-finite local profile")
    return _Profile(
        M,
        coefficients,
        residual,
        right,
        inverse**2 if smooth else np.zeros_like(inverse),
        ranks,
        smooth,
        value,
    )


def _gradient(
    Y: np.ndarray, U: np.ndarray, state: _Profile, mass: np.ndarray
) -> np.ndarray:
    weighted = (
        mass[:, None, None]
        * state.residual[:, :, None]
        * state.coefficients[:, None, :]
    ).reshape(-1, Y.shape[1])
    G = -2 * (U.reshape(-1, Y.shape[0]).T @ weighted)
    return G - Y @ (Y.T @ G)


class _AngleProfile:
    """Schur через ортогональное исключение augmented постоянных столбцов."""

    def __init__(
        self,
        M: np.ndarray,
        w: np.ndarray,
        a: np.ndarray,
        I: np.ndarray,
        mass: np.ndarray,
        tau: float,
    ) -> None:
        self.M, self.w, self.a, self.I, self.mass, self.tau = M, w, a, I, mass, tau
        J, p, m = M.shape
        complement = np.linalg.svd(a[None, :], full_matrices=True)[2][1:].T
        C = M @ complement
        u = (M @ a).reshape(J, p)
        data = np.stack((I, u, w), axis=-1)  # (J,p,3)
        self.safe = True
        if m > 1:
            if tau:
                C = np.concatenate(
                    (
                        C,
                        np.broadcast_to(
                            np.sqrt(tau) * np.eye(m - 1), (J, m - 1, m - 1)
                        ),
                    ),
                    axis=1,
                )
                data = np.concatenate((data, np.zeros((J, m - 1, 3))), axis=1)
            left, singular, _ = np.linalg.svd(C, full_matrices=False)
            cutoff = np.finfo(float).eps * max(p, m)
            keep = singular > cutoff * singular[:, :1]
            self.safe = bool(
                tau
                or (
                    np.all(keep)
                    and np.all(
                        singular[:, -1] > np.sqrt(np.finfo(float).eps) * singular[:, 0]
                    )
                )
            )
            coordinates = left.swapaxes(1, 2) @ data
            data = data - left @ (coordinates * keep[:, :, None])
        gram = data.swapaxes(1, 2) @ data  # PSD bilinear forms, no cancellation
        self.k, self.b, self.c = gram[:, 0, 0], gram[:, 0, 1], gram[:, 0, 2]
        self.a2, self.d, self.e = gram[:, 1, 1], gram[:, 1, 2], gram[:, 2, 2]
        self.evaluations = 0

    def design(self, t: float) -> np.ndarray:
        delta = (self.M @ self.a) * (np.cos(t) - 1) + self.w * np.sin(t)
        return self.M + delta[:, :, None] * self.a

    def value(self, t: float) -> float:
        self.evaluations += 1
        ct, st = np.cos(t), np.sin(t)
        denominator = self.a2 * ct**2 + 2 * self.d * ct * st + self.e * st**2 + self.tau
        guard = 64 * np.finfo(float).eps * (self.a2 + self.e + self.tau)
        if not self.safe or np.any(denominator <= guard):
            return _profile(self.design(t), self.I, self.mass, self.tau).value
        numerator = self.b * ct + self.c * st
        return float(self.mass @ (self.k - numerator**2 / denominator))


def _core_jacobian(
    state: _Profile, W: np.ndarray, mass: np.ndarray, tau: float
) -> tuple[np.ndarray, np.ndarray]:
    """Полный eq:dg Jacobian для Z=V K, K=(q,m); все cross terms сохранены."""
    if not state.smooth:
        raise ValueError(
            "full-rank profile derivative requires well-conditioned local systems"
        )
    J, p, m = state.M.shape
    q = W.shape[2]
    rows = p + (m if tau else 0)
    jac = np.empty((J, rows, q * m))
    for k in range(q):
        w = W[:, :, k]
        mtw = (state.M.swapaxes(1, 2) @ w[..., None]).squeeze(-1)
        wr = np.einsum("jp,jp->j", w, state.residual)
        for l in range(m):
            rhs = -mtw * state.coefficients[:, l, None]
            rhs[:, l] += wr
            dg = state.normal_solve(rhs[..., None]).squeeze(-1)
            jac[:, :p, k * m + l] = -w * state.coefficients[:, l, None] - (
                state.M @ dg[..., None]
            ).squeeze(-1)
            if tau:
                jac[:, p:, k * m + l] = -np.sqrt(tau) * dg
    residual = (
        state.residual
        if not tau
        else np.concatenate(
            (state.residual, -np.sqrt(tau) * state.coefficients), axis=1
        )
    )
    root = np.sqrt(mass)[:, None]
    return (jac * root[:, :, None]).reshape(J * rows, q * m), (root * residual).ravel()


def _polar(
    Y: np.ndarray, M: np.ndarray, V: np.ndarray, W: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Polar retraction and its cached operator action; no d*d matrix."""
    values, vectors = np.linalg.eigh(np.eye(Y.shape[1]) + K.T @ K)
    root = (vectors * (1 / np.sqrt(values))) @ vectors.T
    return (Y + V @ K) @ root, (M + W @ K) @ root


def solve(
    index_init: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    *,
    mass: np.ndarray | None = None,
    lambda_prox: float = 0.05,
    max_steps: int = 5,
    tol: float = 1e-6,
    method: str = "core_gn",
    angle_backend: str = "schur",
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
    rank_one: Schur/обычный refit (абляция I); spectral: adaptive-q геодезика;
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
    for name, value in (("max_steps", max_steps), ("workspace_bytes", workspace_bytes)):
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
    if method not in {"rank_one", "spectral", "core_gn"}:
        raise ValueError("method must be rank_one, spectral or core_gn")
    if angle_backend not in {"schur", "refit"}:
        raise ValueError("angle_backend must be schur or refit")
    U = np.ascontiguousarray(U)
    Y = _normalize_index(B).T
    m, d = B.shape
    state = _profile(U @ Y, I, mass, local_ridge)
    history = [state.value]
    q_history: list[int] = []
    evaluations = gn_solves = fallback_rank = fallback_memory = 0
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
        if current == "core_gn" and not state.smooth:
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
        W = U @ V
        candidate = None
        if current == "core_gn":
            rows = len(I) * (I.shape[1] + (m if local_ridge else 0))
            # Jacobian, weighting copy and augmented lstsq copies are bounded.
            estimate = 8 * (4 * rows * q * m + (q * m) ** 2)
            if estimate > workspace_bytes:
                current = "spectral"
                fallback_memory += 1
            else:
                jac, residual = _core_jacobian(state, W, mass, local_ridge)
                damping = lambda_prox
                for _trial in range(8):
                    design = np.vstack((jac, np.sqrt(damping) * np.eye(q * m)))
                    rhs = np.concatenate((-residual, np.zeros(q * m)))
                    K = np.linalg.lstsq(design, rhs, rcond=None)[0].reshape(q, m)
                    gn_solves += 1
                    length = np.linalg.norm(K, ord=2)
                    if length > np.tan(max_angle):
                        K *= np.tan(max_angle) / length
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
            for backtrack in range(20):
                t = max_angle * 0.5**backtrack
                angles = t * angular_speed
                delta = (Y @ R.T) * (np.cos(angles) - 1) - rotation_V * np.sin(angles)
                delta_M = (state.M @ R.T) * (np.cos(angles) - 1) - rotation_W * np.sin(
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
    state = _profile(U @ Y, I, mass, local_ridge)
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
    return HPAOResult(Y.T, state.coefficients, diagnostics)
