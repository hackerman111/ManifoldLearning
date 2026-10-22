"""Проверенный CPU Grassmann: QR/SVD профиль, cached core GN и Armijo.

EXACT/NUMERICAL оптимизация прежней finite-sketch цели и bounded optimizer.
solve: core_gn по умолчанию; relative_gn=True — явный APPROXIMATE вариант.
local_ridge>0 — ESTIMATOR вариант. Row-basis (m,d).
block_rank_k: отдельный opt-in вход в экспериментальный shared solver.
Экспериментальные flat/full tangent GN находятся в grassman_exp.py.
Эталон и paired evidence: experiments/grassman_optim_2026_10_01/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.optimize import OptimizeResult, minimize_scalar

from .LSMR import HPAOResult, _normalize_index, _validate_inputs

_GRADIENT_WORKSPACE_BYTES = 4 * 1024**2


@dataclass(slots=True)
class _Profile:
    """Малые SVD/QR-факторы; local cutoff соответствует lstsq(rcond=None)."""

    M: np.ndarray  # (J,p,m)
    coefficients: np.ndarray  # (J,m)
    residual: np.ndarray  # (J,p)
    right: np.ndarray | None
    inverse_squared: np.ndarray | None
    ranks: np.ndarray
    smooth: bool
    value: float  # sum mass * (residual**2 + tau*coefficients**2)
    triangular: np.ndarray | None = None  # (J,m,m), только safe m<=2

    def normal_solve(self, rhs: np.ndarray) -> np.ndarray:
        """Применить (M.T M + tau I)^-1 через локальные SVD/QR-факторы."""
        if self.triangular is not None:
            R = self.triangular
            if R.shape[1] == 1:
                return rhs / R[:, :1, :1] / R[:, :1, :1]
            # Два треугольных solve, без формирования normal matrix/inverse.
            a, b, c = R[:, 0, 0, None], R[:, 0, 1, None], R[:, 1, 1, None]
            z0 = rhs[:, 0] / a
            z1 = (rhs[:, 1] - b * z0) / c
            result = np.empty_like(rhs)
            result[:, 1] = z1 / c
            result[:, 0] = (z0 - b * result[:, 1]) / a
            return result
        assert self.right is not None and self.inverse_squared is not None
        coordinates = self.right @ rhs
        return self.right.swapaxes(1, 2) @ (
            self.inverse_squared[:, :, None] * coordinates
        )


def _profile_svd(
    M: np.ndarray, I: np.ndarray, mass: np.ndarray, tau: float
) -> _Profile:
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


def _profile(M: np.ndarray, I: np.ndarray, mass: np.ndarray, tau: float) -> _Profile:
    """NUMERICAL: reorthogonalized QR для m<=2; иначе исходный SVD.

    Быстрый путь требует cond(R)^2*eps < sqrt(eps). Более чувствительные
    задачи целиком идут в эталон, включая minimum-norm и rank/smooth guards.
    Два столбца ортогонализуются повторно; normal equations не строятся.
    """
    J, p, m = M.shape
    if m > 2 or p < m:
        return _profile_svd(M, I, mass, tau)
    if tau:
        design = np.concatenate(
            (M, np.broadcast_to(np.sqrt(tau) * np.eye(m), (J, m, m))), axis=1
        )
        target = np.concatenate((I, np.zeros((J, m))), axis=1)
    else:
        design, target = M, I
    norms = np.linalg.norm(design, axis=1)  # (J,m)
    a = norms[:, 0]
    guard = 64 * np.finfo(float).eps * np.maximum(1.0, np.linalg.norm(norms, axis=1))
    if np.any(a <= guard) or not np.all(np.isfinite(norms)):
        return _profile_svd(M, I, mass, tau)
    Q = np.empty_like(design)
    Q[:, :, 0] = design[:, :, 0] / a[:, None]
    R = np.zeros((J, m, m))
    R[:, 0, 0] = a
    if m == 2:
        b = np.einsum("jp,jp->j", Q[:, :, 0], design[:, :, 1])
        Q[:, :, 1] = design[:, :, 1] - Q[:, :, 0] * b[:, None]
        correction = np.einsum("jp,jp->j", Q[:, :, 0], Q[:, :, 1])
        Q[:, :, 1] -= Q[:, :, 0] * correction[:, None]
        b += correction
        c = np.linalg.norm(Q[:, :, 1], axis=1)
        # Сингулярные числа 2x2 upper-triangular: hypot без вычитания корней.
        largest = (np.hypot(a + c, b) + np.hypot(a - c, b)) * 0.5
        smallest = a * (c / largest)
        if np.any(smallest <= np.maximum(guard, np.finfo(float).eps ** 0.25 * largest)):
            return _profile_svd(M, I, mass, tau)
        Q[:, :, 1] /= c[:, None]
        R[:, 0, 1], R[:, 1, 1] = b, c
    coordinates = (Q.swapaxes(1, 2) @ target[..., None]).squeeze(-1)
    coefficients = np.empty((J, m))
    if m == 2:
        coefficients[:, 1] = coordinates[:, 1] / R[:, 1, 1]
        coefficients[:, 0] = (coordinates[:, 0] - R[:, 0, 1] * coefficients[:, 1]) / a
    else:
        coefficients[:, 0] = coordinates[:, 0] / a
    residual = I - (M @ coefficients[..., None]).squeeze(-1)
    value = float(
        np.sum(
            mass * (np.sum(residual**2, axis=1) + tau * np.sum(coefficients**2, axis=1))
        )
    )
    if not np.isfinite(value) or not np.all(np.isfinite(coefficients)):
        raise RuntimeError("non-finite local profile")
    return _Profile(
        M, coefficients, residual, None, None, np.full(J, m), True, value, R
    )


def _gradient(
    Y: np.ndarray, U: np.ndarray, state: _Profile, mass: np.ndarray
) -> np.ndarray:
    # sum_j (U_j.T r_j) (mass_j g_j).T: один проход по U вместо m.
    # Adjoint и weighted coefficients ограничены 4 MiB; cache не сохраняется.
    J, _, d = U.shape
    m = Y.shape[1]
    chunk = max(1, _GRADIENT_WORKSPACE_BYTES // (8 * (d + m)))
    G = np.zeros((d, m))
    for start in range(0, J, chunk):
        stop = min(start + chunk, J)
        adjoint = (
            U[start:stop].swapaxes(1, 2) @ state.residual[start:stop, :, None]
        ).squeeze(-1)  # (chunk,d)
        weighted = mass[start:stop, None] * state.coefficients[start:stop]
        G -= 2 * (adjoint.T @ weighted)
    return G - Y @ (Y.T @ G)


def _project(U: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """U@basis через wide GEMM; U contiguous, flatten является view.

    Замер frozen J800/p10/d50/k2: transpose GEMM ~3x быстрее tall GEMM.
    Копируется только малый результат (J,p,k), чтобы local kernels получали
    contiguous последние оси; дополнительный temporary того же размера.
    """
    J, p, d = U.shape
    projected = (basis.T @ U.reshape(-1, d).T).T
    return np.ascontiguousarray(projected).reshape(J, p, basis.shape[1])


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
        self.u = u
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
        self.guard = 64 * np.finfo(float).eps * (self.a2 + self.e + tau)
        self.evaluations = 0

    def design(self, t: float) -> np.ndarray:
        delta = self.u * (np.cos(t) - 1) + self.w * np.sin(t)
        return self.M + delta[:, :, None] * self.a

    def value(self, t: float) -> float:
        self.evaluations += 1
        ct, st = np.cos(t), np.sin(t)
        denominator = self.a2 * ct**2 + 2 * self.d * ct * st + self.e * st**2 + self.tau
        if not self.safe or np.any(denominator <= self.guard):
            return _profile(self.design(t), self.I, self.mass, self.tau).value
        numerator = self.b * ct + self.c * st
        return float(self.mass @ (self.k - numerator**2 / denominator))


def _core_jacobian(
    state: _Profile,
    W: np.ndarray,
    mass: np.ndarray,
    tau: float,
    *,
    root_mass: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Полный eq:dg Jacobian для Z=V K, K=(q,m); все cross terms сохранены."""
    if not state.smooth:
        raise ValueError(
            "full-rank profile derivative requires well-conditioned local systems"
        )
    J, p, m = state.M.shape
    q = W.shape[2]
    rows = p + (m if tau else 0)
    # eq:dg сразу для всех (k,l); порядок столбцов тот же, что K.ravel().
    mtw = state.M.swapaxes(1, 2) @ W  # (J,m,q)
    wr = (state.residual[:, None, :] @ W).squeeze(1)  # (J,q)
    rhs = -mtw[:, :, :, None] * state.coefficients[:, None, None, :]
    for l in range(m):
        rhs[:, l, :, l] += wr
    del mtw, wr  # Освобождаем scratch до normal solve / сборки jac.
    dg = state.normal_solve(rhs.reshape(J, m, q * m))  # (J,m,q*m)
    del rhs
    jac = np.empty((J, rows, q * m))
    jac[:, :p] = (-W[:, :, :, None] * state.coefficients[:, None, None, :]).reshape(
        J, p, q * m
    )
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
    root = np.sqrt(mass)[:, None] if root_mass is None else root_mass
    jac *= root[:, :, None]
    return jac.reshape(J * rows, q * m), (root * residual).ravel()


def _polar(
    Y: np.ndarray, M: np.ndarray, V: np.ndarray, W: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Polar retraction and its cached operator action; no d*d matrix."""
    values, vectors = np.linalg.eigh(np.eye(Y.shape[1]) + K.T @ K)
    root = (vectors * (1 / np.sqrt(values))) @ vectors.T
    return (Y + V @ K) @ root, (M + W @ K) @ root


def _intrinsic_model(
    U: np.ndarray,
    I: np.ndarray,
    mass: np.ndarray,
    state: _Profile,
    cached_ranks: np.ndarray,
) -> tuple[np.ndarray, _Profile] | None:
    """Только GN proposal: численный rank(U)=rank(UY)<m, tau=0.

    Неполноранговые центры остаются в полном профиле/градиенте/сертификате.
    SVD входов: chunk до 4MiB (либо один больший центр).
    """
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


def solve(
    index_init: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    *,
    mass: np.ndarray | None = None,
    lambda_prox: float = 0.05,
    max_steps: int = 20,
    tol: float = 1e-6,
    rank: int | None = None,
    energy_tol: float = 0.05,
    local_ridge: float = 0.0,
    max_angle: float = 0.5,
    workspace_bytes: int = 16 * 1024**2,
    relative_gn: bool = True,
    quadratic_trials: int = 3,
    constant_centers: bool = True,
    smooth_max_steps: int | None = None,
) -> HPAOResult:
    """Минимизировать sum mass*min_g(||I-U Y g||²+local_ridge||g||²).

    По умолчанию двадцать шагов; отдельный smooth cap отключён;
    отдельное frozen решение может требовать большего max_steps.
    index_init/result.index — (m,d), Y — (d,m). lambda_prox — положительное
    GN damping, не постоянный штраф в статистическом функционале.
    core_gn: полный q*m core Jacobian (II), polar retraction + Armijo.
    При tau=0 около rank boundary используется value-based rank-one вместо GN.
    При лимите workspace GN заменяется spectral; причина видна в diagnostics.
    Кэши O(J*p*(m+q)); GN workspace O(J*(p+m)*q*m), ограничен явно.
    Ограниченные шаги возвращают converged=False; глобальный минимум не обещан.

    relative_gn=True: regular-center GN при численном intrinsic rank loss,
    damping=lambda_prox*||J||F²/(q*m), полный true-profile Armijo. Иначе direct
    rank-one Armijo; smooth GN прежний. quadratic_trials=3 выбирает лучшую
    фактически принятую Armijo квадратичную пробу. constant_centers=True
    исключает только U_j=0, восстанавливая loss/coefs и gradient scale.
    Бюджет 20 измерен отдельно от состава поиска; EDR не монотонна по бюджету.
    Прежний режим: max_steps=5, relative_gn=False, quadratic_trials=0,
    constant_centers=False. Evidence: grassman_accuracy_2026_10_18/COMBINED_*.
    """
    if hasattr(U, "__cuda_array_interface__"):
        raise NotImplementedError("Grassmann solver requires CPU statistics")
    if not isinstance(relative_gn, bool):
        raise TypeError("relative_gn must be a boolean")
    if not isinstance(constant_centers, bool):
        raise TypeError("constant_centers must be a boolean")
    if (
        isinstance(quadratic_trials, bool)
        or not isinstance(quadratic_trials, (int, np.integer))
        or not 0 <= quadratic_trials <= 3
    ):
        raise ValueError("quadratic_trials must be an integer in [0,3]")
    if smooth_max_steps is not None and (
        isinstance(smooth_max_steps, bool)
        or not isinstance(smooth_max_steps, (int, np.integer))
        or smooth_max_steps <= 0
    ):
        raise ValueError("smooth_max_steps must be a positive integer or None")
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
    constant_count = 0
    if constant_centers:
        active = np.any(U != 0, axis=(1, 2))
        constant_count = int(np.count_nonzero(~active))
        if np.any(active) and not np.all(active):
            scale = max(1.0, float(np.sum(mass[:, None] * I**2)))
            active_scale = max(1.0, float(np.sum(mass[active, None] * I[active] ** 2)))
            constant = float(np.sum(mass[~active, None] * I[~active] ** 2))
            result = solve(
                B,
                U[active],
                I[active],
                mass=mass[active],
                lambda_prox=lambda_prox,
                max_steps=max_steps,
                tol=tol * scale / active_scale,
                rank=rank,
                energy_tol=energy_tol,
                local_ridge=local_ridge,
                max_angle=max_angle,
                workspace_bytes=workspace_bytes,
                relative_gn=relative_gn,
                quadratic_trials=quadratic_trials,
                constant_centers=False,
                smooth_max_steps=smooth_max_steps,
            )
            coefficients = np.zeros((len(I), B.shape[0]))
            coefficients[active] = result.coefficients
            diag = dict(result.diagnostics)
            diag["constant_centers"] = int(np.count_nonzero(~active))
            diag["profile_objective"] = (
                cast(float, diag["profile_objective"]) + constant
            )
            diag["loss"] = cast(float, diag["loss"]) + constant / 2
            for key, offset in (
                ("profile_history", constant),
                ("loss_history", constant / 2),
            ):
                diag[key] = tuple(
                    v + offset for v in cast(tuple[float, ...], diag[key])
                )
            if not local_ridge:
                diag["local_rank_loss"] = cast(int, diag["local_rank_loss"]) + int(
                    np.count_nonzero(~active)
                )
            diag["riemannian_gradient"] = cast(float, diag["gradient_norm"]) / scale
            return HPAOResult(result.index, coefficients, diag)
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
    stop = "max_steps"
    scale = max(1.0, float(np.sum(mass[:, None] * I**2)))
    converged = False
    cached_ranks = np.full(len(I), -1, dtype=int) if relative_gn else None
    intrinsic_proposals = normalized_proposals = damping_scale_fallbacks = 0
    step_budget = (
        min(max_steps, smooth_max_steps)
        if (relative_gn or quadratic_trials)
        and state.smooth
        and smooth_max_steps is not None
        else max_steps
    )
    for _ in range(step_budget):
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
        gn_state, gn_mask = state, slice(None)
        if not state.smooth:
            model = (
                _intrinsic_model(U, I, mass, state, cached_ranks)
                if cached_ranks is not None and not local_ridge
                else None
            )
            if model is not None:
                gn_mask, gn_state = model
                intrinsic_proposals += 1
            else:
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
                if relative_gn and not state.smooth:
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
        if current == "rank_one" and (relative_gn or quadratic_trials):
            a = At[0]
            sign = -1.0 if float((V[:, 0] @ G) @ a) > 0 else 1.0
            v, w = sign * V[:, 0], sign * W[:, :, 0]
            u = (state.M @ a).reshape(I.shape)
            slope = float(v @ G @ a)
            angle = max_angle
            for _proposal in range(quadratic_trials):
                delta, sine = np.cos(angle) - 1, np.sin(angle)
                Yn = Y + (Y @ a * delta + v * sine)[:, None] * a
                Mn = state.M + (u * delta + w * sine)[:, :, None] * a
                trial = _profile(Mn, I, mass, local_ridge)
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
            for backtrack in range(20 if candidate is None else 0):
                angle = max_angle * 0.5**backtrack
                delta, sine = np.cos(angle) - 1, np.sin(angle)
                Yn = Y + (Y @ a * delta + v * sine)[:, None] * a
                Mn = state.M + (u * delta + w * sine)[:, :, None] * a
                trial = _profile(Mn, I, mass, local_ridge)
                evaluations += 1
                if trial.value <= state.value + 1e-4 * angle * slope:
                    candidate = Yn, trial
                    break
        elif current == "rank_one":
            a = At[0]
            # V from QR; choose sign by actual derivative, not QR convention.
            sign = -1.0 if float((V[:, 0] @ G) @ a) > 0 else 1.0
            v, w = sign * V[:, 0], sign * W[:, :, 0]
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
        "method": "core_gn",
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
    if relative_gn:
        diagnostics.update(
            method="core_gn_relative",
            angle_backend="refit_armijo",
            effective_max_steps=step_budget,
            intrinsic_gn_proposals=intrinsic_proposals,
            normalized_gn_proposals=normalized_proposals,
            damping_scale_fallbacks=damping_scale_fallbacks,
        )
    if quadratic_trials:
        diagnostics.update(
            angle_backend="refit_quadratic",
            quadratic_trials=quadratic_trials,
            effective_max_steps=step_budget,
        )
    if constant_centers:
        diagnostics["constant_centers"] = constant_count
    return HPAOResult(Y.T, state.coefficients, diagnostics)


def block_rank_k(
    index_init: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    *,
    mass: np.ndarray | None = None,
    lambda_prox: float = 0.05,
    max_steps: int = 5,
    tol: float = 1e-6,
    rank_k: int | None = None,
    rank_strategy: str = "energy",
    block_variant: str = "optimized",
    block_max_steps: int = 2,
    energy_tol: float = 0.05,
    local_ridge: float = 0.0,
    max_angle: float = 0.5,
    workspace_bytes: int = 16 * 1024**2,
) -> HPAOResult:
    """Явный экспериментальный rank-k solver; solve/core_gn не изменён.

    rank_k=None: energy или bounded cost-пилот; иначе фиксированный ранг.
    spectral — общая длина, optimized — совместные углы, local refit.
    workspace_bytes ограничивает дополнительный GN scratch, не вход U и
    базовое состояние профиля. Алгоритм и диагностики — в grassman_exp.
    """
    from .grassman_exp import solve as experimental_solve

    return experimental_solve(
        index_init,
        U,
        I,
        mass=mass,
        lambda_prox=lambda_prox,
        max_steps=max_steps,
        tol=tol,
        method="block_rank_k",
        rank_k=rank_k,
        rank_strategy=rank_strategy,
        block_variant=block_variant,
        block_max_steps=block_max_steps,
        energy_tol=energy_tol,
        local_ridge=local_ridge,
        max_angle=max_angle,
        workspace_bytes=workspace_bytes,
    )
