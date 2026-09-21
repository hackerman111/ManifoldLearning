from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import linalg as sparse_linalg

from ADP.solver._multi_operator import (
    adjoint as multi_adjoint,
    forward as multi_forward,
)

# EXACT: matrix-free actions preserve the fixed dense-U objective.
_LOCAL_REFIT_BYTES = 4 * 1024**2


@dataclass(frozen=True, slots=True)
class HPAOResult:
    """Результат внутреннего HPAO-LSMR при фиксированных статистиках."""

    index: np.ndarray
    coefficients: np.ndarray
    diagnostics: dict[str, object]


def lsmr(
    beta_init: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    **settings: Any,
) -> np.ndarray:
    """Сохранить прежний интерфейс, возвращающий только найденный индекс."""
    return solve(beta_init, U, I, **settings).index


def solve(
    index_init: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    *,
    mass: np.ndarray | None = None,
    lambda_prox: float = 1.0,
    max_steps: int = 10,
    tol: float = 1e-6,
    theta: float = 0.1,
    trust_radius: float | None = None,
    lsmr_maxiter: int | None = None,
    linear_solver: str = "lsmr",
    dense_max_unknowns: int = 256,
    dense_max_bytes: int = 64 * 1024**2,
    hybrid_inner_rtol: float | None = None,
) -> HPAOResult:
    """Решить dense-``U`` задачу HPAO-LSMR.

    ``U`` имеет форму ``(J, p, d)``, ``I`` — ``(J, p)``. Single-index
    задаётся вектором ``(d,)``, multi-index — матрицей строк ``P`` формы
    ``(m, d)``. Штраф ``lambda_prox`` применяется к correction-шагу. Он не
    входит в статистический функционал. ``mass`` задаёт внешний множитель
    ``c_j``; при уже ненормированных ``U`` и ``I`` используйте ``mass=None``.
    ``hybrid_inner_rtol`` включает отдельный APPROXIMATE-режим HYBRID:
    сертификат относительной ошибки коррекции; по умолчанию отключён.
    """
    index, U, I, mass = _validate_inputs(index_init, U, I, mass)
    _validate_settings(lambda_prox, max_steps, tol, theta, lsmr_maxiter)
    if linear_solver not in {"lsmr", "hybrid"}:
        raise ValueError("linear_solver must be 'lsmr' or 'hybrid'")
    for name, value in (
        ("dense_max_unknowns", dense_max_unknowns),
        ("dense_max_bytes", dense_max_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if linear_solver == "hybrid" and index.ndim != 2:
        raise ValueError("hybrid currently requires a multi-index matrix")
    if hybrid_inner_rtol is not None:
        if linear_solver != "hybrid":
            raise ValueError("hybrid_inner_rtol requires the hybrid solver")
        if not np.isfinite(hybrid_inner_rtol) or not 0 < hybrid_inner_rtol <= theta:
            raise ValueError("hybrid_inner_rtol must be finite and in (0, theta]")
    index = _normalize_index(index)
    m = 1 if index.ndim == 1 else index.shape[0]
    if trust_radius is None:
        trust_radius = _default_trust_radius(U.shape[2], m)
    if not np.isfinite(trust_radius) or trust_radius <= 0:
        raise ValueError("trust_radius must be finite and positive")

    coefficients, local_ranks = _local_refit(I, U, index)
    loss = _loss(I, U, index, coefficients, mass)
    loss_history = [loss]
    lambda_history: list[float] = []
    rank_loss_history = [int(np.count_nonzero(local_ranks < m))]
    lambda_current = float(lambda_prox)
    lambda_floor = lambda_current / 1e6
    lambda_cap = lambda_current * 1e6
    small_correction_count = 0
    certified_count = 0
    accepted_steps = 0
    accepted_correction_norm = math.inf
    U_norm2 = np.einsum("jpd,jpd->j", U, U, optimize=True)
    gradient = local_gradient = orthogonality = math.inf
    hybrid_calls = 0
    hybrid_iterations = 0
    hybrid_screened = 0
    hybrid_screening_matvecs = 0
    hybrid_recycled_solves = 0
    hybrid_projected_solves = 0
    hybrid_refinements = 0
    hybrid_forward_calls = 0
    hybrid_adjoint_calls = 0
    hybrid_backends: list[str] = []
    last: dict[str, object] = {
        "lsmr_stop": 0,
        "lsmr_iterations": 0,
        "normal_residual_ratio": math.inf,
        "normal_residual": math.inf,
        "correction_norm": math.inf,
        "gauge_error": math.inf,
        "global_rank": m,
    }

    for _ in range(max_steps):
        prior = index
        old_loss = loss
        workspace = None
        if linear_solver == "hybrid":
            from ADP.solver.HYBRID import RidgeWorkspace

            workspace = RidgeWorkspace(
                U,
                I,
                prior,
                coefficients,
                mass,
                max_unknowns=dense_max_unknowns,
                max_bytes=dense_max_bytes,
            )
            hybrid_backends.append(workspace.backend)

        while lambda_current <= lambda_cap:
            # EXACT/NUMERICAL: отклоняем только доказанно слишком длинный
            # ridge-шаг. После первого принятого шага screening нужен лишь
            # при повторной пробе: обычно прежняя lambda уже подходит.
            if (
                workspace is not None
                and lambda_current > 0
                and (accepted_steps == 0 or workspace.calls > 0)
                and workspace.rejects_trust(
                    lambda_current, trust_radius * math.sqrt(m), lsmr_maxiter
                )
            ):
                lambda_current *= 2.0
                continue
            step = (
                workspace.correction(
                    lambda_current, tol, lsmr_maxiter, relative_tol=hybrid_inner_rtol
                )
                if workspace is not None
                else _global_correction(
                    I,
                    U,
                    prior,
                    coefficients,
                    mass,
                    lambda_current,
                    tol,
                    lsmr_maxiter,
                )
            )
            correction = step[0].reshape(prior.shape)
            correction_norm = float(np.linalg.norm(correction) / math.sqrt(m))
            required_ratio = theta if hybrid_inner_rtol is None else hybrid_inner_rtol
            if step[3] > required_ratio or correction_norm > trust_radius:
                if lambda_current == 0:
                    raise RuntimeError(
                        "unregularized HPAO step failed the trust certificate"
                    )
                lambda_current *= 2.0
                continue

            raw_index = prior + correction
            candidate, gauge_coefficients, global_rank = _gauge_fix(
                raw_index, coefficients, prior
            )
            raw_fitted = _predict(U, raw_index, coefficients)
            gauge_fitted = _predict(U, candidate, gauge_coefficients)
            gauge_error = float(
                np.linalg.norm(gauge_fitted - raw_fitted)
                / max(1.0, np.linalg.norm(raw_fitted))
            )
            candidate_coefficients, local_ranks = _local_refit(I, U, candidate)
            candidate_loss = _loss(I, U, candidate, candidate_coefficients, mass)
            rounding = 64 * np.finfo(float).eps * max(1.0, old_loss)
            if candidate_loss > old_loss + rounding:
                if lambda_current == 0:
                    raise RuntimeError(
                        "unregularized HPAO step did not decrease the objective"
                    )
                lambda_current *= 2.0
                continue

            index = candidate
            coefficients = candidate_coefficients
            loss = candidate_loss
            accepted_steps += 1
            accepted_correction_norm = correction_norm
            lambda_history.append(lambda_current)
            loss_history.append(loss)
            rank_loss_history.append(int(np.count_nonzero(local_ranks < m)))
            last = {
                "lsmr_stop": step[1],
                "lsmr_iterations": step[2],
                "normal_residual_ratio": step[3],
                "normal_residual": step[4],
                "correction_norm": correction_norm,
                "gauge_error": gauge_error,
                "global_rank": global_rank,
            }
            break
        else:
            raise RuntimeError(
                "HPAO-LSMR could not produce a certified decreasing step"
            )

        if workspace is not None:
            hybrid_calls += workspace.calls
            hybrid_iterations += workspace.iterations
            hybrid_screened += workspace.screened_trials
            hybrid_screening_matvecs += workspace.screening_matvecs
            hybrid_recycled_solves += workspace.recycled_solves
            hybrid_projected_solves += workspace.projected_solves
            hybrid_refinements += workspace.refinements
            hybrid_forward_calls += workspace.forward_calls
            hybrid_adjoint_calls += workspace.adjoint_calls

        relative_change = abs(old_loss - loss) / max(1.0, old_loss)
        aligned_step = _index_distance(index, prior)
        gradient, local_gradient, orthogonality = _stationarity(
            I, U, index, coefficients, mass, loss, U_norm2=U_norm2
        )
        certified = (
            relative_change < tol
            and max(gradient, local_gradient, orthogonality) < tol
            and aligned_step < tol
        )
        certified_count = certified_count + 1 if certified else 0
        if certified_count == 2:
            break

        if accepted_correction_norm > 0.9 * trust_radius:
            lambda_current = min(2.0 * lambda_current, lambda_cap)
            small_correction_count = 0
        elif accepted_correction_norm < 0.25 * trust_radius and loss < old_loss:
            small_correction_count += 1
            if small_correction_count == 2:
                lambda_current = max(0.5 * lambda_current, lambda_floor)
                small_correction_count = 0
        else:
            small_correction_count = 0

    diagnostics = {
        **last,
        "accepted_steps": accepted_steps,
        "converged": certified_count == 2,
        "loss": loss,
        "loss_history": tuple(loss_history),
        "lambda_history": tuple(lambda_history),
        "local_rank_loss": tuple(rank_loss_history),
        "riemannian_gradient": gradient,
        "local_gradient": local_gradient,
        "orthogonality": orthogonality,
    }
    if linear_solver == "hybrid":
        diagnostics.update(
            linear_solver="hybrid",
            linear_backends=tuple(hybrid_backends),
            linear_solves_total=hybrid_calls,
            linear_iterations_total=hybrid_iterations,
            linear_screened_trials=hybrid_screened,
            linear_screening_matvecs=hybrid_screening_matvecs,
            linear_recycled_solves=hybrid_recycled_solves,
            linear_projected_solves=hybrid_projected_solves,
            linear_refinements=hybrid_refinements,
            linear_forward_calls=hybrid_forward_calls,
            linear_adjoint_calls=hybrid_adjoint_calls,
            hybrid_inner_rtol=hybrid_inner_rtol,
        )
    return HPAOResult(index, coefficients, diagnostics)


def _validate_inputs(index, U, I, mass):
    """Проверить формы и конечность dense-статистик перед solver-ом."""
    for name, value in (("initial index", index), ("U", U), ("I", I)):
        array = np.asarray(value)
        if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} must be a real numeric array")

    index = np.asarray(index, dtype=float)
    U = np.asarray(U, dtype=float)
    I = np.asarray(I, dtype=float)
    if U.ndim != 3:
        raise ValueError("U must have shape (J, p, d)")
    if 0 in U.shape:
        raise ValueError("U dimensions J, p, and d must be positive")
    if I.shape != U.shape[:2]:
        raise ValueError("I must have shape (J, p) matching U")
    if index.ndim == 1:
        valid_index = index.shape == (U.shape[2],)
    elif index.ndim == 2:
        valid_index = index.shape[1] == U.shape[2] and 1 <= index.shape[0] <= U.shape[2]
    else:
        valid_index = False
    if not valid_index:
        raise ValueError("initial index must have shape (d,) or (m, d)")
    if not all(np.all(np.isfinite(array)) for array in (index, U, I)):
        raise ValueError("initial index, U, and I must contain only finite values")

    if mass is None:
        mass = np.ones(U.shape[0])
    else:
        mass_array = np.asarray(mass)
        if np.iscomplexobj(mass_array) or not np.issubdtype(
            mass_array.dtype, np.number
        ):
            raise TypeError("mass must be a real numeric array")
        mass = np.asarray(mass, dtype=float)
    if mass.shape != (U.shape[0],):
        raise ValueError("mass must have shape (J,)")
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("mass must contain only finite positive values")
    return index, U, I, mass


def _validate_settings(lambda_prox, max_steps, tol, theta, lsmr_maxiter):
    """Проверить численные допуски, лимиты итераций и trust-параметры."""
    if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)):
        raise TypeError("max_steps must be an integer")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if lsmr_maxiter is not None:
        if isinstance(lsmr_maxiter, bool) or not isinstance(
            lsmr_maxiter, (int, np.integer)
        ):
            raise TypeError("lsmr_maxiter must be an integer or None")
        if lsmr_maxiter < 1:
            raise ValueError("lsmr_maxiter must be positive")
    if not np.isfinite(lambda_prox) or lambda_prox < 0:
        raise ValueError("lambda_prox must be finite and nonnegative")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if not np.isfinite(theta) or not 0 < theta < 1:
        raise ValueError("theta must be finite and lie between zero and one")


def _normalize_index(index: np.ndarray) -> np.ndarray:
    """Нормировать single-index или ортонормировать строки multi-index."""
    if index.ndim == 1:
        norm = np.linalg.norm(index)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("initial index must have a nonzero finite norm")
        return index / norm
    if np.linalg.matrix_rank(index) < index.shape[0]:
        raise ValueError("initial multi-index matrix must have full row rank")
    basis, _ = np.linalg.qr(index.T, mode="reduced")
    return basis.T


def _default_trust_radius(d: int, m: int) -> float:
    """Выбрать консервативный радиус correction по ``d`` и рангу ``m``."""
    if m == 1:
        return 0.35 if d <= 100 else 0.20
    return 0.25 if d <= 100 else 0.15


def _local_refit(
    I: np.ndarray, U: np.ndarray, index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Найти minimum-norm локальные коэффициенты без normal equations."""
    if index.ndim == 1:
        projected = U @ index  # (J, p)
        denominator = np.einsum("jp,jp->j", projected, projected, optimize=True)
        numerator = np.einsum("jp,jp->j", I, projected, optimize=True)
        coefficients = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        return coefficients, (denominator > 0).astype(np.intp)

    J, P, _ = U.shape
    m = len(index)
    coefficients = np.empty((J, m))
    ranks = np.empty(J, dtype=np.intp)
    # NUMERICAL: minimum-norm SVD сохраняет cutoff исходного lstsq(rcond=None).
    # Пакеты убирают J вызовов Python; рабочие SVD-массивы ограничены 4 MiB
    # либо размером одного центра, если один центр превышает этот бюджет.
    block_size = max(1, _LOCAL_REFIT_BYTES // (8 * (3 * P * m + 3 * m * m)))
    rcond = np.finfo(float).eps * max(P, m)
    for start in range(0, J, block_size):
        stop = min(J, start + block_size)
        projected = U[start:stop] @ index.T
        left, singular, right = np.linalg.svd(projected, full_matrices=False)
        keep = singular > rcond * singular[:, :1]
        coordinates = (left.swapaxes(1, 2) @ I[start:stop, :, None]).squeeze(-1)
        np.divide(coordinates, singular, out=coordinates, where=keep)
        coordinates[~keep] = 0
        coefficients[start:stop] = (
            right.swapaxes(1, 2) @ coordinates[..., None]
        ).squeeze(-1)
        ranks[start:stop] = np.count_nonzero(keep, axis=1)
        # Вблизи потери ранга разные LAPACK-драйверы заметно расходятся
        # по слабым компонентам. Сохраняем исходный lstsq для таких центров.
        sensitive = singular[:, -1] <= math.sqrt(np.finfo(float).eps) * singular[:, 0]
        for local in np.flatnonzero(sensitive):
            j = start + local
            coefficients[j], _, ranks[j], _ = np.linalg.lstsq(
                projected[local], I[j], rcond=None
            )
    return coefficients, ranks


def _predict(U: np.ndarray, index: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Вычислить локальные fitted values для single- или multi-index."""
    if index.ndim == 1:
        return coefficients[:, None] * (U @ index)
    return multi_forward(U, index, coefficients)


def _loss(I, U, index, coefficients, mass) -> float:
    """Вычислить взвешенную половину квадратичной ошибки ``I - fitted``."""
    residual = I - _predict(U, index, coefficients)
    return 0.5 * float(np.einsum("j,jp,jp->", mass, residual, residual))


def _linear_operator(
    U: np.ndarray,
    coefficients: np.ndarray,
    sqrt_mass: np.ndarray,
    index_shape: tuple[int, ...],
) -> sparse_linalg.LinearOperator:
    """Построить глобальный оператор; adjoint задан явно."""
    rows = U.shape[0] * U.shape[1]
    size = math.prod(index_shape)

    def matvec(vector):
        """Применить scaled forward action глобального least-squares оператора."""
        if len(index_shape) == 1:
            data = coefficients[:, None] * (U @ vector)
            return (sqrt_mass[:, None] * data).ravel()
        matrix = vector.reshape(index_shape)
        data = multi_forward(U, matrix, coefficients)
        return (sqrt_mass[:, None] * data).ravel()

    def rmatvec(vector):
        """Применить adjoint action того же оператора."""
        data = vector.reshape(U.shape[:2])
        if len(index_shape) == 1:
            return np.einsum(
                "j,j,jpd,jp->d",
                sqrt_mass,
                coefficients,
                U,
                data,
                optimize=True,
            )
        return multi_adjoint(U, sqrt_mass[:, None] * data, coefficients).ravel()

    linear_operator: Any = sparse_linalg.LinearOperator
    return linear_operator((rows, size), matvec=matvec, rmatvec=rmatvec, dtype=float)


def _global_correction(
    I,
    U,
    index,
    coefficients,
    mass,
    lambda_prox,
    tol,
    maxiter,
):
    """Решить matrix-free ridge correction и вернуть residual certificate."""
    operator = _linear_operator(U, coefficients, np.sqrt(mass), index.shape)
    residual = np.sqrt(mass)[:, None] * (I - _predict(U, index, coefficients))
    krylov_tol = min(1e-10, tol * 0.1)
    lsmr_method: Any = sparse_linalg.lsmr
    result = lsmr_method(
        operator,
        residual.ravel(),
        damp=math.sqrt(lambda_prox),
        atol=krylov_tol,
        btol=krylov_tol,
        maxiter=maxiter or max(50, 5 * index.size),
    )
    correction = result[0]
    if not np.all(np.isfinite(correction)):
        raise RuntimeError("LSMR returned a non-finite correction")

    # Сертификат (33) пересчитывается в исходных координатах correction.
    normal_residual = operator.rmatvec(residual.ravel()) - (
        operator.rmatvec(operator @ correction) + lambda_prox * correction
    )
    normal_norm = float(np.linalg.norm(normal_residual))
    initial_normal = float(np.linalg.norm(operator.rmatvec(residual.ravel())))
    denominator = (
        lambda_prox * float(np.linalg.norm(correction))
        if lambda_prox > 0
        else initial_normal
    )
    zero_step_tolerance = 64 * np.finfo(float).eps * max(1.0, initial_normal)
    ratio = (
        normal_norm / denominator
        if denominator > 0
        else (0.0 if normal_norm <= zero_step_tolerance else math.inf)
    )
    return correction, int(result[1]), int(result[2]), ratio, normal_norm


def _gauge_fix(raw_index, coefficients, prior):
    """Ортонормировать индекс, сохранив все fitted values."""
    if raw_index.ndim == 1:
        norm = np.linalg.norm(raw_index)
        if norm == 0:
            return prior.copy(), np.zeros_like(coefficients), 0
        index = raw_index / norm
        gauge_coefficients = norm * coefficients
        if np.dot(index, prior) < 0:
            index = -index
            gauge_coefficients = -gauge_coefficients
        return index, gauge_coefficients, 1

    basis, factor = np.linalg.qr(raw_index.T, mode="reduced")
    index = basis.T
    gauge_coefficients = coefficients @ factor.T
    left, _, right_transpose = np.linalg.svd(prior @ index.T)
    rotation = left @ right_transpose
    index = rotation @ index
    gauge_coefficients = gauge_coefficients @ rotation.T
    return index, gauge_coefficients, int(np.linalg.matrix_rank(raw_index))


def _index_distance(index: np.ndarray, prior: np.ndarray) -> float:
    """Измерить sign-invariant distance для линии или projector-distance для basis."""
    if index.ndim == 1:
        return float(min(np.linalg.norm(index - prior), np.linalg.norm(index + prior)))
    # NUMERICAL: для ортонормальных строк это ||P*P-Q*Q||_F/sqrt(2).
    # Остаток проекции не вычитает близкие traces и требует (m,d), не (d,d).
    return float(np.linalg.norm(index - (index @ prior.T) @ prior, ord="fro"))


def _stationarity(I, U, index, coefficients, mass, loss, *, U_norm2=None):
    """Посчитать Riemannian/local stationarity и ортонормированность индекса."""
    residual = I - _predict(U, index, coefficients)
    transposed_residual = np.einsum("jpd,jp->jd", U, residual, optimize=True)
    if U_norm2 is None:
        U_norm2 = np.einsum("jpd,jpd->j", U, U, optimize=True)

    if index.ndim == 1:
        gradient = -np.einsum(
            "j,j,jd->d", mass, coefficients, transposed_residual, optimize=True
        )
        riemannian = gradient - np.dot(gradient, index) * index
        projected = U @ index
        local = -mass * np.einsum("jp,jp->j", projected, residual, optimize=True)
        local_scale = np.maximum(
            1.0, np.linalg.norm(projected, axis=1) * np.linalg.norm(I, axis=1)
        )
        local_score = float(np.max(np.abs(local) / local_scale))
        operator_norm2 = float(np.sum(mass * np.square(coefficients) * U_norm2))
        orthogonality = abs(float(np.dot(index, index)) - 1.0)
    else:
        gradient = -np.einsum(
            "j,jm,jd->md",
            mass,
            coefficients,
            transposed_residual,
            optimize=True,
        )
        product = gradient @ index.T
        riemannian = gradient - (0.5 * (product + product.T)) @ index
        projected = np.einsum("jpd,md->jpm", U, index, optimize=True)
        local = -mass[:, None] * np.einsum(
            "jpm,jp->jm", projected, residual, optimize=True
        )
        local_scale = np.maximum(
            1.0,
            np.linalg.norm(projected, axis=(1, 2)) * np.linalg.norm(I, axis=1),
        )
        local_score = float(np.max(np.linalg.norm(local, axis=1) / local_scale))
        coefficient_norm2 = np.einsum(
            "jm,jm->j", coefficients, coefficients, optimize=True
        )
        operator_norm2 = float(np.sum(mass * coefficient_norm2 * U_norm2))
        orthogonality = float(
            np.linalg.norm(index @ index.T - np.eye(index.shape[0]), ord="fro")
        )

    gradient_scale = max(1.0, math.sqrt(operator_norm2 * 2.0 * loss))
    return (
        float(np.linalg.norm(riemannian) / gradient_scale),
        local_score,
        orthogonality,
    )
