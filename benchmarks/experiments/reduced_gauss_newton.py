"""Изолированный H7: полный residual Jacobian reduced multi-index objective."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr

from ADP.gpu import array_module
from ADP.solver.LSMR import (
    HPAOResult,
    _index_distance,
    _normalize_index,
    _stationarity,
    _validate_inputs,
    _validate_settings,
)

from .multi_solver_derivation import _retract, evaluate_reduced
from .reduced_lbfgs import _horizontal

_operator_type: Any = LinearOperator
_lsmr_method: Any = lsmr


class _ReducedState:
    """Batched spectral residual; хранит только малые локальные SVD-факторы."""

    def __init__(self, B, U, I, mass, U_norm2):
        self.B, self.U = B, U
        self.root = np.sqrt(mass)
        self.left, self.singular, self.right = np.linalg.svd(
            U @ B.T, full_matrices=False
        )
        self.alpha = (self.left.swapaxes(1, 2) @ I[..., None]).squeeze(-1)
        self.q0 = I - (self.left @ self.alpha[..., None]).squeeze(-1)
        tau = np.finfo(float).eps * max(U.shape[1], B.shape[0])
        self.keep = self.singular > tau * self.singular[:, :1]
        self.ranks = np.count_nonzero(self.keep, axis=1)
        retained_guard = (64 * np.finfo(float).eps * np.maximum(1.0, np.sqrt(U_norm2)))[
            :, None
        ]
        if np.any(self.keep & (self.singular < retained_guard)):
            raise RuntimeError("retained singular value below absolute guard")
        if np.any(
            self.keep & (self.singular < 2 * tau * self.singular[:, :1])
        ) or np.any(~self.keep & (self.singular > 0.5 * tau * self.singular[:, :1])):
            raise RuntimeError("singular value near cutoff")
        if np.any((self.ranks == 0) & (U_norm2 != 0)):
            raise RuntimeError("rank-zero projected system has nonzero U")
        self.inverse = np.divide(
            1.0,
            self.singular,
            out=np.zeros_like(self.singular),
            where=self.keep,
        )
        coordinates = self.alpha * self.inverse
        self.coefficients = (
            self.right.swapaxes(1, 2) @ coordinates[..., None]
        ).squeeze(-1)
        residual = I - (self.left @ (self.alpha * self.keep)[..., None]).squeeze(-1)
        self.residual = self.root[:, None] * residual
        self.objective = 0.5 * float(np.sum(self.residual**2))
        self.value_defect = absolute_gradient_defect = 0.0
        # Полный retained rank даёт нулевой defect. Чувствительные центры
        # проверяет прежний независимый reference, без копии большого U.
        for j in np.flatnonzero((self.ranks > 0) & (self.ranks < B.shape[0])):
            reference = evaluate_reduced(B, U[j : j + 1], I[j : j + 1], mass[j : j + 1])
            self.value_defect += reference.value_defect
            scale = max(
                1.0,
                math.sqrt(
                    mass[j]
                    * np.sum(reference.coefficients**2)
                    * U_norm2[j]
                    * 2
                    * reference.objective
                ),
            )
            absolute_gradient_defect += reference.gradient_defect * scale
        scale = max(
            1.0,
            math.sqrt(
                float(np.sum(mass * np.sum(self.coefficients**2, axis=1) * U_norm2))
                * 2
                * self.objective
            ),
        )
        # Triangle bound консервативен при нескольких rank-deficient центрах.
        self.gradient_defect = absolute_gradient_defect / scale
        if (
            not np.isfinite(self.objective + self.value_defect + self.gradient_defect)
            or self.value_defect > 1e-10 * max(1.0, self.objective)
            or self.gradient_defect > 1e-8
        ):
            raise RuntimeError("reduced objective is outside the defect guards")
        self.operator = _operator_type(
            (I.size, B.size), matvec=self.matvec, rmatvec=self.rmatvec, dtype=float
        )
        self.gradient = self.rmatvec(self.residual.ravel()).reshape(B.shape)

    def _pairs(self):
        for i in range(self.B.shape[0]):
            for k in range(self.B.shape[0]):
                active = self.keep[:, i] & ~self.keep[:, k]
                if np.any(active):
                    gap = self.singular[:, i] ** 2 - self.singular[:, k] ** 2
                    inverse_gap = np.divide(
                        1.0, gap, out=np.zeros_like(gap), where=active
                    )
                    yield i, k, inverse_gap

    def matvec(self, value):
        direction = _horizontal(self.B, value.reshape(self.B.shape))
        rotated = (self.U @ direction.T) @ self.right.swapaxes(1, 2)
        value = (rotated @ (self.alpha * self.inverse)[..., None]).squeeze(-1)
        value -= (self.left @ (self.left.swapaxes(1, 2) @ value[..., None])).squeeze(-1)
        value += (
            self.left
            @ (np.sum(self.q0[..., None] * rotated, axis=1) * self.inverse)[..., None]
        ).squeeze(-1)
        for i, k, inverse_gap in self._pairs():
            qi, qk = self.left[:, :, i], self.left[:, :, k]
            mixing = (
                self.singular[:, i] * np.sum(qk * rotated[:, :, i], axis=1)
                + self.singular[:, k] * np.sum(qi * rotated[:, :, k], axis=1)
            ) * inverse_gap
            value += mixing[:, None] * (
                self.alpha[:, i, None] * qk + self.alpha[:, k, None] * qi
            )
        return (-self.root[:, None] * value).ravel()

    def rmatvec(self, value):
        value = -self.root[:, None] * value.reshape(self.U.shape[:2])
        coordinates = (self.left.swapaxes(1, 2) @ value[..., None]).squeeze(-1)
        complement = value - (self.left @ coordinates[..., None]).squeeze(-1)
        rotated = (
            complement[..., None] * (self.alpha * self.inverse)[:, None, :]
            + self.q0[..., None] * (coordinates * self.inverse)[:, None, :]
        )
        for i, k, inverse_gap in self._pairs():
            mixing = (
                self.alpha[:, i] * coordinates[:, k]
                + self.alpha[:, k] * coordinates[:, i]
            ) * inverse_gap
            rotated[:, :, i] += (mixing * self.singular[:, i])[:, None] * self.left[
                :, :, k
            ]
            rotated[:, :, k] += (mixing * self.singular[:, k])[:, None] * self.left[
                :, :, i
            ]
        gradient = np.einsum("jpm,jpd->md", rotated @ self.right, self.U, optimize=True)
        return _horizontal(self.B, gradient).ravel()


def _direction(state: _ReducedState, ridge: float):
    """Augmented LS и certificate именно вычисленного направления, до Armijo."""
    J = state.operator
    rows, columns = J.shape
    root = math.sqrt(ridge)
    augmented = _operator_type(
        (rows + columns, columns),
        matvec=lambda x: np.concatenate((J @ x, root * x)),
        rmatvec=lambda v: J.rmatvec(v[:rows]) + root * v[rows:],
        dtype=float,
    )
    rhs = np.concatenate((-state.residual.ravel(), np.zeros(columns)))
    g = state.gradient.ravel()
    limit, iterations = max(50, 5 * columns), 0
    initial = None
    ratio = math.inf
    for inner_tol in (1e-12, 1e-14, np.finfo(float).eps):
        result = _lsmr_method(
            augmented,
            rhs,
            atol=inner_tol,
            btol=inner_tol,
            maxiter=limit - iterations,
            x0=initial,
        )
        iterations += int(result[2])
        initial = _horizontal(state.B, result[0].reshape(state.B.shape)).ravel()
        if not np.all(np.isfinite(initial)):
            raise RuntimeError("nonfinite reduced GN direction")
        error = -g - (J.rmatvec(J @ initial) + ridge * initial)
        denominator = ridge * np.linalg.norm(initial)
        ratio = (
            float(np.linalg.norm(error) / denominator) if denominator > 0 else math.inf
        )
        if ratio <= 0.1 or iterations >= limit:
            break
    return initial.reshape(state.B.shape), ratio, iterations


def solve(
    index_init: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    *,
    mass: np.ndarray | None = None,
    lambda_prox: float = 0.05,
    max_steps: int = 160,
    tol: float = 1e-6,
) -> HPAOResult:
    """Решить прежний reduced objective; guarded APPROXIMATE CPU-вариант."""
    if array_module(U) is not np:
        raise NotImplementedError("experimental reduced GN requires CPU arrays")
    B, U, I, mass = _validate_inputs(index_init, U, I, mass)
    _validate_settings(lambda_prox, max_steps, tol, 0.1, None)
    if B.ndim != 2 or B.shape[0] > U.shape[1]:
        raise ValueError("reduced GN requires a multi-index matrix with m<=P")
    if lambda_prox <= 0:
        raise ValueError("reduced GN requires positive lambda_prox")
    B = _normalize_index(B)
    U_norm2 = np.einsum("jpd,jpd->j", U, U)
    current = _ReducedState(B, U, I, mass, U_norm2)
    pattern = current.ranks.copy()
    history = [current.objective]
    ridge = lambda_prox
    accepted = consecutive = iterations = rejections = evaluations = 0
    linear_rejections = guard_rejections = rank_changes = 0
    relative = aligned = ratio = math.inf
    score = _stationarity(
        I, U, B, current.coefficients, mass, current.objective, U_norm2=U_norm2
    )
    for _ in range(max_steps):
        if np.linalg.norm(current.gradient) == 0:
            break
        for _ in range(12):
            direction, ratio, count = _direction(current, ridge)
            iterations += count
            if ratio <= 0.1:
                break
            ridge *= 4
            linear_rejections += 1
        else:
            raise RuntimeError("reduced GN linear certificate failed")
        derivative = float(np.sum(current.gradient * direction))
        if not np.isfinite(derivative) or derivative >= 0:
            raise RuntimeError("reduced GN direction is not descending")
        step = min(1.0, 0.25 * math.sqrt(B.shape[0]) / np.linalg.norm(direction))
        last_rejection = "Armijo objective did not decrease"
        for _ in range(30):
            candidate_B = _retract(B, step * direction)
            evaluations += 1
            candidate = None
            try:
                candidate = _ReducedState(candidate_B, U, I, mass, U_norm2)
            except (RuntimeError, np.linalg.LinAlgError) as error:
                guard_rejections += 1
                last_rejection = str(error)
            if candidate is not None:
                if not np.array_equal(candidate.ranks, pattern):
                    rank_changes += 1
                    last_rejection = "local rank pattern changed"
                else:
                    last_rejection = "Armijo objective did not decrease"
            if (
                candidate is not None
                and np.array_equal(candidate.ranks, pattern)
                and candidate.objective <= current.objective + 1e-4 * step * derivative
            ):
                break
            step *= 0.5
            rejections += 1
        else:
            raise RuntimeError(
                f"reduced GN line search exhausted its budget: {last_rejection}; "
                f"guard rejects={guard_rejections}, rank changes={rank_changes}"
            )
        assert candidate is not None
        predicted = -step * derivative - 0.5 * step**2 * float(
            np.sum((current.operator @ direction.ravel()) ** 2)
        )
        reduction = current.objective - candidate.objective
        if predicted > 0:
            if reduction / predicted > 0.75:
                ridge = max(lambda_prox, ridge * 0.5)
            elif reduction / predicted < 0.25:
                ridge *= 2
        relative = abs(reduction) / max(1.0, current.objective)
        aligned = _index_distance(candidate_B, B)
        B, current = candidate_B, candidate
        accepted += 1
        history.append(current.objective)
        score = _stationarity(
            I, U, B, current.coefficients, mass, current.objective, U_norm2=U_norm2
        )
        certified = relative < tol and aligned < tol and max(score) < tol
        consecutive = consecutive + 1 if certified else 0
        if consecutive == 2:
            break
    return HPAOResult(
        B,
        current.coefficients,
        {
            "solver": "reduced-gn",
            "loss": current.objective,
            "converged": consecutive == 2,
            "accepted_steps": accepted,
            "stop_reason": "converged"
            if consecutive == 2
            else ("stationary_initial" if accepted == 0 else "max_steps"),
            "loss_history": tuple(history),
            "riemannian_gradient": score[0],
            "local_gradient": score[1],
            "orthogonality": score[2],
            "relative_loss_change": relative,
            "aligned_step": aligned,
            "consecutive_certified_steps": consecutive,
            "direction_normal_residual_ratio": ratio,
            "linear_iterations": iterations,
            "linear_rejections": linear_rejections,
            "armijo_rejections": rejections,
            "guard_rejections": guard_rejections,
            "rank_changes": rank_changes,
            "objective_evaluations": evaluations,
            "value_defect": current.value_defect,
            "gradient_defect_bound": current.gradient_defect,
        },
    )
