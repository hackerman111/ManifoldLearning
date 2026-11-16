from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import lsmr

from .._multi_operator import forward
from .HYBRID import (
    DENSE_MAX_BYTES,
    DENSE_MAX_UNKNOWNS,
    _operator_type,
    design_matrix,
    use_dense,
)

if TYPE_CHECKING:
    from ..LSMR import HPAOResult

_lsmr_method: Any = lsmr
TRUST_MAX_VECTORS = 32


class RidgeWorkspace:
    """Фиксированные U,L,I,B0; подпространство общее для всех проб lambda."""

    def __init__(
        self,
        U: np.ndarray,
        I: np.ndarray,
        index: np.ndarray,
        coefficients: np.ndarray,
        mass: np.ndarray,
        *,
        max_unknowns: int = DENSE_MAX_UNKNOWNS,
        max_bytes: int = DENSE_MAX_BYTES,
    ) -> None:
        """Подготовить фиксированные статистики для повторных ridge-проб."""
        self.U, self.coefficients, self.index = U, coefficients, index
        self.root = np.sqrt(mass)
        self._weighted_coefficients = self.root[:, None] * coefficients
        # EXACT: один (J,d) буфер на фиксированные статистики, без копий U.
        self._local = np.empty((U.shape[0], U.shape[2]))
        self.residual = (
            self.root[:, None] * (I - forward(U, index, coefficients))
        ).ravel()
        self.singular: np.ndarray | None = None
        self.calls = 0
        self.iterations = 0
        self.screened_trials = 0
        self.screening_matvecs = 0
        self.recycled_solves = 0
        self.projected_solves = 0
        self.refinements = 0
        self.forward_calls = 0
        self.adjoint_calls = 0
        self._trust_basis: np.ndarray | None = None
        self._trust_projection: np.ndarray | None = None
        self._initial_correction: np.ndarray | None = None
        rows, columns = I.size, index.size
        # Консервативный бюджет: basis, рабочие векторы и малая SVD.
        self._trust_vectors = min(
            TRUST_MAX_VECTORS, columns, max_bytes // (48 * columns)
        )
        self.backend = "scaled-lsmr"
        if use_dense(rows, columns, max_unknowns, max_bytes):
            design = design_matrix(U, coefficients, self.root)
            left, self.singular, self.right = linalg.svd(
                design, full_matrices=False, check_finite=False
            )
            self.projected_rhs = left.T @ self.residual
            self.backend = "cached-svd"
        else:
            self.diagonal = (
                (mass[:, None] * coefficients**2).T @ np.einsum("jpd,jpd->jd", U, U)
            ).ravel()
        self.normal_rhs = self.rmatvec(self.residual)
        self.initial_normal = float(np.linalg.norm(self.normal_rhs))
        if not np.all(np.isfinite(self.residual)) or not np.isfinite(
            self.initial_normal
        ):
            raise RuntimeError("hybrid linear system contains non-finite statistics")

    def matvec(self, value: np.ndarray) -> np.ndarray:
        """Применить weighted joint design ``A`` к flattened multi-index."""
        self.forward_calls += 1
        np.matmul(
            self._weighted_coefficients,
            value.reshape(self.index.shape),
            out=self._local,
        )
        return (self.U @ self._local[..., None]).ravel()

    def rmatvec(self, value: np.ndarray) -> np.ndarray:
        """Применить математически сопряжённый оператор ``A.T``."""
        self.adjoint_calls += 1
        np.matmul(
            self.U.swapaxes(1, 2),
            value.reshape(*self.U.shape[:2], 1),
            out=self._local[..., None],
        )
        return (self._weighted_coefficients.T @ self._local).ravel()

    def rejects_trust(self, ridge: float, radius: float, maxiter: int | None) -> bool:
        """NUMERICAL: нижняя граница нормы неизвестного ridge-решения.

        При H=A* A+lambda I и H delta=g для любого z верно
        ||delta|| >= |<Az,r>| / ||Hz||. Малое пространство Крылова служит
        только поиску z; итоговая граница проверяется исходными A и A*.
        Потеря ранга пространства ослабляет отсев, но не меняет LS-задачу.
        """
        count = min(self._trust_vectors, maxiter or self._trust_vectors)
        if self.singular is not None or count == 0 or self.initial_normal == 0:
            return False
        if self._trust_basis is None:
            basis = np.empty((self.index.size, count), order="F")
            projection = np.zeros((count + 1, count))
            vector = self.normal_rhs / self.initial_normal
            for k in range(count):
                basis[:, k] = vector
                normal = self.rmatvec(self.matvec(vector))
                self.screening_matvecs += 1
                # Два прохода устраняют накопление потери ортогональности.
                active = basis[:, : k + 1]
                coordinates = active.T @ normal
                vector = normal - active @ coordinates
                extra = active.T @ vector
                vector -= active @ extra
                projection[: k + 1, k] = coordinates + extra
                norm = float(np.linalg.norm(vector))
                projection[k + 1, k] = norm
                cutoff = 64 * np.finfo(float).eps * np.linalg.norm(normal)
                if norm <= cutoff:
                    basis = basis[:, : k + 1]
                    projection = projection[: k + 2, : k + 1]
                    break
                vector /= norm
            self._trust_basis, self._trust_projection = basis, projection

        basis, projection = self._trust_basis, self._trust_projection
        assert projection is not None
        # NUMERICAL: A* A V = [V,v_next] T. SVD имеет размер (q+1,q),
        # полная SVD (md,q) больше не нужна. Округление проекции может
        # ослабить поиск, но граница ниже проверяется исходными A/A*.
        shifted = projection + ridge * np.eye(*projection.shape)
        left, singular, right = linalg.svd(
            shifted, full_matrices=False, check_finite=False
        )
        if singular[0] == 0:
            return False
        relative = singular / singular[0]
        projected_rhs = basis.T @ self.normal_rhs
        keep = relative > np.finfo(float).eps * max(basis.shape)
        factors = np.divide(
            right @ projected_rhs,
            relative**2,
            out=np.zeros_like(singular),
            where=keep,
        )
        vector = basis @ (right.T @ factors)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm == 0:
            return False
        vector /= norm
        projected = self.matvec(vector)
        normal = self.rmatvec(projected) + ridge * vector
        self.screening_matvecs += 1
        # Консервативный запас на округление; при плохой обусловленности
        # запас оставляет сомнительные пробы полному LSMR.
        energy = float(np.sum(self.diagonal))
        rounding = 64 * np.finfo(float).eps * max(*self.U.shape, len(self.index))
        numerator = abs(float(projected @ self.residual))
        numerator -= rounding * math.sqrt(energy) * np.linalg.norm(self.residual)
        denominator = np.linalg.norm(normal) + rounding * (energy + ridge)
        lower = max(0.0, numerator) / denominator if denominator > 0 else 0.0
        rejected = np.isfinite(lower) and lower > radius * (
            1 + math.sqrt(np.finfo(float).eps)
        )
        self.screened_trials += int(rejected)
        if not rejected and ridge > 0:
            # NUMERICAL: min ||H V y-g|| даёт начальное приближение.
            # Точность проверяет augmented LSMR; центр исходного ridge-штрафа
            # остаётся нулевым при любом начальном приближении.
            factors = np.divide(
                left[:-1].T @ projected_rhs,
                singular,
                out=np.zeros_like(singular),
                where=keep,
            )
            self._initial_correction = basis @ (right.T @ factors)
        return bool(rejected)

    def correction(
        self,
        ridge: float,
        tol: float,
        maxiter: int | None,
        *,
        relative_tol: float | None = None,
    ) -> tuple[np.ndarray, int, int, float, float]:
        """Решить одну ridge correction-пробу и вернуть certificate residual."""
        self.calls += 1
        if self.singular is not None:
            s = self.singular
            cutoff = (
                np.finfo(float).eps
                * max(self.U.shape[0] * self.U.shape[1], self.index.size)
                * s[0]
            )
            if ridge == 0:
                factors = np.divide(1.0, s, out=np.zeros_like(s), where=s > cutoff)
            else:
                factors = s / (s**2 + ridge)
            correction = self.right.T @ (factors * self.projected_rhs)
            stop, iterations = 0, 0
        else:
            rows, columns = self.residual.size, self.index.size
            root_ridge = math.sqrt(ridge)
            # При lambda=0 сохраняем minimum-norm correction без right scaling.
            scale = (
                1.0 / np.sqrt(self.diagonal + ridge) if ridge > 0 else np.ones(columns)
            )

            def matvec(z: np.ndarray) -> np.ndarray:
                """Применить scaled augmented design в cached ridge solve."""
                value = scale * z
                return np.concatenate((self.matvec(value), root_ridge * value))

            def rmatvec(v: np.ndarray) -> np.ndarray:
                """Применить adjoint scaled augmented design."""
                return scale * (self.rmatvec(v[:rows]) + root_ridge * v[rows:])

            operator = _operator_type(
                (rows + columns, columns), matvec=matvec, rmatvec=rmatvec, dtype=float
            )
            initial = self._initial_correction if ridge > 0 else None
            self.recycled_solves += int(initial is not None)
            krylov_tol = min(1e-10, tol * 0.1)
            requested_tol = krylov_tol
            if relative_tol is not None and ridge > 0:
                # APPROXIMATE, opt-in: ||H delta-g||/(lambda ||delta||)
                # ограничивает относительную ошибку коррекции. Малый residual
                # проекции недостаточен: проверяем исходные A/A*.
                lower = self.initial_normal / (float(np.sum(self.diagonal)) + ridge)
                if initial is not None:
                    ratio, normal_norm = self._certificate(initial, ridge)
                    if ratio <= relative_tol:
                        self.projected_solves += 1
                        return initial.copy(), 0, 0, ratio, normal_norm
                    lower = max(lower, np.linalg.norm(initial) - normal_norm / ridge)
                # Только оценка начального atol; после LSMR обязателен
                # исходный сертификат и при необходимости строгий refinement.
                # ||[A;sqrt(lambda)I] S||_F = sqrt(md) для Jacobi S.
                denominator = (
                    math.sqrt(float(np.max(self.diagonal)) + ridge)
                    * math.sqrt(columns)
                    * np.linalg.norm(self.residual)
                )
                if denominator > 0:
                    requested_tol = max(
                        krylov_tol,
                        min(1e-3, 0.1 * relative_tol * ridge * lower / denominator),
                    )
            limit = maxiter or max(50, 5 * columns)
            rhs = np.concatenate((self.residual, np.zeros(columns)))
            result = _lsmr_method(
                operator,
                rhs,
                atol=requested_tol,
                btol=requested_tol,
                maxiter=limit,
                x0=initial / scale if initial is not None else None,
            )
            correction = scale * result[0]
            stop, iterations = int(result[1]), int(result[2])
            if (
                relative_tol is not None
                and requested_tol > krylov_tol
                and iterations < limit
            ):
                ratio, _ = self._certificate(correction, ridge)
                if ratio > relative_tol:
                    self.refinements += 1
                    result = _lsmr_method(
                        operator,
                        rhs,
                        atol=krylov_tol,
                        btol=krylov_tol,
                        maxiter=limit - iterations,
                        x0=result[0],
                    )
                    correction = scale * result[0]
                    stop = int(result[1])
                    iterations += int(result[2])
        self.iterations += iterations
        ratio, normal_norm = self._certificate(correction, ridge)
        if stop not in (0, 1, 2, 4, 5):
            ratio = math.inf
        elif self.singular is None and ridge > 0:
            self._initial_correction = correction
        return correction, stop, iterations, ratio, normal_norm

    def _certificate(self, correction: np.ndarray, ridge: float) -> tuple[float, float]:
        """Проверить normal-residual коррекции в исходных координатах."""
        """Остаток в исходных координатах; при lambda>0 H >= lambda I."""
        normal = (
            self.rmatvec(self.residual - self.matvec(correction)) - ridge * correction
        )
        normal_norm = float(np.linalg.norm(normal))
        initial_normal = self.initial_normal
        denominator = (
            ridge * float(np.linalg.norm(correction)) if ridge > 0 else initial_normal
        )
        zero_tolerance = 64 * np.finfo(float).eps * max(1.0, initial_normal)
        ratio = (
            normal_norm / denominator
            if denominator > 0
            else (0.0 if normal_norm <= zero_tolerance else math.inf)
        )
        if not np.all(np.isfinite(correction)) or not np.isfinite(normal_norm):
            raise RuntimeError("hybrid returned a non-finite correction")
        return ratio, normal_norm


def solve(
    index_init: np.ndarray, U: np.ndarray, I: np.ndarray, **settings: Any
) -> HPAOResult:
    """HPAO: cached-SVD либо LSMR, использующий накопленное подпространство.

    ``hybrid_inner_rtol=None`` сохраняет строгие допуски LSMR. Положительное
    значение включает APPROXIMATE-режим: исходный нормальный остаток делится
    на ``lambda * ||delta||`` и должен быть не больше заданного порога.
    """
    from ..LSMR import solve as hpao

    return hpao(index_init, U, I, linear_solver="hybrid", **settings)
