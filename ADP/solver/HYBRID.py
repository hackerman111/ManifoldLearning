from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import LinearOperator, cg, lsmr

from ._multi_operator import adjoint, forward

if TYPE_CHECKING:
    from .LSMR import HPAOResult

_operator_type: Any = LinearOperator
_lsmr_method: Any = lsmr
_cg_method: Any = cg

# Верхняя граница для небольшого dense SVD; crossover проверяется бенчмарком.
# Учитывается суммарная память входных матриц и временных рабочих копий.
DENSE_MAX_UNKNOWNS = 256
DENSE_MAX_BYTES = 64 * 1024**2
# 32 направления отсеивают дорогие trust-пробы в benchmarks.hybrid_solvers.
# Размер вспомогательного пространства независим от ранга индекса.
TRUST_MAX_VECTORS = 32


def use_dense(rows: int, columns: int, max_unknowns: int, max_bytes: int) -> bool:
    """Проверить, помещается ли bounded dense SVD в заданный memory budget."""
    return (
        columns <= max_unknowns
        and 8 * (4 * rows * columns + 4 * columns**2) <= max_bytes
    )


def design_matrix(
    U: np.ndarray, coefficients: np.ndarray, root: np.ndarray
) -> np.ndarray:
    """Явный малый joint design: строки (j,p), столбцы (a,d)."""
    J, P, d = U.shape
    design = np.empty((J * P, coefficients.shape[1] * d))
    for a in range(coefficients.shape[1]):
        design[:, a * d : (a + 1) * d] = (
            (root * coefficients[:, a])[:, None, None] * U
        ).reshape(J * P, d)
    return design


@dataclass(frozen=True, slots=True)
class LinearResult:
    solution: np.ndarray
    backend: str
    stop: int
    iterations: int
    relative_residual: float
    rank: int | None


def solve_augmented(
    operator: Any,
    rhs: np.ndarray,
    initial: np.ndarray,
    diagonal: np.ndarray,
    *,
    tol: float,
    maxiter: int | None,
    dense: np.ndarray | None = None,
) -> LinearResult:
    """Решить LS-коррекцию и проверить stationarity в исходных координатах."""
    residual = rhs - operator @ initial
    rank = None
    if not np.any(rhs):
        return LinearResult(np.zeros_like(initial), "zero-rhs", 0, 0, 0.0, None)
    if dense is not None:
        correction, _, rank, _ = linalg.lstsq(
            dense, residual, cond=None, lapack_driver="gelsd", check_finite=False
        )
        stop, iterations, backend = 0, 0, "dense-svd"
        if rank != operator.shape[1]:
            raise RuntimeError(
                f"hybrid augmented system is rank-deficient: rank={rank}"
            )
    else:
        if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0):
            raise RuntimeError("hybrid preconditioner has non-positive diagonal")
        scale = 1.0 / np.sqrt(diagonal)
        scaled = _operator_type(
            operator.shape,
            matvec=lambda z: operator @ (scale * z),
            rmatvec=lambda v: scale * operator.rmatvec(v),
            dtype=float,
        )
        result = _lsmr_method(
            scaled,
            residual,
            atol=min(1e-12, tol * 0.01),
            btol=min(1e-12, tol * 0.01),
            maxiter=maxiter or max(50, 5 * operator.shape[1]),
        )
        correction = scale * result[0]
        stop, iterations, backend = int(result[1]), int(result[2]), "scaled-lsmr"
        if stop not in (0, 1, 2, 4, 5):
            raise RuntimeError(
                f"hybrid LSMR did not converge: stop={stop}, iterations={iterations}"
            )
    solution = initial + correction
    normal = operator.rmatvec(rhs - operator @ solution)
    normal_rhs = operator.rmatvec(rhs)
    relative = float(np.linalg.norm(normal)) / max(
        float(np.linalg.norm(normal_rhs)), np.finfo(float).tiny
    )
    if not np.all(np.isfinite(solution)) or not np.isfinite(relative):
        raise RuntimeError("hybrid returned a non-finite solution or residual")
    if relative > max(10 * tol, 256 * np.finfo(float).eps):
        raise RuntimeError(f"hybrid residual certificate failed: {relative:.3e}")
    return LinearResult(solution, backend, stop, iterations, relative, rank)


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
    from .LSMR import solve as hpao

    return hpao(index_init, U, I, linear_solver="hybrid", **settings)


class PenaltyRoot:
    """NUMERICAL: sqrt(I-sum w_j P_j.T P_j) через малый спектральный фактор."""

    def __init__(self, projectors: np.ndarray, weights: np.ndarray) -> None:
        """Построить малый спектральный фактор ``sqrt(I-F*F)``."""
        factor = (np.sqrt(weights)[:, None, None] * projectors).reshape(
            -1, projectors.shape[-1]
        )
        _, singular, self.basis = linalg.svd(
            factor, full_matrices=False, check_finite=False
        )
        if singular[0] > 1 + 256 * np.finfo(float).eps * max(factor.shape):
            raise RuntimeError("manifold penalty is not positive semidefinite")
        squares = np.minimum(singular**2, 1.0)
        self.factors = squares / (1.0 + np.sqrt(1.0 - squares))

    def apply(self, B: np.ndarray) -> np.ndarray:
        """Применить корень manifold penalty к матрице ``B``."""
        return B - ((B @ self.basis.T) * self.factors) @ self.basis


def _manifold_pcg(
    U: np.ndarray,
    I: np.ndarray,
    gamma: np.ndarray,
    normalized: np.ndarray,
    projectors: np.ndarray,
    slopes: np.ndarray,
    initial: np.ndarray,
    ridge: float,
    tol: float,
    maxiter: int | None,
) -> LinearResult:
    """Joint PCG: блок (m,m) для каждого признака вместо scalar Jacobi.

    Normal equations выбраны по benchmark manifold150; исходный остаток
    обязателен, при отказе вызывающий код запускает augmented LSMR.
    """
    m, d = initial.shape
    weighted_slopes = gamma[:, None] * slopes
    adjoint_U = U.swapaxes(1, 2)
    factor = (np.sqrt(normalized)[:, None, None] * projectors).reshape(-1, d)
    local = np.empty((len(U), d))

    def matvec(vector: np.ndarray) -> np.ndarray:
        """Применить block-PCG normal operator без dense ``d x d`` матриц."""
        B = vector.reshape(m, d)
        np.matmul(slopes, B, out=local)
        projected = U @ local[..., None]
        np.matmul(adjoint_U, projected, out=local[..., None])
        result = weighted_slopes.T @ local
        # EXACT: C=I-F*F; два GEMM без (J,m,m) и перестановок внутри PCG.
        penalty = B - (B @ factor.T) @ factor
        return (result + ridge * penalty).ravel()

    energy = np.einsum("jpd,jpd->jd", U, U)
    blocks = np.empty((d, m, m))
    for a in range(m):
        blocks[:, :, a] = (
            (gamma[:, None] * slopes).T @ (slopes[:, a, None] * energy)
        ).T
    penalty_diag = np.maximum(
        1 - np.einsum("j,jmd,jmd->d", normalized, projectors, projectors), 0
    )
    blocks[:, np.arange(m), np.arange(m)] += ridge * penalty_diag[:, None]
    values, vectors = np.linalg.eigh(blocks)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        return LinearResult(initial.ravel(), "block-pcg-failed", -1, 0, math.inf, None)

    def precondition(vector: np.ndarray) -> np.ndarray:
        """Применить block spectral preconditioner к flattened ``B``."""
        local = vector.reshape(m, d).T[..., None]
        coordinates = (vectors.swapaxes(1, 2) @ local) / values[..., None]
        return (vectors @ coordinates).squeeze(-1).T.ravel()

    operator = _operator_type(
        (m * d, m * d), matvec=matvec, rmatvec=matvec, dtype=float
    )
    preconditioner = _operator_type(
        operator.shape, matvec=precondition, rmatvec=precondition, dtype=float
    )
    rhs = adjoint(U, gamma[:, None] * I, slopes).ravel()
    iterations = 0

    def record(_: np.ndarray) -> None:
        """Считать итерации block-PCG для linear diagnostics."""
        nonlocal iterations
        iterations += 1

    solution, info = _cg_method(
        operator,
        rhs,
        x0=initial.ravel(),
        M=preconditioner,
        rtol=tol,
        atol=0.0,
        maxiter=maxiter or max(50, min(1000, 5 * m * d)),
        callback=record,
    )
    relative = float(np.linalg.norm(matvec(solution) - rhs)) / max(
        float(np.linalg.norm(rhs)), np.finfo(float).tiny
    )
    accepted = (
        info == 0
        and np.all(np.isfinite(solution))
        and np.isfinite(relative)
        and relative <= max(10 * tol, 256 * np.finfo(float).eps)
    )
    return LinearResult(
        solution,
        "block-pcg" if accepted else "block-pcg-failed",
        int(info),
        iterations,
        relative,
        None,
    )


def solve_manifold(
    U: np.ndarray,
    I: np.ndarray,
    mass: np.ndarray,
    weights: np.ndarray,
    projectors: np.ndarray,
    slopes: np.ndarray,
    initial: np.ndarray,
    *,
    ridge: float,
    tol: float,
    maxiter: int | None = None,
    dense_max_unknowns: int = DENSE_MAX_UNKNOWNS,
    dense_max_bytes: int = DENSE_MAX_BYTES,
) -> LinearResult:
    """Исходная manifold B-задача: bounded SVD либо block-PCG/augmented LSMR."""
    from .LSMR import _validate_inputs

    initial, U, I, mass = _validate_inputs(initial, U, I, mass)
    if initial.ndim != 2:
        raise ValueError("manifold initial must have shape (m,d)")
    for name, value, shape in (
        ("weights", weights, (len(U),)),
        ("slopes", slopes, (len(U), initial.shape[0])),
        ("projectors", projectors, (len(U), *initial.shape)),
    ):
        if np.iscomplexobj(value) or not np.issubdtype(
            np.asarray(value).dtype, np.number
        ):
            raise TypeError(f"{name} must be real numeric")
        if np.shape(value) != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite with shape {shape}")
    weights = np.asarray(weights, dtype=float)
    slopes = np.asarray(slopes, dtype=float)
    projectors = np.asarray(projectors, dtype=float)
    if np.any(weights <= 0):
        raise ValueError("weights must be positive")
    if not np.allclose(
        projectors @ projectors.swapaxes(1, 2),
        np.eye(initial.shape[0]),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("source projectors must have orthonormal rows")
    if not np.isfinite(ridge) or ridge < 0 or not np.isfinite(tol) or tol <= 0:
        raise ValueError(
            "ridge must be nonnegative and tol must be positive and finite"
        )
    for name, value in (
        ("maxiter", maxiter),
        ("dense_max_unknowns", dense_max_unknowns),
        ("dense_max_bytes", dense_max_bytes),
    ):
        if value is None and name == "maxiter":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if value < (1 if name == "maxiter" else 0):
            raise ValueError(f"{name} is out of range")
    root = np.sqrt(mass * weights)
    normalized = weights / weights.sum()
    rows, columns = I.size, initial.size
    dense_enabled = use_dense(
        rows + columns, columns, dense_max_unknowns, dense_max_bytes
    )
    attempted_iterations = 0
    if not dense_enabled:
        attempt = _manifold_pcg(
            U,
            I,
            mass * weights,
            normalized,
            projectors,
            slopes,
            initial,
            ridge,
            tol,
            maxiter,
        )
        if attempt.backend == "block-pcg":
            return attempt
        attempted_iterations = attempt.iterations
    penalty = PenaltyRoot(projectors, normalized)
    root_ridge = math.sqrt(ridge)

    def matvec(vector: np.ndarray) -> np.ndarray:
        """Применить augmented manifold forward operator."""
        B = vector.reshape(initial.shape)
        data = root[:, None] * forward(U, B, slopes)
        return np.concatenate((data.ravel(), root_ridge * penalty.apply(B).ravel()))

    def rmatvec(vector: np.ndarray) -> np.ndarray:
        """Применить adjoint augmented manifold operator."""
        data = root[:, None] * vector[:rows].reshape(I.shape)
        result = adjoint(U, data, slopes)
        result += root_ridge * penalty.apply(vector[rows:].reshape(initial.shape))
        return result.ravel()

    operator = _operator_type(
        (rows + columns, columns), matvec=matvec, rmatvec=rmatvec, dtype=float
    )
    data_diagonal = (mass[:, None] * weights[:, None] * slopes**2).T @ np.einsum(
        "jpd,jpd->jd", U, U
    )
    penalty_diagonal = 1 - np.einsum("j,jmd,jmd->d", normalized, projectors, projectors)
    diagonal = (
        data_diagonal + ridge * np.maximum(penalty_diagonal, 0)[None, :]
    ).ravel()
    dense = None
    if dense_enabled:
        dense = np.zeros((rows + columns, columns))
        dense[:rows] = design_matrix(U, slopes, root)
        # Только bounded dense ветка; в большой задаче этот (d,d) фактор отсутствует.
        d = initial.shape[1]
        block = root_ridge * penalty.apply(np.eye(d))
        for a in range(initial.shape[0]):
            dense[rows + a * d : rows + (a + 1) * d, a * d : (a + 1) * d] = block
    rhs = np.concatenate(((root[:, None] * I).ravel(), np.zeros(columns)))
    linear = solve_augmented(
        operator,
        rhs,
        initial.ravel(),
        diagonal,
        tol=tol,
        maxiter=maxiter,
        dense=dense,
    )
    # Сертификат проверяется повторно через исходный penalty, без sqrt-factor.
    B = linear.solution.reshape(initial.shape)
    projected = (B @ projectors.swapaxes(1, 2)) * normalized[:, None, None]
    penalty_B = B - projected.transpose(1, 0, 2).reshape(
        len(B), -1
    ) @ projectors.reshape(-1, U.shape[2])
    gradient = (
        adjoint(U, (mass * weights)[:, None] * (forward(U, B, slopes) - I), slopes)
        + ridge * penalty_B
    )
    normal_rhs = adjoint(U, (mass * weights)[:, None] * I, slopes)
    relative = float(np.linalg.norm(gradient)) / max(
        float(np.linalg.norm(normal_rhs)), np.finfo(float).tiny
    )
    if not np.isfinite(relative) or relative > max(10 * tol, 256 * np.finfo(float).eps):
        raise RuntimeError(f"hybrid original manifold residual failed: {relative:.3e}")
    return LinearResult(
        linear.solution,
        linear.backend if dense_enabled else "block-pcg->" + linear.backend,
        linear.stop,
        linear.iterations + attempted_iterations,
        relative,
        linear.rank,
    )
