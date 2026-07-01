from __future__ import annotations

import numpy as np

from ..common.types import LocalStatistics, VariantName
from ..common.utils import safe_solve
from ..engine.base import ADPBase


class RandomProjectionADP(ADPBase):
    """ADP из manifold_new.tex: локальные суммы по случайным направлениям."""

    variant: VariantName = "new"

    def _compute_statistics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        centers: np.ndarray,
        h: float,
        beta: np.ndarray,
        directions: np.ndarray | None,
        anisotropy: float | None,
        b_value: float | None,
    ) -> LocalStatistics:
        """Вычисляет new-статистики Ima, S, U.

        Вход:
            X: матрица наблюдений n x d.
            y: вектор ответов длины n.
            centers: матрица центров J x d.
            h: текущая bandwidth.
            beta: текущее направление EDR.
            directions: направления J x P x d.
            anisotropy: значение rho или None.
            b_value: не используется в new-варианте.
        Выход:
            LocalStatistics для RandomProjectionADP.
        """

        if directions is None:
            raise ValueError("new-вариант требует directions")
        J, P, d = directions.shape
        imav = np.zeros((J, P))
        s_all = np.zeros((J, P))
        u_all = np.zeros((J, P, d))
        weight_means: list[float] = []
        rho = 1.0 if anisotropy is None else float(anisotropy)

        for start in range(0, J, self.config.chunk_size):
            stop = min(start + self.config.chunk_size, J)
            diff = X[None, :, :] - centers[start:stop, None, :]
            norm2 = np.einsum("cnd,cnd->cn", diff, diff)
            if anisotropy is None:
                q = norm2 / (h * h)
            else:
                proj2 = np.square(np.einsum("cnd,d->cn", diff, beta))
                q = (rho * rho * norm2 + proj2) / (h * h)
            chunk_imav, chunk_s, chunk_u, weight_mean = self.backend.random_projection_sums(
                diff,
                y,
                directions[start:stop],
                q,
                self.config.kernel,
            )
            imav[start:stop] = chunk_imav
            s_all[start:stop] = chunk_s
            u_all[start:stop] = chunk_u
            weight_means.append(weight_mean)

        return LocalStatistics(
            variant="new",
            imav=imav,
            centers=centers,
            h=h,
            weights_mean=float(np.mean(weight_means)),
            directions=directions,
            S=s_all,
            U=u_all,
            anisotropy=anisotropy,
        )

    def _solve_local_coefficients(self, stats: LocalStatistics, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Решает локальные коэффициенты new-варианта.

        Вход:
            stats: статистики с матрицами S и U.
            beta: текущее направление EDR.
        Выход:
            Кортеж intercepts и slopes.
        """

        if stats.S is None or stats.U is None:
            raise ValueError("Некорректные статистики new-варианта")
        J = stats.S.shape[0]
        intercepts = np.zeros(J)
        slopes = np.zeros(J)
        for j in range(J):
            col0 = stats.S[j]
            col1 = stats.U[j] @ beta
            design = np.column_stack([col0, col1])
            lhs = design.T @ design + self.config.ridge * np.eye(2)
            rhs = design.T @ stats.imav[j]
            intercepts[j], slopes[j] = safe_solve(lhs, rhs)
        return intercepts, slopes

    def _solve_beta(
        self,
        stats: LocalStatistics,
        intercepts: np.ndarray,
        slopes: np.ndarray,
        prior: np.ndarray,
        lambda_penalty: float,
    ) -> np.ndarray:
        """Решает beta для new-варианта.

        Вход:
            stats: статистики с матрицами S и U.
            intercepts: локальные свободные члены.
            slopes: локальные наклоны.
            prior: beta предыдущего outer-шага.
            lambda_penalty: сила регуляризации.
        Выход:
            Новый ненормированный вектор beta.
        """

        if stats.S is None or stats.U is None:
            raise ValueError("Некорректные статистики new-варианта")
        d = stats.U.shape[2]
        lhs = lambda_penalty * np.eye(d)
        rhs = lambda_penalty * prior
        for j, slope in enumerate(slopes):
            Uj = stats.U[j]
            residual = stats.imav[j] - intercepts[j] * stats.S[j]
            lhs += slope * slope * (Uj.T @ Uj)
            rhs += slope * (Uj.T @ residual)
        return safe_solve(lhs + self.config.ridge * np.eye(d), rhs)

    def _objective(
        self,
        stats: LocalStatistics,
        beta: np.ndarray,
        intercepts: np.ndarray,
        slopes: np.ndarray,
        prior: np.ndarray,
        lambda_penalty: float,
    ) -> float:
        """Считает objective для new-варианта.

        Вход:
            stats: статистики с матрицами S и U.
            beta: текущее направление EDR.
            intercepts: локальные свободные члены.
            slopes: локальные наклоны.
            prior: beta предыдущего outer-шага.
            lambda_penalty: сила регуляризации.
        Выход:
            Значение objective.
        """

        if stats.S is None or stats.U is None:
            raise ValueError("Некорректные статистики new-варианта")
        total = 0.0
        for j, slope in enumerate(slopes):
            pred = intercepts[j] * stats.S[j] + slope * (stats.U[j] @ beta)
            total += float(np.sum((stats.imav[j] - pred) ** 2))
        total += lambda_penalty * float(np.sum((beta - prior) ** 2))
        return total
