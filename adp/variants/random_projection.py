from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from ..common.types import LocalStatistics, VariantName
from ..common.utils import pairwise_norm2, pairwise_projection2
from ..engine.base import ADPBase


class RandomProjectionADP(ADPBase):
    """ADP из manifold_new.tex: локальные суммы по случайным направлениям."""

    variant: VariantName = "new"

    def _compute_statistics(
        self,
        X: np.ndarray,  # Матрица наблюдений n x d.
        y: np.ndarray,  # Вектор ответов длины n.
        centers: np.ndarray,  # Матрица центров J x d.
        h: float,  # Текущий масштаб h.
        beta: np.ndarray,  # Текущее направление beta.
        directions: np.ndarray | None,  # Направления J x P x d.
        anisotropy: float | None,  # rho или None на первом шаге.
        b_value: float | None,  # Не используется в new.
    ) -> LocalStatistics:
        """Вычисляет new-статистики Ima, S, U.

        Вход:
            X: матрица наблюдений.
            y: вектор ответов.
            centers: локальные центры.
            h: текущий bandwidth.
            beta: текущее EDR-направление.
            directions: случайные направления для каждого центра.
            anisotropy: rho для адаптивных весов или None.
            b_value: не используется.
        Выход:
            LocalStatistics с imav, S, U и directions.
        """

        if directions is None:
            raise ValueError("new-вариант требует directions")
        J, P, d = directions.shape
        imav = np.zeros((J, P))
        s_all = np.zeros((J, P))
        u_all = np.zeros((J, P, d))
        n_all = np.zeros(J)
        weight_means: list[float] = []

        for start in range(0, J, self.config.chunk_size):
            stop = min(start + self.config.chunk_size, J)
            center_chunk = centers[start:stop]
            norm2 = pairwise_norm2(X, center_chunk)
            if anisotropy is None:
                # Первый внешний шаг из manifold_new.tex: изотропные веса.
                q = norm2 / (h * h)
            else:
                # Адаптивные веса new: q = (rho^2 ||dx||^2 + <dx,beta>^2) / h^2.
                proj2 = pairwise_projection2(X, center_chunk, beta)
                q = (float(anisotropy) * float(anisotropy) * norm2 + proj2) / (h * h)

            weights = self.backend.kernel(q, self.config.kernel)
            n_chunk = np.asarray(weights.sum(axis=1), dtype=float)
            n_all[start:stop] = n_chunk
            weight_means.append(float(n_chunk.mean()))

            for local_index, j in enumerate(range(start, stop)):
                w = np.asarray(weights[local_index], dtype=float)
                safe_count = max(float(n_chunk[local_index]), np.finfo(float).eps)
                # Важный пункт из efficient reference: Xbar_j нормируется на N_j.
                xbar = (w @ X) / safe_count
                centered = X - xbar
                projected = centered @ directions[j].T
                weighted_projected = projected * w[:, None]
                s_all[j] = weighted_projected.sum(axis=0)
                imav[j] = (w * y) @ projected
                u_all[j] = weighted_projected.T @ centered

        return LocalStatistics(
            variant="new",
            imav=imav,
            centers=centers,
            h=h,
            weights_mean=float(np.mean(weight_means)),
            directions=directions,
            S=s_all,
            U=u_all,
            N=n_all,
            anisotropy=anisotropy,
        )

    def _solve_local_coefficients(
        self,
        stats: LocalStatistics,  # Статистики с S и U.
        beta: np.ndarray,  # Текущее beta.
    ) -> tuple[np.ndarray, np.ndarray]:
        """Решает локальные коэффициенты new-варианта.

        Вход:
            stats: статистики с S_j и U_j.
            beta: текущее направление.
        Выход:
            Кортеж intercepts и slopes.
        """

        if stats.U is None:
            raise ValueError("Некорректные статистики new-варианта")
        intercepts = np.zeros(stats.U.shape[0])
        ubeta = np.einsum("jpd,d->jp", stats.U, beta)
        numerator = np.einsum("jp,jp->j", stats.imav, ubeta)
        denominator = np.einsum("jp,jp->j", ubeta, ubeta)
        denominator = np.maximum(denominator, np.finfo(float).tiny)
        slopes = numerator / denominator
        return intercepts, slopes

    def _solve_beta(
        self,
        stats: LocalStatistics,  # Статистики с S и U.
        intercepts: np.ndarray,  # Локальные свободные члены.
        slopes: np.ndarray,  # Локальные наклоны.
        prior: np.ndarray,  # beta предыдущего внешнего шага.
        lambda_penalty: float,  # Сила регуляризации.
    ) -> np.ndarray:
        """Решает beta для new-варианта.

        Вход:
            stats: статистики с U_j.
            intercepts: локальные свободные члены.
            slopes: локальные наклоны.
            prior: направление регуляризации.
            lambda_penalty: сила регуляризации.
        Выход:
            Новый ненормированный beta.
        """

        if stats.S is None or stats.U is None:
            raise ValueError("Некорректные статистики new-варианта")
        d = stats.U.shape[2]
        residual = stats.imav - intercepts[:, None] * stats.S
        regularization = float(lambda_penalty) + float(self.config.ridge)

        def matvec(vector: np.ndarray) -> np.ndarray:
            projected = np.einsum("jpd,d->jp", stats.U, vector)
            result = np.einsum("j,jp,jpd->d", slopes**2, projected, stats.U)
            result += regularization * vector
            return result

        rhs = np.einsum("j,jp,jpd->d", slopes, residual, stats.U)
        rhs += float(lambda_penalty) * prior
        operator = LinearOperator((d, d), matvec=matvec, dtype=float)
        beta, info = cg(
            operator,
            rhs,
            x0=prior,
            rtol=self.config.tol,
            atol=0.0,
            maxiter=max(50, min(500, 5 * d)),
        )
        if info < 0 or not np.all(np.isfinite(beta)) or np.linalg.norm(beta) == 0:
            return prior
        return beta

    def _objective(
        self,
        stats: LocalStatistics,  # Статистики с S и U.
        beta: np.ndarray,  # Текущее beta.
        intercepts: np.ndarray,  # Локальные свободные члены.
        slopes: np.ndarray,  # Локальные наклоны.
        prior: np.ndarray,  # Направление регуляризации.
        lambda_penalty: float,  # Сила регуляризации.
    ) -> float:
        """Считает objective для new-варианта.

        Вход:
            stats: статистики с imav, S, U.
            beta: текущее направление.
            intercepts: локальные свободные члены.
            slopes: локальные наклоны.
            prior: направление регуляризации.
            lambda_penalty: сила регуляризации.
        Выход:
            Значение objective.
        """

        if stats.S is None or stats.U is None:
            raise ValueError("Некорректные статистики new-варианта")
        ubeta = np.einsum("jpd,d->jp", stats.U, beta)
        pred = intercepts[:, None] * stats.S + slopes[:, None] * ubeta
        total = float(np.sum((stats.imav - pred) ** 2))
        total += lambda_penalty * float(np.sum((beta - prior) ** 2))
        return total
