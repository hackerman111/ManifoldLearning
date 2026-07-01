from __future__ import annotations

import numpy as np

from ..common.types import LocalStatistics, VariantName
from ..common.utils import safe_solve
from ..engine.base import ADPBase


class FullMomentADP(ADPBase):
    """ADP из manifold_old.tex: полная матрица локальных моментов."""

    variant: VariantName = "old"

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
        """Вычисляет old-статистики Ima, N, S, VP.

        Вход:
            X: матрица наблюдений n x d.
            y: вектор ответов длины n.
            centers: матрица центров J x d.
            h: текущая bandwidth.
            beta: текущее направление EDR.
            directions: не используется в old-варианте.
            anisotropy: не используется в old-варианте.
            b_value: продольная bandwidth b или None.
        Выход:
            LocalStatistics для FullMomentADP.
        """

        J, d = centers.shape
        imav = np.zeros((J, d + 1))
        n_all = np.zeros(J)
        s_all = np.zeros((J, d))
        vp_all = np.zeros((J, d, d))
        weight_means: list[float] = []

        for start in range(0, J, self.config.chunk_size):
            stop = min(start + self.config.chunk_size, J)
            diff = X[None, :, :] - centers[start:stop, None, :]
            norm2 = np.einsum("cnd,cnd->cn", diff, diff)
            if b_value is None:
                q = norm2 / (h * h)
            else:
                proj2 = np.square(np.einsum("cnd,d->cn", diff, beta))
                q = norm2 / (h * h) + proj2 / (b_value * b_value)
            chunk_imav, chunk_n, chunk_s, chunk_vp, weight_mean = self.backend.full_moment_sums(
                diff,
                y,
                q,
                self.config.kernel,
            )
            imav[start:stop] = chunk_imav
            n_all[start:stop] = chunk_n
            s_all[start:stop] = chunk_s
            vp_all[start:stop] = chunk_vp
            weight_means.append(weight_mean)

        return LocalStatistics(
            variant="old",
            imav=imav,
            centers=centers,
            h=h,
            weights_mean=float(np.mean(weight_means)),
            N=n_all,
            S=s_all,
            VP=vp_all,
            b=b_value,
        )

    def _solve_local_coefficients(self, stats: LocalStatistics, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Решает локальные коэффициенты old-варианта.

        Вход:
            stats: статистики с N, S и VP.
            beta: текущее направление EDR.
        Выход:
            Кортеж intercepts и slopes.
        """

        if stats.N is None or stats.S is None or stats.VP is None:
            raise ValueError("Некорректные статистики old-варианта")
        J = stats.N.shape[0]
        intercepts = np.zeros(J)
        slopes = np.zeros(J)
        for j in range(J):
            col0 = np.concatenate([[stats.N[j]], stats.S[j]])
            col1 = np.concatenate([[stats.S[j] @ beta], stats.VP[j] @ beta])
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
        """Решает beta для old-варианта.

        Вход:
            stats: статистики с N, S и VP.
            intercepts: локальные свободные члены.
            slopes: локальные наклоны.
            prior: beta предыдущего outer-шага.
            lambda_penalty: сила регуляризации.
        Выход:
            Новый ненормированный вектор beta.
        """

        if stats.N is None or stats.S is None or stats.VP is None:
            raise ValueError("Некорректные статистики old-варианта")
        d = stats.S.shape[1]
        lhs = lambda_penalty * np.eye(d)
        rhs = lambda_penalty * prior
        for j, slope in enumerate(slopes):
            sj = stats.S[j]
            vpj = stats.VP[j]
            im0 = stats.imav[j, 0]
            im1 = stats.imav[j, 1:]
            lhs += slope * slope * (np.outer(sj, sj) + vpj.T @ vpj)
            rhs += slope * sj * (im0 - intercepts[j] * stats.N[j])
            rhs += slope * (vpj.T @ (im1 - intercepts[j] * sj))
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
        """Считает objective для old-варианта.

        Вход:
            stats: статистики с N, S и VP.
            beta: текущее направление EDR.
            intercepts: локальные свободные члены.
            slopes: локальные наклоны.
            prior: beta предыдущего outer-шага.
            lambda_penalty: сила регуляризации.
        Выход:
            Значение objective.
        """

        if stats.N is None or stats.S is None or stats.VP is None:
            raise ValueError("Некорректные статистики old-варианта")
        total = 0.0
        for j, slope in enumerate(slopes):
            pred0 = intercepts[j] * stats.N[j] + slope * (stats.S[j] @ beta)
            pred1 = intercepts[j] * stats.S[j] + slope * (stats.VP[j] @ beta)
            residual = stats.imav[j] - np.concatenate([[pred0], pred1])
            total += float(np.sum(residual**2))
        total += lambda_penalty * float(np.sum((beta - prior) ** 2))
        return total
