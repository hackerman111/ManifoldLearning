from __future__ import annotations

import numpy as np

from ..common.types import LocalStatistics, VariantName
from ..common.utils import safe_solve
from ..engine.base import ADPBase


class FullMomentADP(ADPBase):
    """ADP из manifold_old.tex: полная матрица локальных моментов без random projections."""

    variant: VariantName = "old"

    def _compute_statistics(
        self,
        X: np.ndarray,  # Матрица наблюдений n x d.
        y: np.ndarray,  # Вектор ответов длины n.
        centers: np.ndarray,  # Матрица центров J x d.
        h: float,  # Текущий масштаб h.
        beta: np.ndarray,  # Текущее beta.
        directions: np.ndarray | None,  # Не используется в old.
        anisotropy: float | None,  # Не используется в old.
        b_value: float | None,  # Продольный масштаб b или None.
    ) -> LocalStatistics:
        """Вычисляет old-статистики Ima, N, S, VP.

        Вход:
            X: матрица наблюдений.
            y: вектор ответов.
            centers: локальные центры.
            h: текущий ортогональный bandwidth.
            beta: текущее EDR-направление.
            directions: не используется.
            anisotropy: не используется.
            b_value: bandwidth вдоль beta или None.
        Выход:
            LocalStatistics с imav, N, S, VP.
        """

        J, d = centers.shape
        imav = np.zeros((J, d + 1))
        n_all = np.zeros(J)
        s_all = np.zeros((J, d))
        vp_all = np.zeros((J, d, d))
        weight_means: list[float] = []

        for start in range(0, J, self.config.chunk_size):
            # Блочная обработка нужна особенно old-варианту: VP_j хранит d x d
            # матрицу для каждого центра.
            stop = min(start + self.config.chunk_size, J)
            diff = X[None, :, :] - centers[start:stop, None, :]
            norm2 = np.einsum("cnd,cnd->cn", diff, diff)
            if b_value is None:
                # Первый внешний шаг manifold_old.tex: изотропные веса.
                q = norm2 / (h * h)
            else:
                # Дальше old использует q = ||dx||^2 / h^2 + <dx,beta>^2 / b^2.
                proj2 = np.square(np.einsum("cnd,d->cn", diff, beta))
                q = norm2 / (h * h) + proj2 / (b_value * b_value)

            # Вычислитель собирает полный локальный момент KSe_j и Ima_j.
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

    def _solve_local_coefficients(
        self,
        stats: LocalStatistics,  # Статистики с N, S, VP.
        beta: np.ndarray,  # Текущее beta.
    ) -> tuple[np.ndarray, np.ndarray]:
        """Решает локальные коэффициенты old-варианта.

        Вход:
            stats: статистики с N_j, S_j и VP_j.
            beta: текущее направление.
        Выход:
            Кортеж intercepts и slopes.
        """

        if stats.N is None or stats.S is None or stats.VP is None:
            raise ValueError("Некорректные статистики old-варианта")
        J = stats.N.shape[0]
        intercepts = np.zeros(J)
        slopes = np.zeros(J)
        for j in range(J):
            # Лемма Lcjlj в manifold_old.tex: для фиксированного beta
            # решается двумерная задача наименьших квадратов с колонками
            # [N_j, S_j] и [S_j^T beta, VP_j beta].
            col0 = np.concatenate([[stats.N[j]], stats.S[j]])
            col1 = np.concatenate([[stats.S[j] @ beta], stats.VP[j] @ beta])
            design = np.column_stack([col0, col1])
            lhs = design.T @ design + self.config.ridge * np.eye(2)
            rhs = design.T @ stats.imav[j]
            intercepts[j], slopes[j] = safe_solve(lhs, rhs)
        return intercepts, slopes

    def _solve_beta(
        self,
        stats: LocalStatistics,  # Статистики с N, S, VP.
        intercepts: np.ndarray,  # Локальные свободные члены.
        slopes: np.ndarray,  # Локальные наклоны.
        prior: np.ndarray,  # beta предыдущего внешнего шага.
        lambda_penalty: float,  # Сила регуляризации.
    ) -> np.ndarray:
        """Решает beta для old-варианта.

        Вход:
            stats: статистики с S и VP.
            intercepts: локальные свободные члены.
            slopes: локальные наклоны.
            prior: направление регуляризации.
            lambda_penalty: сила регуляризации.
        Выход:
            Новый ненормированный beta.
        """

        if stats.N is None or stats.S is None or stats.VP is None:
            raise ValueError("Некорректные статистики old-варианта")
        d = stats.S.shape[1]
        # Лемма Lbeta в manifold_old.tex: собираем нормальные уравнения по beta
        # из скалярной части Ima_{j,0} и векторной части Ima_{j,1}.
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
        stats: LocalStatistics,  # Статистики с N, S, VP.
        beta: np.ndarray,  # Текущее beta.
        intercepts: np.ndarray,  # Локальные свободные члены.
        slopes: np.ndarray,  # Локальные наклоны.
        prior: np.ndarray,  # Направление регуляризации.
        lambda_penalty: float,  # Сила регуляризации.
    ) -> float:
        """Считает objective для old-варианта.

        Вход:
            stats: статистики с imav, N, S, VP.
            beta: текущее направление.
            intercepts: локальные свободные члены.
            slopes: локальные наклоны.
            prior: направление регуляризации.
            lambda_penalty: сила регуляризации.
        Выход:
            Значение objective.
        """

        if stats.N is None or stats.S is None or stats.VP is None:
            raise ValueError("Некорректные статистики old-варианта")
        total = 0.0
        for j, slope in enumerate(slopes):
            # Цель old: ||Ima_j - KSe_j [c_j, l_j beta]^T||^2.
            pred0 = intercepts[j] * stats.N[j] + slope * (stats.S[j] @ beta)
            pred1 = intercepts[j] * stats.S[j] + slope * (stats.VP[j] @ beta)
            residual = stats.imav[j] - np.concatenate([[pred0], pred1])
            total += float(np.sum(residual**2))
        total += lambda_penalty * float(np.sum((beta - prior) ** 2))
        return total
