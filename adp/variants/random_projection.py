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
        weight_means: list[float] = []
        rho = 1.0 if anisotropy is None else float(anisotropy)

        for start in range(0, J, self.config.chunk_size):
            # Блочная обработка ограничивает память: diff имеет размер
            # блок x n x d, а не J x n x d для всех центров сразу.
            stop = min(start + self.config.chunk_size, J)
            diff = X[None, :, :] - centers[start:stop, None, :]
            norm2 = np.einsum("cnd,cnd->cn", diff, diff)
            if anisotropy is None:
                # Первый внешний шаг из manifold_new.tex: изотропные веса.
                q = norm2 / (h * h)
            else:
                # Адаптивные веса new: q = (rho^2 ||dx||^2 + <dx,beta>^2) / h^2.
                proj2 = np.square(np.einsum("cnd,d->cn", diff, beta))
                q = (rho * rho * norm2 + proj2) / (h * h)

            # Вычислитель считает три суммы из формул для Ima_{j,phi}, S_{j,phi}
            # и U_{j,phi}; здесь только раскладываем блок обратно по центрам.
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

        if stats.S is None or stats.U is None:
            raise ValueError("Некорректные статистики new-варианта")
        J = stats.S.shape[0]
        intercepts = np.zeros(J)
        slopes = np.zeros(J)
        for j in range(J):
            # Для фиксированного beta TeX-цель по (c_j, l_j) становится
            # обычной двумерной задачей наименьших квадратов:
            # Ima_j ~= c_j S_j + l_j U_j beta.
            col0 = stats.S[j]
            col1 = stats.U[j] @ beta
            design = np.column_stack([col0, col1])
            lhs = design.T @ design + self.config.ridge * np.eye(2)
            rhs = design.T @ stats.imav[j]
            intercepts[j], slopes[j] = safe_solve(lhs, rhs)
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
        # Формируем нормальные уравнения леммы Lbeta для new-варианта:
        # сумма l_j^2 U_j^T U_j + lambda I.
        lhs = lambda_penalty * np.eye(d)
        rhs = lambda_penalty * prior
        for j, slope in enumerate(slopes):
            Uj = stats.U[j]
            # Из Ima_j вычитается вклад свободного члена c_j S_j.
            residual = stats.imav[j] - intercepts[j] * stats.S[j]
            lhs += slope * slope * (Uj.T @ Uj)
            rhs += slope * (Uj.T @ residual)
        return safe_solve(lhs + self.config.ridge * np.eye(d), rhs)

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
        total = 0.0
        for j, slope in enumerate(slopes):
            # Цель после исключения наблюдаемых величин:
            # ||Ima_j - c_j S_j - l_j U_j beta||^2.
            pred = intercepts[j] * stats.S[j] + slope * (stats.U[j] @ beta)
            total += float(np.sum((stats.imav[j] - pred) ** 2))
        total += lambda_penalty * float(np.sum((beta - prior) ** 2))
        return total
