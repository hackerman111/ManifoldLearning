from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from ..common.types import LocalStatistics, VariantName
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
        norm2_all = self._cached_pairwise_norm2(X, centers)
        proj2_all = self._cached_pairwise_projection2(X, centers, beta) if anisotropy is not None else None

        for start in range(0, J, self.config.chunk_size):
            stop = min(start + self.config.chunk_size, J)
            center_chunk = centers[start:stop]
            norm2 = norm2_all[start:stop]
            if anisotropy is None:
                # Первый внешний шаг из manifold_new.tex: изотропные веса.
                q = norm2 / (h * h)
            else:
                # Адаптивные веса new: q = (rho^2 ||dx||^2 + <dx,beta>^2) / h^2.
                if proj2_all is None:
                    raise RuntimeError("projection cache не подготовлен")
                proj2 = proj2_all[start:stop]
                q = (float(anisotropy) * float(anisotropy) * norm2 + proj2) / (h * h)

            imav_chunk, s_chunk, u_chunk, n_chunk, weights_mean = self.backend.random_projection_sums(
                X=X,
                y=y,
                centers=center_chunk,
                directions=directions[start:stop],
                q=q,
                kernel=self.config.kernel,
            )
            n_all[start:stop] = n_chunk
            weight_means.append(weights_mean)
            imav[start:stop] = imav_chunk
            s_all[start:stop] = s_chunk
            u_all[start:stop] = u_chunk

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
        ubeta = np.einsum("jpd,d->jp", stats.U, beta, optimize=True)
        numerator = np.einsum("jp,jp->j", stats.imav, ubeta, optimize=True)
        denominator = np.einsum("jp,jp->j", ubeta, ubeta, optimize=True)
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
        x0: np.ndarray | None = None,  # Стартовая точка CG или None.
    ) -> np.ndarray:
        """Решает beta для new-варианта.

        Вход:
            stats: статистики с U_j.
            intercepts: локальные свободные члены.
            slopes: локальные наклоны.
            prior: направление регуляризации.
            lambda_penalty: сила регуляризации.
            x0: warm-start для CG; регуляризация все равно идет к prior.
        Выход:
            Новый ненормированный beta.
        """

        if stats.S is None or stats.U is None:
            raise ValueError("Некорректные статистики new-варианта")
        d = stats.U.shape[2]
        residual = stats.imav - intercepts[:, None] * stats.S
        regularization = float(lambda_penalty) + float(self.config.ridge)

        def matvec(vector: np.ndarray) -> np.ndarray:
            projected = np.einsum("jpd,d->jp", stats.U, vector, optimize=True)
            result = np.einsum("j,jp,jpd->d", slopes**2, projected, stats.U, optimize=True)
            result += regularization * vector
            return result

        rhs = np.einsum("j,jp,jpd->d", slopes, residual, stats.U, optimize=True)
        rhs += float(lambda_penalty) * prior
        operator = LinearOperator((d, d), matvec=matvec, dtype=float)
        preconditioner = None
        if self.config.use_cg_preconditioner:
            diagonal = np.einsum("j,jpd,jpd->d", slopes**2, stats.U, stats.U, optimize=True)
            diagonal += regularization
            inverse_diagonal = 1.0 / np.maximum(diagonal, np.finfo(float).tiny)
            preconditioner = LinearOperator(
                (d, d),
                matvec=lambda vector: inverse_diagonal * vector,
                dtype=float,
            )
        beta, info = cg(
            operator,
            rhs,
            x0=prior if x0 is None else x0,
            M=preconditioner,
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
        ubeta = np.einsum("jpd,d->jp", stats.U, beta, optimize=True)
        pred = intercepts[:, None] * stats.S + slopes[:, None] * ubeta
        total = float(np.sum((stats.imav - pred) ** 2))
        total += lambda_penalty * float(np.sum((beta - prior) ** 2))
        return total
