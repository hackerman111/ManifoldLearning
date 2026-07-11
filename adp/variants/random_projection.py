from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from ..common.types import LocalStatistics, VariantName
from ..engine.base import ADPBase


class RandomProjectionADP(ADPBase):
    """ADP из manifold_new.tex: локальные суммы по случайным направлениям."""

    variant: VariantName = "new"

    def _compute_statistics_default(
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
        directions = self.backend.asarray(directions)
        J, P, d = directions.shape
        X_backend, y_backend, centers_backend, directions_backend = self.backend.prepare_statistics_inputs(
            X,
            y,
            centers,
            directions,
        )
        accumulator = self.backend.create_statistics_accumulator(J, P, d)
        norm2_all = self._cached_pairwise_norm2(X, centers)
        proj2_all = self._cached_pairwise_projection2(X, centers, beta) if anisotropy is not None else None

        for start in range(0, J, self.config.chunk_size):
            stop = min(start + self.config.chunk_size, J)
            center_chunk = centers_backend[start:stop]
            norm2 = norm2_all[start:stop]
            if anisotropy is not None and proj2_all is None:
                raise RuntimeError("projection cache не подготовлен")
            proj2 = None if proj2_all is None else proj2_all[start:stop]
            q = self.backend.kernel_argument(
                norm2,
                h=h,
                projection2=proj2,
                anisotropy=anisotropy,
            )

            imav_chunk, s_chunk, u_chunk, n_chunk, weights_mean = self.backend.random_projection_sums(
                X=X_backend,
                y=y_backend,
                centers=center_chunk,
                directions=directions_backend[start:stop],
                q=q,
                kernel=self.config.kernel,
            )
            self.backend.accumulate_statistics(
                accumulator,
                start,
                stop,
                (imav_chunk, s_chunk, u_chunk, n_chunk, weights_mean),
            )

        imav, s_all, u_all, n_all, weights_mean = self.backend.finalize_statistics(accumulator)

        return LocalStatistics(
            variant="new",
            imav=imav,
            centers=centers,
            h=h,
            weights_mean=float(weights_mean),
            # Направления нужны только внутри _compute_statistics; сохранение
            # полного J x P x d массива дублировало бы рабочую память fit.
            directions=directions if self.config.save_directions else None,
            n_directions=P,
            S=s_all,
            U=u_all,
            N=n_all,
            anisotropy=anisotropy,
        )

    def _compute_statistics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        centers: np.ndarray,
        h: float,
        beta: np.ndarray,
        directions: np.ndarray | None,
        anisotropy: float | None,
    ) -> LocalStatistics:
        """Совместимый адаптер к выбранному statistics builder."""

        algorithm = getattr(self, "algorithm", None)
        if algorithm is None:
            return self._compute_statistics_default(
                X, y, centers, h, beta, directions, anisotropy
            )
        return algorithm.components["statistics_builder"].compute(
            X, y, centers, h, beta, directions, anisotropy
        )

    def _solve_local_coefficients_default(
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
        ubeta = stats.U @ beta
        numerator = np.sum(stats.imav * ubeta, axis=1)
        denominator = np.sum(ubeta * ubeta, axis=1)
        denominator = np.maximum(denominator, np.finfo(float).tiny)
        slopes = numerator / denominator
        return intercepts, slopes

    def _solve_local_coefficients(
        self,
        stats: LocalStatistics,
        beta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Совместимый адаптер к выбранному local solver."""

        algorithm = getattr(self, "algorithm", None)
        if algorithm is None:
            return self._solve_local_coefficients_default(stats, beta)
        return algorithm.components["local_solver"].solve(stats, beta)

    def _solve_beta_default(
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
        u_flat = stats.U.reshape(-1, d)
        slope_flat = np.repeat(np.asarray(slopes, dtype=float), stats.U.shape[1])
        slope_sq_flat = slope_flat**2
        residual_flat = residual.reshape(-1)

        def matvec(vector: np.ndarray) -> np.ndarray:
            projected = u_flat @ vector
            result = u_flat.T @ (slope_sq_flat * projected)
            result += regularization * vector
            return result

        rhs = u_flat.T @ (slope_flat * residual_flat)
        rhs = rhs + float(lambda_penalty) * prior
        operator = LinearOperator((d, d), matvec=matvec, dtype=float)
        preconditioner = None
        if self.config.use_cg_preconditioner:
            diagonal = np.sum((u_flat * u_flat) * slope_sq_flat[:, None], axis=0)
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

    def _solve_beta(
        self,
        stats: LocalStatistics,
        intercepts: np.ndarray,
        slopes: np.ndarray,
        prior: np.ndarray,
        lambda_penalty: float,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        """Совместимый адаптер к выбранному beta solver."""

        algorithm = getattr(self, "algorithm", None)
        if algorithm is None:
            return self._solve_beta_default(
                stats,
                intercepts,
                slopes,
                prior,
                lambda_penalty,
                x0=x0,
            )
        return algorithm.components["beta_solver"].solve(
            stats,
            intercepts,
            slopes,
            prior,
            lambda_penalty,
            x0=x0,
        )

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
        ubeta = stats.U @ beta
        pred = intercepts[:, None] * stats.S + slopes[:, None] * ubeta
        total = float(np.sum((stats.imav - pred) ** 2))
        total += lambda_penalty * float(np.sum((beta - prior) ** 2))
        return total
