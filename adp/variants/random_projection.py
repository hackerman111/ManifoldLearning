from __future__ import annotations

import math
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from ..common.types import LocalStatistics, VariantName
from ..common.utils import stable_l2_norm
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

        return self._compute_statistics_with_backend_sums(
            X,
            y,
            centers,
            h,
            beta,
            directions,
            anisotropy,
        )

    def _compute_statistics_with_backend_sums(
        self,
        X: np.ndarray,
        y: np.ndarray,
        centers: np.ndarray,
        h: float,
        beta: np.ndarray,
        directions: np.ndarray | None,
        anisotropy: float | None,
    ) -> LocalStatistics:
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
        distance_started = time.perf_counter()
        norm2_all = self._cached_pairwise_norm2(X, centers)
        proj2_all = self._cached_pairwise_projection2(X, centers, beta) if anisotropy is not None else None
        distance_time = time.perf_counter() - distance_started
        weight_sum2 = (
            np.zeros(J, dtype=float) if self.config.record_telemetry else None
        )
        weight_nonzero = (
            np.zeros(J, dtype=int) if self.config.record_telemetry else None
        )
        min_weight = (
            np.zeros(J, dtype=float) if self.config.record_telemetry else None
        )
        max_weight = (
            np.zeros(J, dtype=float) if self.config.record_telemetry else None
        )
        weights_time = 0.0
        statistics_time = 0.0

        for start in range(0, J, self.config.chunk_size):
            stop = min(start + self.config.chunk_size, J)
            center_chunk = centers_backend[start:stop]
            norm2 = norm2_all[start:stop]
            if anisotropy is not None and proj2_all is None:
                raise RuntimeError("projection cache не подготовлен")
            proj2 = None if proj2_all is None else proj2_all[start:stop]
            argument_started = time.perf_counter()
            q = self.backend.kernel_argument(
                norm2,
                h=h,
                projection2=proj2,
                anisotropy=anisotropy,
            )
            distance_time += time.perf_counter() - argument_started

            chunk = self.backend.random_projection_sums(
                X=X_backend,
                y=y_backend,
                centers=center_chunk,
                directions=directions_backend[start:stop],
                q=q,
                kernel=self.config.kernel,
                record_telemetry=self.config.record_telemetry,
            )
            imav_chunk, s_chunk, u_chunk, n_chunk, weights_mean = chunk[:5]
            self.backend.accumulate_statistics(
                accumulator,
                start,
                stop,
                (imav_chunk, s_chunk, u_chunk, n_chunk, weights_mean),
            )
            if self.config.record_telemetry:
                block = chunk[5]
                assert weight_sum2 is not None
                assert weight_nonzero is not None
                assert min_weight is not None
                assert max_weight is not None
                weight_sum2[start:stop] = block["sum_w2"]
                weight_nonzero[start:stop] = block["nonzero"]
                min_weight[start:stop] = block["min_weight"]
                max_weight[start:stop] = block["max_weight"]
                weights_time += float(block["weights_time_sec"])
                statistics_time += float(block["statistics_time_sec"])

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
            weight_sum2=weight_sum2,
            weight_nonzero=weight_nonzero,
            min_weight=min_weight,
            max_weight=max_weight,
            distance_time_sec=distance_time if self.config.record_telemetry else 0.0,
            weights_time_sec=weights_time,
            statistics_time_sec=statistics_time,
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

        if stats.S is None or stats.U is None:
            raise ValueError("Некорректные статистики new-варианта")
        projected = stats.U @ beta
        design = np.stack((stats.S, projected), axis=-1)
        gram = np.swapaxes(design, 1, 2) @ design
        gram += float(self.config.ridge) * np.eye(
            2,
            dtype=gram.dtype,
        )[None, :, :]
        rhs = np.einsum("jpk,jp->jk", design, stats.imav, optimize=True)
        try:
            coefficients = np.linalg.solve(gram, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            coefficients = np.empty((design.shape[0], 2), dtype=design.dtype)
            ridge = float(self.config.ridge)
            ridge_scale = design.dtype.type(np.sqrt(ridge))
            ridge_rows = ridge_scale * np.eye(2, dtype=design.dtype)
            ridge_targets = np.zeros(2, dtype=design.dtype)
            for center_index in range(design.shape[0]):
                augmented_design = np.concatenate(
                    (design[center_index], ridge_rows),
                    axis=0,
                )
                augmented_response = np.concatenate(
                    (
                        np.asarray(
                            stats.imav[center_index],
                            dtype=design.dtype,
                        ),
                        ridge_targets,
                    ),
                    axis=0,
                )
                coefficients[center_index] = np.linalg.lstsq(
                    augmented_design,
                    augmented_response,
                    rcond=None,
                )[0]
        return coefficients[:, 0], coefficients[:, 1]

    def _solve_local_coefficients_zero_intercept(
        self,
        stats: LocalStatistics,
        beta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Решает ADP-регрессию с зафиксированным нулевым intercept."""

        if stats.U is None:
            raise ValueError("Некорректные статистики new-варианта")
        dtype = np.dtype(self.config.dtype)
        projected = np.asarray(stats.U, dtype=dtype) @ np.asarray(
            beta,
            dtype=dtype,
        )
        response = np.asarray(stats.imav, dtype=dtype)
        numerator = np.sum(response * projected, axis=1, dtype=dtype)
        denominator = np.sum(projected * projected, axis=1, dtype=dtype)
        denominator = np.maximum(denominator, np.finfo(dtype).tiny)
        intercepts = np.zeros(projected.shape[0], dtype=dtype)
        slopes = np.asarray(numerator / denominator, dtype=dtype)
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
        prior: np.ndarray,  # beta предыдущего внутреннего шага.
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
        solver_dtype = np.dtype(self.config.dtype)
        d = stats.U.shape[2]
        imav = np.asarray(stats.imav, dtype=solver_dtype)
        s_values = np.asarray(stats.S, dtype=solver_dtype)
        u_values = np.asarray(stats.U, dtype=solver_dtype)
        intercept_values = np.asarray(intercepts, dtype=solver_dtype)
        prior_values = np.asarray(prior, dtype=solver_dtype)
        lambda_value = solver_dtype.type(lambda_penalty)
        regularization = solver_dtype.type(
            float(lambda_penalty) + float(self.config.ridge)
        )
        residual = imav - intercept_values[:, None] * s_values
        u_flat = u_values.reshape(-1, d)
        slope_flat = np.repeat(
            np.asarray(slopes, dtype=solver_dtype),
            u_values.shape[1],
        )
        slope_sq_flat = slope_flat**2
        residual_flat = residual.reshape(-1)

        def matvec(vector: np.ndarray) -> np.ndarray:
            vector_values = np.asarray(vector, dtype=solver_dtype)
            projected = u_flat @ vector_values
            result = u_flat.T @ (slope_sq_flat * projected)
            result += regularization * vector_values
            return result

        rhs = u_flat.T @ (slope_flat * residual_flat)
        rhs = rhs + lambda_value * prior_values
        if not np.all(np.isfinite(rhs)):
            self._last_solver_telemetry = {
                "gradient_norm": np.nan,
                "linear_residual_norm": np.nan,
                "relative_linear_residual": np.nan,
                "linear_solver_iterations": 0,
                "linear_solver_status": "invalid_system",
                "solver_residual_trace": (),
                "scipy_info": None,
            }
            raise RuntimeError("CG получил неконечную правую часть")
        rhs_norm = stable_l2_norm(rhs)
        dtype_info = np.finfo(solver_dtype)
        residual_scale = max(rhs_norm, float(dtype_info.eps))
        operator = LinearOperator((d, d), matvec=matvec, dtype=solver_dtype)
        preconditioner = None
        if self.config.use_cg_preconditioner:
            diagonal = np.sum((u_flat * u_flat) * slope_sq_flat[:, None], axis=0)
            diagonal += regularization
            if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
                self._last_solver_telemetry = {
                    "gradient_norm": np.nan,
                    "linear_residual_norm": np.nan,
                    "relative_linear_residual": np.nan,
                    "linear_solver_iterations": 0,
                    "linear_solver_status": "invalid_system",
                    "solver_residual_trace": (),
                    "scipy_info": None,
                }
                raise RuntimeError("CG получил некорректный предобуславливатель")
            inverse_diagonal = solver_dtype.type(1.0) / np.maximum(
                diagonal,
                dtype_info.tiny,
            )
            preconditioner = LinearOperator(
                (d, d),
                matvec=lambda vector: inverse_diagonal
                * np.asarray(vector, dtype=solver_dtype),
                dtype=solver_dtype,
            )
        iterations = 0
        residual_trace: list[float] = []

        def record_iteration(candidate: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1
            if self.config.record_solver_trace:
                residual = matvec(candidate) - rhs
                if np.all(np.isfinite(residual)):
                    residual_trace.append(
                        stable_l2_norm(residual) / residual_scale
                    )
                else:
                    residual_trace.append(np.nan)

        beta, info = cg(
            operator,
            rhs,
            x0=np.asarray(
                prior_values if x0 is None else x0,
                dtype=solver_dtype,
            ),
            M=preconditioner,
            rtol=self.config.tol,
            atol=0.0,
            maxiter=max(50, min(500, 5 * d)),
            callback=record_iteration,
        )
        beta = np.asarray(beta, dtype=solver_dtype)
        status = "converged" if info == 0 else "max_iterations"
        if info < 0:
            status = "breakdown"
        beta_is_finite = bool(np.all(np.isfinite(beta)))
        beta_norm = stable_l2_norm(beta) if beta_is_finite else math.nan
        if (not beta_is_finite or beta_norm == 0.0) and status != "breakdown":
            status = "invalid_solution"
        residual = matvec(beta) - rhs
        residual_is_finite = bool(np.all(np.isfinite(residual)))
        absolute_residual = (
            stable_l2_norm(residual) if residual_is_finite else math.nan
        )
        if not residual_is_finite and status != "breakdown":
            status = "invalid_solution"
        self._last_solver_telemetry = {
            "gradient_norm": 2.0 * absolute_residual,
            "linear_residual_norm": absolute_residual,
            "relative_linear_residual": absolute_residual / residual_scale,
            "linear_solver_iterations": iterations,
            "linear_solver_status": status,
            "solver_residual_trace": tuple(residual_trace),
            "scipy_info": int(info),
        }
        if status in {"breakdown", "invalid_solution"}:
            raise RuntimeError(f"CG завершился с ошибкой {status}")
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
        ridge = float(self.config.ridge)
        total += ridge * float(
            np.sum(intercepts**2) + np.sum(slopes**2) + np.sum(beta**2)
        )
        total += float(lambda_penalty) * float(np.sum((beta - prior) ** 2))
        return total
