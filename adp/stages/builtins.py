from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..common.utils import stable_l2_norm, unit_vector
from .contracts import ADPState, StageContext


def _e1(dimension: int, dtype: np.dtype[Any] = np.dtype(float)) -> np.ndarray:
    beta = np.zeros(dimension, dtype=dtype)
    beta[0] = 1.0
    return beta


def _finite_unit_or_e1(values: np.ndarray, dimension: int) -> np.ndarray:
    beta = np.asarray(values, dtype=float).reshape(-1)
    if (
        beta.shape != (dimension,)
        or not np.all(np.isfinite(beta))
        or stable_l2_norm(beta) <= np.finfo(float).eps
    ):
        return _e1(dimension)
    return unit_vector(beta)


class DefaultBetaInitializer:
    def __init__(self, context: StageContext) -> None:
        self.model = context.model

    def initialize(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.model._initial_beta_default(X, y)


class E1BetaInitializer:
    def __init__(self, context: StageContext) -> None:
        self.backend = context.backend

    def initialize(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        del y
        return self.backend.asarray(_e1(X.shape[1]))


class PcaBetaInitializer:
    def __init__(self, context: StageContext) -> None:
        self.backend = context.backend

    def initialize(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        del y
        X_values = np.asarray(self.backend.to_numpy(X), dtype=float)
        centered = X_values - X_values.mean(axis=0, keepdims=True)
        try:
            _, _, right_vectors = np.linalg.svd(
                centered,
                full_matrices=False,
            )
            beta = right_vectors[0]
        except np.linalg.LinAlgError:
            covariance = centered.T @ centered
            _, vectors = np.linalg.eigh(covariance)
            beta = vectors[:, -1]
        beta = _finite_unit_or_e1(beta, X_values.shape[1])
        pivot = int(np.argmax(np.abs(beta)))
        if beta[pivot] < 0.0:
            beta = -beta
        return self.backend.asarray(beta)


class RidgeBetaInitializer:
    def __init__(self, context: StageContext, eta_scale: float) -> None:
        self.backend = context.backend
        self.eta_scale = float(eta_scale)

    def initialize(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X_values = np.asarray(self.backend.to_numpy(X), dtype=float)
        y_values = np.asarray(self.backend.to_numpy(y), dtype=float).reshape(-1)
        centered_y = y_values - y_values.mean()
        gram = X_values.T @ X_values
        rhs = X_values.T @ centered_y
        dimension = X_values.shape[1]
        eta = self.eta_scale * float(np.trace(gram)) / dimension
        system = gram + eta * np.eye(dimension)
        try:
            if self.eta_scale == 0.0:
                beta, *_ = np.linalg.lstsq(system, rhs, rcond=None)
            else:
                beta = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            beta, *_ = np.linalg.lstsq(system, rhs, rcond=None)
        return self.backend.asarray(_finite_unit_or_e1(beta, dimension))


class RandomCenterSelector:
    def __init__(self, context: StageContext) -> None:
        self.model = context.model

    def select(self, X: np.ndarray) -> np.ndarray:
        return self.model._choose_centers_default(X)


class AdaptiveMassBandwidthSelector:
    def __init__(self, context: StageContext) -> None:
        self.model = context.model

    def select_initial(self, X: np.ndarray, centers: np.ndarray, index: Any = None) -> float:
        return self.model._select_isotropic_bandwidth_default(X, centers, index)

    def select_anisotropy(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        h: float,
        beta: np.ndarray,
    ) -> float:
        return self.model._select_new_anisotropy_default(X, centers, h, beta)


class FixedLocalMassBandwidthSelector:
    def __init__(
        self,
        context: StageContext,
        quantile: float | None,
    ) -> None:
        self.model = context.model
        self.quantile = quantile

    def select_initial(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        index: Any = None,
    ) -> float:
        diff_norm2 = self.model._cached_pairwise_norm2(X, centers)
        high_hint = None
        if index is not None:
            k = min(
                max(1, int(math.ceil(self.model.config.min_neighbors))),
                X.shape[0],
            )
            kth = index.kth_distances(centers, k)
            if kth is not None and np.all(np.isfinite(kth)):
                hint_quantile = (
                    0.5
                    if self.quantile is None
                    else min(1.0, max(0.0, 1.0 - self.quantile))
                )
                high_hint = float(np.nanquantile(kth, hint_quantile))

        def score_for(h: float) -> float:
            return self.model.backend.local_mass_score(
                diff_norm2 / (h * h),
                self.model.config.kernel,
                quantile=self.quantile,
            )

        return self.model._binary_search_scale(score_for, high_hint)

    def select_anisotropy(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        h: float,
        beta: np.ndarray,
    ) -> float:
        return self.model._select_new_anisotropy_default(X, centers, h, beta)


class KnnQuantileBandwidthSelector:
    def __init__(
        self,
        context: StageContext,
        neighbor_multiplier: int,
    ) -> None:
        self.model = context.model
        self.neighbor_multiplier = int(neighbor_multiplier)

    def select_initial(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        index: Any = None,
    ) -> float:
        neighbor_count = min(
            X.shape[0],
            max(
                1,
                int(
                    math.ceil(
                        self.model.config.min_neighbors
                        * self.neighbor_multiplier
                    )
                ),
            ),
        )
        kth = (
            None
            if index is None
            else index.kth_distances(centers, neighbor_count)
        )
        if kth is None or not np.all(np.isfinite(kth)):
            distance2 = np.asarray(
                self.model.backend.to_numpy(
                    self.model._cached_pairwise_norm2(X, centers)
                ),
                dtype=float,
            )
            kth2 = np.partition(
                distance2,
                neighbor_count - 1,
                axis=1,
            )[:, neighbor_count - 1]
            kth = np.sqrt(np.maximum(kth2, 0.0))
        h0 = float(np.quantile(np.asarray(kth, dtype=float), 0.9))
        return max(h0, float(np.finfo(float).eps))

    def select_anisotropy(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        h: float,
        beta: np.ndarray,
    ) -> float:
        return self.model._select_new_anisotropy_default(X, centers, h, beta)


class RandomSphereDirectionSampler:
    def __init__(self, context: StageContext) -> None:
        self.model = context.model

    def prepare(
        self,
        centers: np.ndarray,
        d: int,
        directions: np.ndarray | None,
        *,
        beta: np.ndarray | None = None,
        anisotropy: float | None = None,
    ) -> np.ndarray | None:
        if directions is not None:
            return self.model._prepare_directions_default(centers, d, directions)
        if beta is None or anisotropy is None:
            return self.model._prepare_directions_default(centers, d, None)
        return self.model._sample_directions(
            centers.shape[0],
            self.model.config.n_directions,
            d,
            beta=beta,
            anisotropy=anisotropy,
        )


class RandomProjectionStatisticsBuilder:
    def __init__(self, context: StageContext) -> None:
        self.model = context.model

    def compute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        centers: np.ndarray,
        h: float,
        beta: np.ndarray,
        directions: np.ndarray | None,
        anisotropy: float | None,
    ):
        return self.model._compute_statistics_default(
            X, y, centers, h, beta, directions, anisotropy
        )


class LeastSquaresLocalSolver:
    def __init__(self, context: StageContext) -> None:
        self.model = context.model

    def solve(self, statistics, beta: np.ndarray):
        return self.model._solve_local_coefficients_default(statistics, beta)


class ZeroInterceptLocalSolver:
    def __init__(self, context: StageContext) -> None:
        self.model = context.model

    def solve(self, statistics, beta: np.ndarray):
        return self.model._solve_local_coefficients_zero_intercept(statistics, beta)


class ConjugateGradientBetaSolver:
    def __init__(self, context: StageContext) -> None:
        self.model = context.model

    def solve(
        self,
        statistics,
        intercepts: np.ndarray,
        slopes: np.ndarray,
        prior: np.ndarray,
        lambda_penalty: float,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.model._solve_beta_default(
            statistics,
            intercepts,
            slopes,
            prior,
            lambda_penalty,
            x0=x0,
        )


class ConvergenceStopRule:
    def __init__(self, context: StageContext) -> None:
        self.config = context.config

    def should_stop(
        self,
        phase: str,
        state: ADPState,
        *,
        step: Any = None,
        **metrics: Any,
    ) -> bool:
        if phase == "inner":
            beta_delta = float(metrics.get("beta_delta", math.inf))
            objective_delta = float(metrics.get("objective_delta", math.inf))
            return beta_delta < self.config.tol or objective_delta < self.config.tol
        if phase == "outer":
            anisotropy = state.anisotropy
            return (
                self.config.anisotropy_min is not None
                and anisotropy is not None
                and float(anisotropy) <= self.config.anisotropy_min
            )
        raise ValueError("phase должен быть 'inner' или 'outer'")


BUILTIN_STAGE_TYPES = {
    "beta_initializer": {
        "default": DefaultBetaInitializer,
        "e1": E1BetaInitializer,
        "pca": PcaBetaInitializer,
        "ridge_0": lambda context: RidgeBetaInitializer(context, 0.0),
        "ridge_1e-4": lambda context: RidgeBetaInitializer(context, 1e-4),
        "ridge_1e-5": lambda context: RidgeBetaInitializer(context, 1e-5),
        "ridge_1e-6": lambda context: RidgeBetaInitializer(context, 1e-6),
        "ridge_1e-7": lambda context: RidgeBetaInitializer(context, 1e-7),
        "ridge_1e-2": lambda context: RidgeBetaInitializer(context, 1e-2),
    },
    "center_selector": {"random_sample": RandomCenterSelector},
    "bandwidth_selector": {
        "adaptive_mass": AdaptiveMassBandwidthSelector,
        "local_mass_mean": lambda context: FixedLocalMassBandwidthSelector(
            context,
            None,
        ),
        "local_mass_q0": lambda context: FixedLocalMassBandwidthSelector(
            context,
            0.0,
        ),
        "local_mass_q05": lambda context: FixedLocalMassBandwidthSelector(
            context,
            0.05,
        ),
        "local_mass_q10": lambda context: FixedLocalMassBandwidthSelector(
            context,
            0.1,
        ),
        "local_mass_q25": lambda context: FixedLocalMassBandwidthSelector(
            context,
            0.25,
        ),
        "knn_q90_k1": lambda context: KnnQuantileBandwidthSelector(
            context,
            1,
        ),
        "knn_q90_k2": lambda context: KnnQuantileBandwidthSelector(
            context,
            2,
        ),
        "knn_q90_k4": lambda context: KnnQuantileBandwidthSelector(
            context,
            4,
        ),
    },
    "direction_sampler": {"random_sphere": RandomSphereDirectionSampler},
    "statistics_builder": {"random_projection": RandomProjectionStatisticsBuilder},
    "local_solver": {
        "least_squares": LeastSquaresLocalSolver,
        "zero_intercept": ZeroInterceptLocalSolver,
    },
    "beta_solver": {"cg": ConjugateGradientBetaSolver},
    "stop_rule": {"convergence": ConvergenceStopRule},
}


def build_builtin_stage(
    category: str,
    implementation: str,
    context: StageContext,
):
    try:
        stage_type = BUILTIN_STAGE_TYPES[category][implementation]
    except KeyError as exc:
        raise ValueError(
            f"Нет встроенного этапа {category!r} ({implementation!r})"
        ) from exc
    return stage_type(context)
