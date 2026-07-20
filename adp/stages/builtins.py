from __future__ import annotations

import math
from typing import Any

import numpy as np

from .contracts import ADPState, StageContext


class DefaultBetaInitializer:
    def __init__(self, context: StageContext) -> None:
        self.model = context.model

    def initialize(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.model._initial_beta_default(X, y)


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


class CpuBatchedStatisticsBuilder:
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
        return self.model._compute_statistics_cpu_batched(
            X, y, centers, h, beta, directions, anisotropy
        )


class CpuCompactFactoredStatisticsBuilder:
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
        return self.model._compute_statistics_cpu_compact_factored(
            X, y, centers, h, beta, directions, anisotropy
        )


class LeastSquaresLocalSolver:
    def __init__(self, context: StageContext) -> None:
        self.model = context.model

    def solve(self, statistics, beta: np.ndarray):
        return self.model._solve_local_coefficients_default(statistics, beta)


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
    "beta_initializer": {"default": DefaultBetaInitializer},
    "center_selector": {"random_sample": RandomCenterSelector},
    "bandwidth_selector": {"adaptive_mass": AdaptiveMassBandwidthSelector},
    "direction_sampler": {"random_sphere": RandomSphereDirectionSampler},
    "statistics_builder": {
        "random_projection": RandomProjectionStatisticsBuilder,
        "cpu_batched": CpuBatchedStatisticsBuilder,
        "cpu_compact_factored": CpuCompactFactoredStatisticsBuilder,
    },
    "local_solver": {"least_squares": LeastSquaresLocalSolver},
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
