from functools import partial
from typing import cast

import numpy as np

from ...engine.common import utils
from ...engine.common.ADP_Statistic_engine import (
    calculate_statistics,
    calculate_statistics_gpu,
)
from ...engine.common.box_kernel import (
    NeighborhoodEngine,
    SparseStatisticsCache,
    initialize_basis_local_sparse,
    search_sparse_bandwidth,
    search_sparse_scale,
    sparse_kernel_parameters,
)
from ...engine.common.calculus import (
    calculate_rho_k,
    generate_proj,
    pairwise_distance2,
    search_bandwidth,
)
from ...engine.common.initialize import initialize_beta_local
from ...engine.common.logger import finish_tracking, start_tracking, track_stage
from ...engine.common.weights import calculate_weight
from ...engine.single_index import ADP_single_index_engine
from ...gpu import require_cupy
from ...solver.legacy_lsmr import solve as solve_lsmr
from ..ADP_Config import ADP_Config
from ..ADP_Solver import ADP_solver
from . import ADP_single_index_utils as model_utils
from .ADP_single_index_result import ADP_single_index_result


class ADP_single_index:
    def __init__(
        self,
        config: ADP_Config | None = None,
        solver: ADP_solver | None = None,
    ):
        """Настроить single-index ADP и совместимый legacy solver."""
        self.config = config or ADP_Config()
        self.solver = solver or ADP_solver(
            solve_lsmr,
            tol=1e-8,
        )
        self.ADP_single_index_result = ADP_single_index_result

    def fit(self, X, Y, *, progress=None):
        """Оценить индекс по ``X, Y`` и сохранить structured result в модели."""
        tracker = start_tracking()
        try:
            return self._fit(X, Y, tracker, progress)
        finally:
            self.profile_ = finish_tracking(tracker)

    def _fit(self, X, Y, tracker, progress):
        """Выполнить полный цикл инициализации, локализации и outer-шагов."""
        with track_stage(tracker, "initialization"):
            config = self.config
            statistics_function = calculate_statistics
            if getattr(config, "gpu", False):
                model_utils.require_gpu_solver(self.solver, solve_lsmr)
                require_cupy()
                statistics_function = calculate_statistics_gpu

            X, Y = utils._prepare_xy(X, Y)
            n, d = X.shape

            # Локальные имена короче self.config.N_loc.
            N_loc = config.N_loc
            N_lin = config.N_lin or 2 * d
            N_J = config.N_J or n
            N_phi = config.N_phi or min(N_loc, d)
            kernel = config.kernel
            a = config.a
            batch_size = config.batch_size
            lambda_penalty = config.lambda_penalty
            local_ridge = config.local_ridge
            outer_steps = config.outer_steps
            index_init = config.index_init
            seed = config.seed
            configured_h_min = config.h_min
            smart_weights = getattr(config, "smart_weights", False)
            estimator = getattr(config, "estimator", "legacy")
            sparse_kernel = sparse_kernel_parameters(kernel)

            model_utils.require_supported_initialization(index_init)
            model_utils.validate_model_sizes(n, d, N_loc, N_lin, N_J, index_init)

            scale = float(np.mean(np.std(X, axis=0)))
            h_min = configured_h_min or max(
                10.0 * scale / n,
                np.finfo(float).eps,
            )

            rng = np.random.default_rng(seed)
            if n == N_J:
                centers = X.copy()
            else:
                centers = X[rng.choice(n, size=N_J, replace=False)]
            if sparse_kernel is None:
                distance2 = pairwise_distance2(X, centers)
                neighborhood_engine = statistics_cache = None
            else:
                distance2 = None
                neighborhood_engine = NeighborhoodEngine(
                    X,
                    centers,
                    block_size=batch_size,
                )
                statistics_cache = SparseStatisticsCache()

            if index_init == "random":
                beta_init = rng.normal(size=d)
                beta_init /= np.linalg.norm(beta_init)
            elif sparse_kernel is not None:
                beta_init = initialize_basis_local_sparse(
                    X,
                    Y,
                    cast(NeighborhoodEngine, neighborhood_engine),
                    N_lin,
                    kernel,
                    local_ridge,
                    1,
                )[:, 0]
            else:
                beta_init = initialize_beta_local(
                    X,
                    Y,
                    centers,
                    cast(np.ndarray, distance2),
                    N_lin,
                    kernel,
                    local_ridge,
                )

            initial_beta = beta_init.copy()

            result = self.ADP_single_index_result(beta_init=initial_beta)

            h0 = (
                search_bandwidth(
                    cast(np.ndarray, distance2), N_loc, kernel, lower=h_min
                )
                if sparse_kernel is None
                else search_sparse_bandwidth(
                    cast(NeighborhoodEngine, neighborhood_engine),
                    N_loc,
                    kernel,
                    lower=h_min,
                )
            )
            h = h0
            rho = 1.0
            localization_beta = (
                np.zeros(d) if estimator == "legacy" else beta_init.copy()
            )
            solver_diagnostics = []
            k = 0

        while True:
            model_utils.require_outer_step(outer_steps, k)

            with track_stage(tracker, "directions"):
                directions = generate_proj(
                    rng,
                    N_J,
                    N_phi,
                    localization_beta,
                    rho,
                )
            with track_stage(tracker, "statistics"):
                if sparse_kernel is None:
                    weights = calculate_weight(
                        X,
                        centers,
                        localization_beta,
                        h,
                        rho,
                        kernel,
                        block_size=batch_size,
                        distance2=cast(np.ndarray, distance2),
                        estimator=estimator,
                    )
                    statistics = statistics_function(
                        X,
                        Y,
                        weights,
                        directions,
                        batch_size=batch_size,
                    )
                else:
                    weights = cast(
                        NeighborhoodEngine, neighborhood_engine
                    ).single_blocks(
                        localization_beta,
                        h,
                        rho,
                        kernel,
                    )
                    statistics = statistics_function(
                        X,
                        Y,
                        weights,
                        directions,
                        batch_size=batch_size,
                        sparse_cache=statistics_cache,
                    )
            with track_stage(tracker, "solver"):
                solver_result = self.solver.fit(
                    statistics,
                    beta_init,
                    lambda_penalty=lambda_penalty,
                    local_ridge=local_ridge,
                )

            beta = model_utils.validate_solver_index(solver_result.index, d)
            beta = ADP_single_index_engine.orient_index(beta, beta_init)

            diagnostics = dict(solver_result.diagnostics)
            solver_diagnostics.append(diagnostics)
            neighborhood_diagnostics = (
                {}
                if sparse_kernel is None
                else {
                    "support_edges": cast(
                        NeighborhoodEngine, neighborhood_engine
                    ).last_support_edges,
                    "boundary_edges": cast(
                        NeighborhoodEngine, neighborhood_engine
                    ).last_boundary_edges,
                    "support_reuse_hits": cast(
                        SparseStatisticsCache, statistics_cache
                    ).last_hits,
                }
            )
            result.trace.append(
                {
                    **diagnostics,
                    **neighborhood_diagnostics,
                    "k": k,
                    "h": float(h),
                    "rho": float(rho),
                    "mean_mass": float(np.mean(statistics.mass)),
                    "beta": beta.copy(),
                    "cosine_initial": float(abs(np.dot(initial_beta, beta))),
                }
            )
            if progress is not None:
                progress(dict(result.trace[-1]))

            with track_stage(tracker, "update"):
                next_h = h / a
                if next_h < h_min:
                    stop_reason = "h_min"
                    result.trace[-1]["stop_reason"] = stop_reason
                    break

                beta_init = beta
                if sparse_kernel is None:
                    next_rho = calculate_rho_k(
                        X,
                        centers,
                        beta_init,
                        next_h,
                        N_loc,
                        kernel,
                        distance2=cast(np.ndarray, distance2),
                        estimator=estimator,
                    )
                else:
                    next_rho = (
                        cast(
                            NeighborhoodEngine, neighborhood_engine
                        ).select_single_box_scale(
                            beta_init,
                            next_h,
                            N_loc,
                        )
                        if sparse_kernel == ("box", None)
                        else NotImplemented
                    )
                    if next_rho is NotImplemented:
                        next_rho = search_sparse_scale(
                            partial(
                                cast(
                                    NeighborhoodEngine, neighborhood_engine
                                ).single_blocks,
                                beta_init,
                                next_h,
                                kernel=kernel,
                                record=False,
                            ),
                            N_loc,
                            rowwise=sparse_kernel == ("box", None),
                        )
                if next_rho is None:
                    stop_reason = "local_mass_limit"
                    result.trace[-1]["stop_reason"] = stop_reason
                    break

                h = float(next_h)
                rho = next_rho
                localization_beta = beta_init
                k += 1

        result.Set_beta_final(beta)
        result.Set_stop_reason(stop_reason)
        self.result_ = result
        self.beta_ = result.beta_final
        self.coefficients_ = solver_result.coefficients
        self.effective_parameters_ = {
            "N_loc": N_loc,
            "N_lin": N_lin,
            "N_J": N_J,
            "N_phi": N_phi,
            "h_min": float(h_min),
            "smart_weights": smart_weights,
            "gpu": getattr(config, "gpu", False),
            "kernel_mode": (
                sparse_kernel[0] if sparse_kernel is not None else "callable"
            ),
            "kernel_tau": (sparse_kernel[1] if sparse_kernel is not None else None),
        }
        return self

    def _check_fitted(self) -> None:
        """Проверить наличие результата fit перед transform/predict-операцией."""
        model_utils.require_fitted(self)
