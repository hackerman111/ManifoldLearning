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
    calculate_alpha_k,
    generate_multi_proj,
    pairwise_distance2,
    search_bandwidth,
)
from ...engine.common.initialize import (
    initialize_basis_local,
    initialize_basis_pilot,
    initialize_basis_random,
)
from ...engine.common.logger import finish_tracking, start_tracking, track_stage
from ...engine.common.weights import calculate_multi_weight
from ...engine.multi_index import ADP_multi_index_engine
from ...gpu import require_cupy
from ...solver.legacy_lsmr import solve as solve_lsmr
from ..ADP_Config import ADP_Config
from ..ADP_Solver import ADP_solver
from . import ADP_multi_index_utils as model_utils
from .ADP_multi_index_result import ADP_multi_index_result


class ADP_multi_index:
    def __init__(
        self,
        index_dim: int,
        config: ADP_Config | None = None,
        solver: ADP_solver | None = None,
    ):
        """Настроить multi-index ADP для ``index_dim``-мерного подпространства."""
        self.index_dim = model_utils.validate_index_dim(index_dim)
        self.config = config or ADP_Config()
        self.solver = solver or ADP_solver(solve_lsmr, tol=1e-7 / 2)
        self.ADP_multi_index_result = ADP_multi_index_result

    def fit(self, X, Y, *, progress=None):
        """Оценить EDR-базис по ``X, Y`` и сохранить diagnostics."""
        tracker = start_tracking()
        try:
            return self._fit(X, Y, tracker, progress)
        finally:
            self.profile_ = finish_tracking(tracker)

    def _fit(self, X, Y, tracker, progress):
        """Выполнить инициализацию, локальные statistics и outer multi-шаги."""
        with track_stage(tracker, "initialization"):
            config = self.config
            statistics_function = calculate_statistics
            if getattr(config, "gpu", False):
                model_utils.require_gpu_solver(self.solver, solve_lsmr)
                require_cupy()
                statistics_function = calculate_statistics_gpu

            X, Y = utils._prepare_xy(X, Y)
            n, d = X.shape
            m = self.index_dim
            model_utils.validate_dimension(m, d)

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
            seed = config.seed
            smart_weights = getattr(config, "smart_weights", False)
            tensor = getattr(config, "multi_tensor", "orthogonal")
            sparse_kernel = sparse_kernel_parameters(kernel)

            model_utils.validate_model_sizes(n, d, N_loc, N_lin, N_J, config.index_init)

            scale = float(np.mean(np.std(X, axis=0)))
            h_min = config.h_min or max(
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

            if config.index_init == "random":
                basis_init = initialize_basis_random(rng, d, m)
            elif config.index_init == "pilot":
                basis_init = initialize_basis_pilot(X, Y, m, seed=seed)
            elif sparse_kernel is not None:
                basis_init = initialize_basis_local_sparse(
                    X,
                    Y,
                    cast(NeighborhoodEngine, neighborhood_engine),
                    N_lin,
                    kernel,
                    local_ridge,
                    m,
                )
            else:
                basis_init = initialize_basis_local(
                    X,
                    Y,
                    centers,
                    cast(np.ndarray, distance2),
                    N_lin,
                    kernel,
                    local_ridge,
                    m,
                )

            initial_basis = basis_init.copy()
            result = self.ADP_multi_index_result(beta_init=initial_basis)
            h = (
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
            alpha = 1.0
            localization_basis = basis_init
            localization_eigenvalues = np.ones(m)
            solver_diagnostics = []
            k = 0

        while True:
            model_utils.require_outer_step(outer_steps, k)

            with track_stage(tracker, "directions"):
                directions = generate_multi_proj(
                    rng,
                    N_J,
                    N_phi,
                    localization_basis,
                    localization_eigenvalues,
                    alpha,
                )
            with track_stage(tracker, "statistics"):
                if sparse_kernel is None:
                    weights = calculate_multi_weight(
                        X,
                        centers,
                        localization_basis,
                        localization_eigenvalues,
                        h,
                        alpha,
                        kernel,
                        block_size=batch_size,
                        distance2=cast(np.ndarray, distance2),
                        tensor=tensor,
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
                    ).multi_blocks(
                        localization_basis,
                        localization_eigenvalues,
                        h,
                        alpha,
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
                    basis_init,
                    lambda_penalty=lambda_penalty,
                    local_ridge=local_ridge,
                )

            basis = ADP_multi_index_engine.orthonormal_basis(
                solver_result.index, (d, m)
            )
            diagnostics = dict(solver_result.diagnostics)
            eigenvalues = model_utils.validate_eigenvalues(
                cast(np.ndarray, diagnostics.get("eigenvalues")), m
            )

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
                    "alpha": float(alpha),
                    "mean_mass": float(np.mean(statistics.mass)),
                    "basis": basis.copy(),
                    "eigenvalues": eigenvalues.copy(),
                    "distance_initial": ADP_multi_index_engine.subspace_distance(
                        initial_basis, basis
                    ),
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

                basis_init = basis
                if sparse_kernel is None:
                    next_alpha = calculate_alpha_k(
                        X,
                        centers,
                        basis,
                        eigenvalues,
                        next_h,
                        N_loc,
                        kernel,
                        distance2=cast(np.ndarray, distance2),
                        tensor=tensor,
                    )
                else:
                    next_alpha = (
                        cast(
                            NeighborhoodEngine, neighborhood_engine
                        ).select_multi_box_scale(
                            basis,
                            eigenvalues,
                            next_h,
                            N_loc,
                        )
                        if sparse_kernel == ("box", None)
                        else NotImplemented
                    )
                    if next_alpha is NotImplemented:
                        next_alpha = search_sparse_scale(
                            partial(
                                cast(
                                    NeighborhoodEngine, neighborhood_engine
                                ).multi_blocks,
                                basis,
                                eigenvalues,
                                next_h,
                                kernel=kernel,
                                record=False,
                            ),
                            N_loc,
                            rowwise=sparse_kernel == ("box", None),
                        )
                if next_alpha is None:
                    stop_reason = "local_mass_limit"
                    result.trace[-1]["stop_reason"] = stop_reason
                    break

                h = float(next_h)
                alpha = next_alpha
                localization_basis = basis
                localization_eigenvalues = eigenvalues
                k += 1

        result.Set_beta_final(basis)
        result.Set_stop_reason(stop_reason)
        self.result_ = result
        self.basis_ = result.beta_final
        self.beta_ = self.basis_
        self.eigenvalues_ = eigenvalues.copy()
        self.coefficients_ = solver_result.coefficients
        self.trace_ = result.trace
        self.solver_diagnostics_ = solver_diagnostics
        self.effective_parameters_ = {
            "N_loc": N_loc,
            "N_lin": N_lin,
            "N_J": N_J,
            "N_phi": N_phi,
            "h_min": float(h_min),
            "index_dim": m,
            "smart_weights": smart_weights,
            "gpu": getattr(config, "gpu", False),
            "kernel_mode": (
                sparse_kernel[0] if sparse_kernel is not None else "callable"
            ),
            "kernel_tau": (sparse_kernel[1] if sparse_kernel is not None else None),
        }
        self.n_features_in_ = d
        return self

    def transform(self, X):
        """Преобразовать наблюдения в координаты оценённого EDR-базиса."""
        self._check_fitted()
        return utils._prepare_transform(X, self.n_features_in_) @ self.basis_

    def _check_fitted(self) -> None:
        """Проверить, что basis и параметры данных созданы вызовом ``fit``."""
        model_utils.require_fitted(self)
