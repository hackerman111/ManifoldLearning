from dataclasses import dataclass, field

import numpy as np

from ..ADP_Config import ADP_Config
from ..ADP_Solver import ADP_solver
from ..ADP_Statistic import calculate_statistics
from ..engine import utils
from ..engine.calculus import (
    calculate_rho_k,
    calculate_weight,
    generate_proj,
    initialize_beta_local,
    pairwise_distance2,
    search_bandwidth,
)
from ..engine.logger import finish_tracking, start_tracking, track_stage
from .solvers.VarPro import solve as solve_varpro


@dataclass(slots=True)
class ADP_single_index_result:
    beta_init: np.ndarray
    beta_final: np.ndarray | None = None
    beta_true: np.ndarray | None = None
    trace: list[dict] = field(default_factory=list)
    stop_reason: str | None = None

    cosine_init: float | None = None
    cosine_final: float | None = None

    def Calculate_cosine(self):
        if self.beta_true is None or self.beta_final is None:
            raise RuntimeError("beta_true and beta_final are required")
        self.cosine_init = float(abs(np.dot(self.beta_init, self.beta_true)))
        self.cosine_final = float(abs(np.dot(self.beta_final, self.beta_true)))

    def Set_beta_init(self, beta_init):
        self.beta_init = beta_init

    def Set_beta_true(self, beta_true):
        self.beta_true = beta_true

    def Set_beta_final(self, beta_final):
        self.beta_final = beta_final

    def Set_stop_reason(self, stop_reason):
        self.stop_reason = stop_reason


class ADP_single_index:
    def __init__(
        self,
        config: ADP_Config | None = None,
        solver: ADP_solver | None = None,
    ):
        self.config = config or ADP_Config()
        self.solver = solver or ADP_solver(
            solve_varpro,
            tol=1e-7 / 2,
        )
        self.ADP_single_index_result = ADP_single_index_result

    def fit(self, X, Y, *, progress=None):
        tracker = start_tracking()
        try:
            return self._fit(X, Y, tracker, progress)
        finally:
            self.profile_ = finish_tracking(tracker)

    def _fit(self, X, Y, tracker, progress):
        with track_stage(tracker, "initialization"):
            X, Y = utils._prepare_xy(X, Y)
            config = self.config
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

            if index_init == "pilot":
                raise ValueError("pilot initialization is multi-index only")

            utils._check_model_sizes(n, d, N_loc, N_lin, N_J, index_init)

            scale = float(np.mean(np.std(X, axis=0)))
            h_min = configured_h_min or max(
                10.0 * scale / n,
                np.finfo(float).eps,
            )

            rng = np.random.default_rng(seed)
            if N_J == n:
                centers = X.copy()
            else:
                centers = X[rng.choice(n, size=N_J, replace=False)]
            distance2 = pairwise_distance2(X, centers)

            if index_init == "random":
                beta_init = rng.normal(size=d)
                beta_init /= np.linalg.norm(beta_init)
            else:
                beta_init = initialize_beta_local(
                    X,
                    Y,
                    centers,
                    distance2,
                    N_lin,
                    kernel,
                    local_ridge,
                )

            initial_beta = beta_init.copy()

            result = self.ADP_single_index_result(beta_init=initial_beta)

            h0 = search_bandwidth(distance2, N_loc, kernel, lower=h_min)
            h = h0
            rho = 1.0
            localization_beta = np.zeros(d)
            solver_diagnostics = []
            k = 0

        while True:
            if outer_steps is not None and k >= outer_steps:
                raise RuntimeError("outer_steps exhausted before reaching h_min")

            with track_stage(tracker, "directions"):
                directions = generate_proj(
                    rng,
                    N_J,
                    N_phi,
                    localization_beta,
                    rho,
                )
            with track_stage(tracker, "statistics"):
                weights = calculate_weight(
                    X,
                    centers,
                    localization_beta,
                    h,
                    rho,
                    kernel,
                    block_size=batch_size,
                    distance2=distance2,
                )
                statistics = calculate_statistics(
                    X,
                    Y,
                    weights,
                    directions,
                    batch_size=batch_size,
                )
            with track_stage(tracker, "solver"):
                solver_result = self.solver.fit(
                    statistics,
                    beta_init,
                    lambda_penalty=lambda_penalty,
                    local_ridge=local_ridge,
                )

            beta = np.asarray(solver_result.index, dtype=float)
            if beta.shape != (d,):
                raise RuntimeError("single-index solver must return shape (d,)")
            beta_norm = np.linalg.norm(beta)
            if not np.isfinite(beta_norm) or beta_norm == 0:
                raise RuntimeError("single-index solver returned an invalid index")
            beta /= beta_norm
            if np.dot(beta, beta_init) < 0:
                beta = -beta

            diagnostics = dict(solver_result.diagnostics)
            solver_diagnostics.append(diagnostics)
            result.trace.append(
                {
                    **diagnostics,
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
                next_rho = calculate_rho_k(
                    X,
                    centers,
                    beta_init,
                    next_h,
                    N_loc,
                    kernel,
                    distance2=distance2,
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
        }
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "beta_"):
            raise RuntimeError("model is not fitted")
