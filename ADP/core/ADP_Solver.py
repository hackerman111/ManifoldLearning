from collections.abc import Callable

import numpy as np

from ..engine.calculus import (
    calculate_h0_from_adp,
    calculate_rho_k_from_adp,
    generate_isotropic_proj,
    generate_proj_from_adp,
)
from ..engine.statistic import calculate_statistics
from ..engine.weights import calculate_weight_from_adp
from ..solver.LSMR import lsmr
from .ADP_Config import ADP_Config
from .ADP_Data import ADP_Data


class ADP_Solver:
    def __init__(
        self,
        config: ADP_Config,
        data: ADP_Data,
        solver: Callable[..., np.ndarray] | None = None,
    ):
        self.config = config
        self.data = data
        self.solver = solver or lsmr

    def fit(self) -> np.ndarray:
        X, Y = self.data.X, self.data.Y
        beta_init = self.data.beta_init
        a = self.config.a
        n_phi = self.config.N_phi
        if n_phi is None:
            raise ValueError("N_phi must be configured before fitting")
        rng = np.random.default_rng(seed=self.config.seed)
        normalized = self.config.estimator == "new"

        def run_step(beta: np.ndarray, h: float, rho: float) -> np.ndarray:
            if self.config.direction_mode == "isotropic":
                proj = generate_isotropic_proj(
                    rng,
                    len(self.data.x_j),
                    n_phi,
                    X.shape[1],
                )
            else:
                proj = generate_proj_from_adp(
                    self.config,
                    self.data,
                    rng,
                    beta,
                    rho,
                )
            weight = calculate_weight_from_adp(
                self.config,
                self.data,
                beta,
                h,
                rho,
            )
            stat = calculate_statistics(
                X,
                Y,
                weight,
                proj,
                batch_size=self.config.batch_size,
                normalized=normalized,
            )
            return self.solver(
                beta,
                stat.U,
                stat.I,
                mass=stat.mass if normalized else None,
                lambda_prox=self.config.lambda_penalty,
                max_steps=3,
            )

        rho_k = 1.0
        h_k = calculate_h0_from_adp(self.config, self.data)
        beta_k = run_step(beta_init, h_k, rho_k)

        outer_iteration = 1
        while h_k / a > self.config.h_min and (
            self.config.outer_steps is None or outer_iteration < self.config.outer_steps
        ):
            h_k = h_k / a
            rho_k = calculate_rho_k_from_adp(self.config, self.data, beta_k, h_k)

            if rho_k is None:
                raise RuntimeError("could not find a feasible localization factor")
            beta_k = run_step(beta_k, h_k, rho_k)
            outer_iteration += 1

        return beta_k
