from collections.abc import Callable

import numpy as np

from ..engine.calculus import (
    calculate_h0_from_adp,
    calculate_rho_k_from_adp,
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
        if self.config.N_phi is None:
            raise ValueError("N_phi must be configured before fitting")
        rng = np.random.default_rng(seed=self.config.seed)

        rho_k = 1.0
        h_k = calculate_h0_from_adp(self.config, self.data)

        proj = generate_proj_from_adp(self.config, self.data, rng, beta_init, rho_k)

        weight = calculate_weight_from_adp(
            self.config, self.data, beta_init, h_k, rho_k
        )

        stat = calculate_statistics(
            X, Y, weight, proj, batch_size=self.config.batch_size
        )
        beta_k = self.solver(beta_init, stat.U, stat.I)

        while h_k / a > self.config.h_min:
            h_k = h_k / a
            rho_k = calculate_rho_k_from_adp(self.config, self.data, beta_k, h_k)
            if rho_k is None:
                raise RuntimeError("could not find a feasible localization factor")
            proj = generate_proj_from_adp(self.config, self.data, rng, beta_k, rho_k)
            weight = calculate_weight_from_adp(
                self.config, self.data, beta_k, h_k, rho_k
            )
            stat = calculate_statistics(
                X, Y, weight, proj, batch_size=self.config.batch_size
            )
            beta_k = self.solver(beta_k, stat.U, stat.I)

        return beta_k
