from collections.abc import Callable
from dataclasses import dataclass

import ADP.calculus
import numpy as np
from ADP import ADP_Config, ADP_Data
from ADP.ADP_Solver import LSMR
from ADP.ADP_Statistic import calculate_statistics
from ADP.calculus import (
    Calculate_h0,
    Calculate_rho_k,
    Calculate_weight,
    Generate_proj,
    T_k,
)


@dataclass(slots=True)
class Generation_param:
    sigma_eps: float
    mu_eps: float
    sigma_x: float
    mu_x: float
    sigma_beta: float
    mu_beta: float
    sigma_beta_k: float
    mu_beta_k: float


def Generate_Data(
    gen_par: Generation_param, rng: np.random.Generator, n: int, d: int, f: Callable
) -> ADP_Data:

    X = rng.normal(loc=gen_par.mu_x, scale=gen_par.sigma_x, size=(d, n))
    beta = rng.normal(loc=gen_par.mu_beta, scale=gen_par.sigma_beta, size=d)
    noise = rng.normal(loc=gen_par.mu_eps, scale=gen_par.sigma_eps, size=d)
    Y = f(beta.T @ X) + noise

    return ADP_Data(X, Y, noise, beta)


class ADP_single_index:
    config: ADP_Config
    gen_param: Generation_param
    n: int
    d: int
    f: Callable

    rng: np.random.Generator
    data: ADP_Data
    x_j: np.ndarray
    type_beta: str
    beta_k: np.ndarray

    def __post_init__(self, config: ADP_Config, gen_param: Generation_param):
        self.rng = np.random.default_rng(seed=config.seed)
        self.data = Generate_Data(gen_param, self.rng, self.n, self.d, self.f)

        # Адаптировать под различные x_j
        self.x_j = self.data.X

        self.beta_k = self.rng.normal(
            loc=self.gen_param.mu_beta_k, scale=self.gen_param.sigma_beta_k, size=self.d
        )

    def step_k(self):
        pass

    def fit(self):
        h_k = Calculate_h0(
            self.data.X,
            self.x_j,
            self.config.N_loc,
            self.config.h_min,  # ty: ignore[invalid-argument-type]
            self.config.kernel,
        )
        Tk = T_k(h_k, 0, np.zeros(self.d))
        while True:
            # pyrefly: ignore [bad-argument-type]
            Phi = Generate_proj(self.rng, self.config.N_phi, Tk)
            weight = Calculate_weight(self.data.X, self.x_j, self.beta_k, Tk)
            stat = calculate_statistics(
                self.data.X, self.data.Y, weight, Phi, batch_size=32
            )
            beta_k, lj_k = LSMR()

            # pyrefly: ignore [unsupported-operation]
            if h_k / self.config.a < self.config.h_min:
                return beta_k
            else:
                h_k /= self.config.a
                rho_k = Calculate_rho_k(
                    self.data.X,
                    self.x_j,
                    beta_k,
                    h_k,
                    self.config.N_loc,
                    kernel=self.config.kernel,
                )
                Tk = T_k(h_k, rho_k, beta_k)
