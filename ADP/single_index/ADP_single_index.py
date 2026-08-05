import numpy as np

from ADP import ADP_Config, ADP_Data
from ADP.ADP_statistic import *

from . import calculus
from .calculus import T_k


class ADP_single_index:
    Calculate_h0 = calculus.Calculate_h0
    Calculate_rho_k = calculus.Calculate_rho_k
    Calculate_weight = calculus.Calculate_weight
    Calculate_Tk_x = staticmethod(calculus.Calculate_Tk_x)
    Calculate_h_k = staticmethod(calculus.Calculate_h_k)
    Generate_proj = staticmethod(calculus.Generate_proj)

    def __init__(self, config: ADP_Config | None = None):
        if config is None:
            config = ADP_Config()

        self.config = config
        self.data = ADP_Data(config)
        self.rng = self.data.rng
        self.n = config.n
        self.d = config.d
        self.N_loc = config.N_loc
        self.N_lin = config.N_lin
        self.N_J = config.N_J
        self.N_phi = config.N_phi
        self.mu_phi = config.mu_phi
        self.sigma_phi = config.sigma_phi
        self.lam = config.lam
        self.kernel = config.kernel
        self.a = config.a
        self.h_min = config.h_min

        self.h_k = None
        self.rho_k = None

        self.X, self.noise = self.data.Initialize_input_data()

        self.beta = np.ndarray
        self.beta_k = None
        self.Y = np.ndarray
        self.x_j = None
        self.proj = None
        self.T_k = T_k()
        self.I = self.U = self.mass = self.mean = self.n_eff = self.eta = None

    # После отработки реализовать новый вариант
    def Initialize_beta(self) -> np.ndarray:
        return self.rng.normal(
            loc=self.config.mu_beta, scale=self.config.sigma_beta, size=self.d
        )

    def Generate_beta_0(self) -> np.ndarray:
        return self.rng.normal(
            loc=self.config.mu_beta, scale=self.config.sigma_beta, size=self.d
        )

    def Calculate_Y(self):
        return self.config.f(self.beta.T @ self.X) + self.noise

    def Initialize_x_j(self):
        # модифицировать под не стандартный выбор
        return self.X

    def Initialize_model(self):
        self.X, self.noise = self.data.Initialize_input_data()
        # pyrefly: ignore [bad-assignment]
        self.beta = self.Initialize_beta()
        self.Y = self.Calculate_Y()
        self.x_j = self.Initialize_x_j()
        self.h_k = self.Calculate_h0()
        self.beta_k = self.Generate_beta_0()
        self.T_k = T_k(1 / self.h_k, 1, np.zeros(self.d))

    def step_k(self):
        self.proj = self.Generate_proj(self.rng, self.N_phi, self.T_k)
        Calculate_statistic(self, self.N_phi, batch_size=128)
        # beta_k, l_k = solver.LSMR()

    def Model_fit(self) -> None:
        pass
