import numpy as np


class ADP_Config:
    def __init__(
        self,
        seed=42,
        sigma_x=1.0,
        mu_x=0.0,
        sigma_eps=1.0,
        mu_eps=0.0,
        sigma_beta=1.0,
        mu_beta=0.0,
        sigma_phi=1.0,
        mu_phi=0.0,
        f=lambda x: np.sin(x),
        d=100,
        n=1000,
        N_loc=10,
        N_lin: int | None = None,
        N_J: int | None = None,
        N_phi: int | None = None,
        lam=1,
        kernel=lambda x: np.maximum(0, 1 - x**2),
        a=np.sqrt(2),
        h_min: float | None = None,
    ):
        self.seed = seed
        self.sigma_x = sigma_x
        self.mu_x = mu_x
        # Переделать под разные варианты шума
        self.sigma_eps = sigma_eps
        self.mu_eps = mu_eps

        self.f = f
        self.d = d
        self.mu_beta = mu_beta
        self.sigma_beta = sigma_beta
        self.mu_phi = mu_phi
        self.sigma_phi = sigma_phi
        self.n = n
        self.N_loc = N_loc

        if N_lin is None:
            N_lin = 2 * d
        self.N_lin = N_lin

        if N_J is None:
            N_J = self.n
        self.N_J = N_J

        if N_phi is None:
            N_phi = N_loc
        self.N_phi = N_phi

        self.lam = lam
        self.kernel = kernel
        self.a = a
        if h_min is None:
            h_min = 10 * self.sigma_x / n
        self.h_min = h_min

    def Get_config(self):
        return (
            self.seed,
            self.sigma_x,
            self.mu_x,
            self.sigma_eps,
            self.mu_eps,
            self.f,
            self.d,
            self.n,
            self.mu_beta,
            self.sigma_beta,
            self.mu_phi,
            self.sigma_phi,
            self.N_loc,
            self.N_lin,
            self.N_J,
            self.N_phi,
            self.lam,
            self.kernel,
            self.a,
            self.h_min,
        )
