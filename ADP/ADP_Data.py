import numpy as np

from ADP import ADP_Config


class ADP_Data:
    def __init__(self, config: ADP_Config | None = None):
        if config is None:
            config = ADP_Config()
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.beta = self._Generate_beta()
        self.X = self._Generate_X()
        self.noise = self._Generate_noise()
        self.Y = self._Generate_Y()

    # Переделать под фабрику для будущих Manifold и Multiindex
    def _Generate_beta(self, beta=None):
        if beta is None:
            beta = self.rng.normal(
                loc=self.config.mu_beta,
                scale=self.config.sigma_beta,
                size=self.config.d,
            )
            return beta / np.linalg.norm(beta)

        return beta

    def _Generate_X(self):
        X = self.rng.normal(
            loc=self.config.mu_x,
            scale=self.config.sigma_x,
            size=(self.config.d, self.config.n),
        )
        return X

    def _Generate_noise(self):
        eps = self.rng.normal(
            loc=self.config.mu_eps,
            scale=self.config.sigma_eps,
            size=(self.config.n),
        )
        return eps

    def _Generate_Y(self):
        return self.config.f(self.beta.T @ self.X) + self.noise

    def Generate_proj(self):
        phi = self.rng.normal(
            loc=self.config.mu_phi,
            scale=self.config.sigma_phi,
            size=(self.config.d, self.config.N_J),
        )

    def Initializе_param(self):
        return self.X, self.noise, self.beta, self.Y
