import numpy as np

from ADP import ADP_Config


class ADP_Data:
    def __init__(self, config: ADP_Config | None = None):
        if config is None:
            config = ADP_Config()
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.X = self._Generate_X()
        self.noise = self._Generate_noise()

    def _Generate_X(self) -> np.ndarray:
        X = self.rng.normal(
            loc=self.config.mu_x,
            scale=self.config.sigma_x,
            size=(self.config.d, self.config.n),
        )
        return X

    def _Generate_noise(self) -> np.ndarray:
        eps = self.rng.normal(
            loc=self.config.mu_eps,
            scale=self.config.sigma_eps,
            size=(self.config.n),
        )
        return eps

    def Initialize_input_data(self):
        return self.X, self.noise
