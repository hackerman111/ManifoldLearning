import numpy as np
from ADP import ADP_Config, ADP_Data
from ADP.ADP_statistic import calculate_statistics


class ADP_single_index:
    def __init__(self, config: ADP_Config | None = None, T_k=None):
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
        self.T_k = T_k

        self.X, self.noise = self.data.Initialize_input_data()
        self.beta = self.Initialize_beta()
        self.Y = self.Calculate_Y()
        self.x_j = self.Initialize_x_j()
        self.proj = None
        self.I = self.U = self.mass = self.mean = self.n_eff = self.eta = None

    # После отработки реализовать новый вариант
    def Initialize_beta(self) -> np.ndarray:
        return self.rng.normal(
            loc=self.config.mu_beta, scale=self.config.sigma_beta, size=self.d
        )

    def Calculate_Y(self) -> None:
        return self.config.f(self.beta.T @ self.X) + self.noise

    def Initialize_x_j(self):
        # модифицировать под не стандартный выбор
        return self.X

    @staticmethod
    def anisotropic_distance_squared(
        X: np.ndarray,
        center: np.ndarray,
        beta: np.ndarray,
        h: float,
        rho: float,
    ) -> np.ndarray:
        """
        X:      (n, d)
        center: (d,)
        beta:   (d,), единичный вектор
        """
        delta = X - center
        ordinary_sq_norm = np.sum(delta * delta, axis=1)
        parallel_component = delta @ beta

        return (rho**2 * ordinary_sq_norm + parallel_component**2) / h**2

    def Generate_proj(
        rng: np.random.Generator,
        beta: np.ndarray,
        rho: float,
        n_directions: int,
    ) -> np.ndarray:
        d = beta.size

        z = rng.standard_normal((n_directions, d))
        xi = rng.standard_normal((n_directions, 1))

        g = rho * z + xi * beta[None, :]
        return g / np.linalg.norm(g, axis=1, keepdims=True)

    def Calculate_statistic(self, weights, directions, batch_size=32) -> None:
        statistics = calculate_statistics(
            self.X.T, self.Y, weights, directions, batch_size
        )
        self.I = statistics["I"]
        self.U = statistics["U"]
        self.mass = statistics["mass"]
        self.mean = statistics["mean"]
        self.n_eff = statistics["n_eff"]
        self.eta = statistics["eta"]

    def Calculate_weight(self, kernel) -> np.ndarray:
        if self.T_k is None:
            raise ValueError("T_k is not set")
        return kernel(self.T_k(self.x_j))

    def Calculate_h0(self) -> float:
        X_sq = np.sum(self.X**2, axis=0)
        centers_sq = np.sum(self.x_j**2, axis=0)
        distances_sq = centers_sq[:, None] + X_sq[None, :] - 2 * self.x_j.T @ self.X
        np.maximum(distances_sq, 0, out=distances_sq)

        target = self.N_loc * self.x_j.shape[1]

        def enough(h):
            return np.sum(self.kernel(distances_sq / h**2)) >= target

        low = float(self.h_min)
        if low <= 0:
            raise ValueError("h_min must be positive")
        if enough(low):
            return low

        high = max(2 * low, np.sqrt(distances_sq.max()))
        for _ in range(100):
            if enough(high):
                break
            high *= 2
        else:
            raise ValueError("N_loc cannot be reached with the configured kernel")

        for _ in range(60):
            middle = (low + high) / 2
            if enough(middle):
                high = middle
            else:
                low = middle
        return high

    def Calculate_rho_k(self) -> None:
        pass

    def Calculate_T_k(self) -> None:
        pass

    def Model_step_0(self) -> None:
        pass

    def Model_step_k(self) -> None:
        pass

    def Model_fit(self) -> None:
        pass
