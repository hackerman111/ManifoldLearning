from __future__ import annotations

import numpy as np
from ADP.ADP_Config import ADP_Config
from ADP.ADP_statistic import calculate_statistics
from ADP.single_index.solvers.LSMR import solve as solve_lsmr


class ADP_single_index:
    """Single-index structure-adaptive average derivative procedure."""

    def __init__(
        self,
        config: ADP_Config | None = None,
        *,
        outer_steps: int = 8,
        inner_steps: int = 2,
        local_ridge: float = 1e-8,
        tol: float = 1e-6,
        batch_size: int = 32,
        beta_init: str = "local",
    ):
        self.config = ADP_Config() if config is None else config
        for name, value in (
            ("N_J", self.config.N_J),
            ("N_phi", self.config.N_phi),
            ("outer_steps", outer_steps),
            ("inner_steps", inner_steps),
            ("batch_size", batch_size),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("N_loc", self.config.N_loc),
            ("N_lin", self.config.N_lin),
            ("lambda penalty", self.config.lam),
            ("local_ridge", local_ridge),
            ("tol", tol),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.config.a) or self.config.a <= 1:
            raise ValueError("a must exceed one")
        if self.config.h_min is not None and (
            not np.isfinite(self.config.h_min) or self.config.h_min <= 0
        ):
            raise ValueError("h_min must be finite and positive")
        if not callable(self.config.kernel):
            raise TypeError("kernel must be callable")
        if beta_init not in {"local", "random"}:
            raise ValueError("beta_init must be 'local' or 'random'")

        self.outer_steps = outer_steps
        self.inner_steps = inner_steps
        self.local_ridge = float(local_ridge)
        self.tol = float(tol)
        self.batch_size = batch_size
        self.beta_init = beta_init

        self.X_ = self.Y_ = self.centers_ = self.distance2_ = None
        self.beta_init_ = self.beta_ = self.slopes_ = None
        self.h0_ = self.h_k = self.rho_k = self.T_k = None
        self.trace_: list[dict[str, int | float | None]] = []

    def fit(self, X, Y):
        """Run initialization and the structure-adaptive outer iterations."""
        self.X_, self.Y_ = self._prepare_inputs(X, Y)
        self._initialize()

        for k in range(self.outer_steps):
            self.trace_.append(self._run_iteration(k))
            if k + 1 == self.outer_steps or not self._advance_localization():
                break

        return self

    def _initialize(self) -> None:
        n, d = self.X_.shape
        if self.config.N_loc > n:
            raise ValueError("N_loc cannot exceed n")
        if self.config.N_lin > n:
            raise ValueError("N_lin cannot exceed n")
        if self.beta_init == "local" and self.config.N_lin <= d + 1:
            raise ValueError("N_lin must exceed d + 1 for local initialization")

        self.rng_ = np.random.default_rng(self.config.seed)
        indices = self.rng_.choice(
            n,
            size=min(self.config.N_J, n),
            replace=False,
        )
        self.centers_ = self.X_[indices]
        self.distance2_ = self._pairwise_distance2(self.X_, self.centers_)

        if self.beta_init == "random":
            beta = self.rng_.normal(size=d)
            beta /= np.linalg.norm(beta)
        else:
            h_lin = self._search_bandwidth(self.config.N_lin)
            beta = self._initial_beta(h_lin)

        self.beta_init_ = beta.copy()
        self.beta_ = beta
        self.slopes_ = None
        self.h0_ = self._search_bandwidth(self.config.N_loc)
        self.h_k = self.h0_
        self.h_min_ = self.config.h_min
        if self.h_min_ is None:
            scale = float(np.mean(np.std(self.X_, axis=0)))
            self.h_min_ = max(10 * scale / n, np.finfo(float).eps)
        self.rho_k = None
        self.T_k = np.eye(d) / self.h_k
        self.trace_ = []

    def _run_iteration(self, k: int) -> dict[str, int | float | None]:
        directions = self._sample_directions()
        weights = self._calculate_weights()
        statistics = calculate_statistics(
            self.X_,
            self.Y_,
            weights,
            directions,
            batch_size=self.batch_size,
        )
        self.beta_, self.slopes_, record = solve_lsmr(
            statistics,
            self.beta_,
            lambda_penalty=float(self.config.lam),
            local_ridge=self.local_ridge,
            max_steps=self.inner_steps,
            tol=self.tol,
        )

        density = float(np.count_nonzero(weights) / weights.size)
        return {
            **record,
            "k": k,
            "h": float(self.h_k),
            "rho": None if self.rho_k is None else float(self.rho_k),
            "mean_mass": float(np.mean(statistics["mass"])),
            "weight_density": density,
        }

    def _advance_localization(self) -> bool:
        next_h = self.h_k / self.config.a
        if next_h < self.h_min_:
            return False
        self.h_k = float(next_h)
        self.Calculate_rho_k()
        self.Calculate_T_k()
        return True

    def Calculate_rho_k(self) -> None:
        """Find the largest rho in [0, 1] preserving the required local mass."""
        projected_centers = self.centers_ @ self.beta_
        projection2 = np.square(
            projected_centers[:, None] - (self.X_ @ self.beta_)[None, :]
        )

        def enough(rho):
            argument = (rho**2 * self.distance2_ + projection2) / self.h_k**2
            return self._mean_mass(argument) >= self.config.N_loc

        if enough(1.0):
            self.rho_k = 1.0
            return
        if not enough(0.0):
            raise RuntimeError("N_loc cannot be preserved at rho=0")

        low, high = 0.0, 1.0
        for _ in range(50):
            middle = (low + high) / 2
            if enough(middle):
                low = middle
            else:
                high = middle
        self.rho_k = float(low)

    def Calculate_T_k(self) -> None:
        """Update the positive square root of the localizing tensor."""
        beta_projection = np.outer(self.beta_, self.beta_)
        self.T_k = (
            self.rho_k * np.eye(self.X_.shape[1])
            + (np.sqrt(1 + self.rho_k**2) - self.rho_k) * beta_projection
        ) / self.h_k

    def _sample_directions(self) -> np.ndarray:
        values = self.rng_.normal(
            size=(
                self.centers_.shape[0],
                self.config.N_phi,
                self.X_.shape[1],
            )
        )
        values = values @ self.T_k.T
        norms = np.linalg.norm(values, axis=2, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError("generated a zero direction")
        return values / norms

    def _calculate_weights(self) -> np.ndarray:
        transformed_X = self.X_ @ self.T_k.T
        transformed_centers = self.centers_ @ self.T_k.T
        return self.config.kernel(
            self._pairwise_distance2(transformed_X, transformed_centers)
        )

    def _search_bandwidth(self, target: float) -> float:
        low = np.finfo(float).eps
        high = max(
            float(np.sqrt(np.max(self.distance2_, initial=0))),
            1.0,
        )
        for _ in range(80):
            if self._mean_mass(self.distance2_ / high**2) >= target:
                break
            high *= 2
        else:
            raise RuntimeError("could not bracket a feasible bandwidth")

        for _ in range(60):
            middle = (low + high) / 2
            if self._mean_mass(self.distance2_ / middle**2) >= target:
                high = middle
            else:
                low = middle
        return float(high)

    def _initial_beta(self, h_lin: float) -> np.ndarray:
        weights = self.config.kernel(self.distance2_ / h_lin**2)
        n, d = self.X_.shape
        ridge_rows = np.zeros((d, d + 1))
        ridge_rows[:, 1:] = np.sqrt(self.local_ridge) * np.eye(d)
        gradients = np.empty((self.centers_.shape[0], d))

        for j, center in enumerate(self.centers_):
            design = np.column_stack((np.ones(n), self.X_ - center))
            root_weight = np.sqrt(weights[j])
            augmented_design = np.vstack((design * root_weight[:, None], ridge_rows))
            augmented_y = np.concatenate((self.Y_ * root_weight, np.zeros(d)))
            gradients[j] = np.linalg.lstsq(
                augmented_design,
                augmented_y,
                rcond=None,
            )[
                0
            ][1:]

        _, singular_values, right_vectors = np.linalg.svd(
            gradients,
            full_matrices=False,
        )
        if singular_values[0] <= np.finfo(float).eps:
            raise RuntimeError("local gradients do not identify beta")
        beta = right_vectors[0]
        beta /= np.linalg.norm(beta)
        if beta[np.argmax(np.abs(beta))] < 0:
            beta = -beta
        return beta

    def _mean_mass(self, argument: np.ndarray) -> float:
        return float(np.mean(np.sum(self.config.kernel(argument), axis=1)))

    @staticmethod
    def _pairwise_distance2(
        X: np.ndarray,
        centers: np.ndarray,
    ) -> np.ndarray:
        distance2 = (
            np.square(centers).sum(axis=1)[:, None]
            + np.square(X).sum(axis=1)[None, :]
            - 2 * centers @ X.T
        )
        np.maximum(distance2, 0, out=distance2)
        return distance2

    @staticmethod
    def _prepare_inputs(X, Y) -> tuple[np.ndarray, np.ndarray]:
        arrays = []
        for value, name in ((X, "X"), (Y, "Y")):
            array = np.asarray(value)
            if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
                raise TypeError(f"{name} must have a real numeric dtype")
            array = array.astype(float, copy=False)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")
            arrays.append(array)
        X, Y = arrays
        if X.ndim != 2 or 0 in X.shape:
            raise ValueError("X must have non-empty shape (n, d)")
        if Y.shape != (X.shape[0],):
            raise ValueError("Y must have shape (n,)")
        if X.shape[0] <= X.shape[1] + 1:
            raise ValueError("n must exceed d + 1")
        return X, Y
