from __future__ import annotations

import math
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr

from ADP.ADP_Statistic import calculate_statistics


class ADP_single_index:
    def __init__(
        self,
        *,
        n_centers=64,
        n_directions=8,
        N_loc=10,
        N_lin=None,
        lambda_penalty=1000.0,
        local_ridge=1e-8,
        outer_steps=8,
        inner_steps=2,
        bandwidth_decay=math.sqrt(2.0),
        h_min=None,
        tol=1e-6,
        batch_size=32,
        seed=42,
        beta_init="local",
    ):
        for name, value in (
            ("n_centers", n_centers),
            ("n_directions", n_directions),
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
            ("N_loc", N_loc),
            ("lambda_penalty", lambda_penalty),
            ("local_ridge", local_ridge),
            ("tol", tol),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if N_lin is not None and (not np.isfinite(N_lin) or N_lin <= 0):
            raise ValueError("N_lin must be finite and positive")
        if not np.isfinite(bandwidth_decay) or bandwidth_decay <= 1:
            raise ValueError("bandwidth_decay must exceed one")
        if h_min is not None and (not np.isfinite(h_min) or h_min <= 0):
            raise ValueError("h_min must be finite and positive")
        if beta_init not in {"local", "random"}:
            raise ValueError("beta_init must be 'local' or 'random'")

        self.n_centers = n_centers
        self.n_directions = n_directions
        self.N_loc = float(N_loc)
        self.N_lin = None if N_lin is None else float(N_lin)
        self.lambda_penalty = float(lambda_penalty)
        self.local_ridge = float(local_ridge)
        self.outer_steps = outer_steps
        self.inner_steps = inner_steps
        self.bandwidth_decay = float(bandwidth_decay)
        self.h_min = None if h_min is None else float(h_min)
        self.tol = float(tol)
        self.batch_size = batch_size
        self.seed = seed
        self.beta_init = beta_init
        self.beta_ = None
        self.beta_init_ = None
        self.slopes_ = None
        self.h0_ = None
        self.trace_ = []
        self.timings_ = {}
        self.weight_density_ = None
        self.mean_zero_weight_fraction_ = None

    def fit(self, X, Y):
        total_started = perf_counter()
        self.timings_ = {
            name: 0.0
            for name in (
                "initialization",
                "rho",
                "directions",
                "weights",
                "statistics",
                "slopes",
                "lsmr",
                "total",
            )
        }
        started = perf_counter()
        X, Y = self._prepare_inputs(X, Y)
        n, d = X.shape
        if self.N_loc > n:
            raise ValueError("N_loc cannot exceed n")

        rng = np.random.default_rng(self.seed)
        centers = X[rng.choice(n, size=min(self.n_centers, n), replace=False)]
        distance2 = self._pairwise_distance2(X, centers)
        if self.beta_init == "random":
            beta = rng.normal(size=d)
            beta /= np.linalg.norm(beta)
        else:
            N_lin = self.N_lin
            if N_lin is None:
                N_lin = min(n, max(2 * d + 2, n // max(1, int(self.N_loc))))
            if N_lin > n:
                raise ValueError("N_lin cannot exceed n")
            h_lin = self._search_bandwidth(distance2, N_lin)
            beta = self._initial_beta(X, Y, centers, distance2, h_lin)
        h = self._search_bandwidth(distance2, self.N_loc)
        feature_scale = float(np.mean(np.std(X, axis=0)))
        h_min = self.h_min or max(10 * feature_scale / n, np.finfo(float).eps)
        self.timings_["initialization"] = perf_counter() - started

        self.beta_init_ = beta.copy()
        self.h0_ = h
        self.trace_ = []
        self.weight_density_ = None
        self.mean_zero_weight_fraction_ = None
        zero_weight_fractions = []
        slopes = None
        for outer in range(self.outer_steps):
            rho = None
            if outer:
                next_h = h / self.bandwidth_decay
                if next_h < h_min:
                    break
                h = next_h
                started = perf_counter()
                rho = self._search_rho(X, centers, distance2, h, beta)
                self.timings_["rho"] += perf_counter() - started

            started = perf_counter()
            directions = self._directions(rng, centers.shape[0], d, beta, rho)
            self.timings_["directions"] += perf_counter() - started
            started = perf_counter()
            weights = self._weights(X, centers, distance2, h, beta, rho)
            self.timings_["weights"] += perf_counter() - started
            self.weight_density_ = float(np.count_nonzero(weights) / weights.size)
            zero_weight_fractions.append(1 - self.weight_density_)
            started = perf_counter()
            statistics = self._calculate_statistics(X, Y, weights, directions)
            self.timings_["statistics"] += perf_counter() - started
            beta, slopes, record = self._alternating(statistics, beta)
            record.update(
                outer=outer,
                h=float(h),
                rho=None if rho is None else float(rho),
                mean_mass=float(np.mean(statistics["mass"])),
                weight_density=self.weight_density_,
                zero_weight_fraction=zero_weight_fractions[-1],
            )
            self.trace_.append(record)

        if slopes is None:
            raise RuntimeError("ADP did not complete an outer step")
        self.beta_ = beta
        self.slopes_ = slopes
        self.mean_zero_weight_fraction_ = float(np.mean(zero_weight_fractions))
        self.timings_["total"] = perf_counter() - total_started
        return self

    @staticmethod
    def _prepare_inputs(X, Y):
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

    @staticmethod
    def _kernel(argument):
        return np.maximum(1 - np.square(argument), 0)

    @staticmethod
    def _pairwise_distance2(X, centers):
        distance2 = (
            np.square(centers).sum(axis=1)[:, None]
            + np.square(X).sum(axis=1)[None, :]
            - 2 * centers @ X.T
        )
        np.maximum(distance2, 0, out=distance2)
        return distance2

    def _mass(self, argument):
        return float(np.mean(np.sum(self._kernel(argument), axis=1)))

    def _search_bandwidth(self, distance2, target):
        low = np.finfo(float).eps
        high = max(float(np.sqrt(np.max(distance2, initial=0))), 1.0)
        for _ in range(80):
            if self._mass(distance2 / high**2) >= target:
                break
            high *= 2
        else:
            raise RuntimeError("could not bracket a feasible bandwidth")
        for _ in range(60):
            middle = (low + high) / 2
            if self._mass(distance2 / middle**2) >= target:
                high = middle
            else:
                low = middle
        return float(high)

    def _initial_beta(self, X, Y, centers, distance2, h_lin):
        weights = self._kernel(distance2 / h_lin**2)
        d = X.shape[1]
        ridge_rows = np.zeros((d, d + 1))
        ridge_rows[:, 1:] = np.sqrt(self.local_ridge) * np.eye(d)
        gradients = np.empty((centers.shape[0], d))
        for j, center in enumerate(centers):
            design = np.column_stack((np.ones(X.shape[0]), X - center))
            root_weight = np.sqrt(weights[j])
            augmented_design = np.vstack((design * root_weight[:, None], ridge_rows))
            augmented_y = np.concatenate((Y * root_weight, np.zeros(d)))
            gradients[j] = np.linalg.lstsq(augmented_design, augmented_y, rcond=None)[
                0
            ][1:]
        _, singular_values, right_vectors = np.linalg.svd(
            gradients, full_matrices=False
        )
        if singular_values[0] <= np.finfo(float).eps:
            raise RuntimeError("local gradients do not identify beta")
        beta = right_vectors[0]
        beta /= np.linalg.norm(beta)
        if beta[np.argmax(np.abs(beta))] < 0:
            beta = -beta
        return beta

    def _search_rho(self, X, centers, distance2, h, beta):
        projected = centers @ beta
        projection2 = np.square(projected[:, None] - (X @ beta)[None, :])

        def mass(rho):
            return self._mass((rho**2 * distance2 + projection2) / h**2)

        if mass(1.0) >= self.N_loc:
            return 1.0
        if mass(0.0) < self.N_loc:
            raise RuntimeError("N_loc cannot be preserved at rho=0")
        low, high = 0.0, 1.0
        for _ in range(50):
            middle = (low + high) / 2
            if mass(middle) >= self.N_loc:
                low = middle
            else:
                high = middle
        return float(low)

    def _directions(self, rng, centers, d, beta, rho):
        values = rng.normal(size=(centers, self.n_directions, d))
        if rho is not None:
            values = (
                rho * values + rng.normal(size=(centers, self.n_directions, 1)) * beta
            )
        norms = np.linalg.norm(values, axis=2, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError("generated a zero direction")
        return values / norms

    def _weights(self, X, centers, distance2, h, beta, rho):
        if rho is None:
            argument = distance2 / h**2
        else:
            projection2 = np.square((centers @ beta)[:, None] - (X @ beta)[None, :])
            argument = (rho**2 * distance2 + projection2) / h**2
        return self._kernel(argument)

    def _calculate_statistics(self, X, Y, weights, directions):
        return calculate_statistics(
            X,
            Y,
            weights,
            directions,
            batch_size=self.batch_size,
        )

    def _slopes(self, I, U, beta):
        projected = U @ beta
        return np.sum(I * projected, axis=1) / (
            np.sum(projected * projected, axis=1) + self.local_ridge
        )

    def _solve_beta(self, I, U, slopes, beta_prior):
        rows = I.size
        d = U.shape[2]
        sqrt_lambda = math.sqrt(self.lambda_penalty)

        def matvec(vector):
            data = (slopes[:, None] * (U @ vector)).ravel()
            return np.concatenate((data, sqrt_lambda * vector))

        def rmatvec(vector):
            data = vector[:rows].reshape(I.shape)
            return (
                np.einsum("j,jpd,jp->d", slopes, U, data, optimize=True)
                + sqrt_lambda * vector[rows:]
            )

        operator = LinearOperator(
            (rows + d, d), matvec=matvec, rmatvec=rmatvec, dtype=float
        )
        rhs = np.concatenate((I.ravel(), sqrt_lambda * beta_prior))
        result = lsmr(
            operator,
            rhs,
            atol=min(self.tol, 1e-8),
            btol=min(self.tol, 1e-8),
            maxiter=max(50, 5 * d),
        )
        beta = result[0]
        norm = np.linalg.norm(beta)
        if not np.all(np.isfinite(beta)) or not np.isfinite(norm) or norm == 0:
            raise RuntimeError("LSMR returned an invalid beta")
        beta /= norm
        if np.dot(beta, beta_prior) < 0:
            beta = -beta
        return beta, int(result[1]), int(result[2])

    def _alternating(self, statistics, beta):
        I, U = statistics["I"], statistics["U"]
        stop = iterations = 0
        delta = math.inf
        for inner in range(self.inner_steps):
            prior = beta
            started = perf_counter()
            slopes = self._slopes(I, U, prior)
            self.timings_["slopes"] += perf_counter() - started
            started = perf_counter()
            beta, stop, iterations = self._solve_beta(I, U, slopes, prior)
            self.timings_["lsmr"] += perf_counter() - started
            delta = min(np.linalg.norm(beta - prior), np.linalg.norm(beta + prior))
            if delta < self.tol:
                break
        started = perf_counter()
        slopes = self._slopes(I, U, beta)
        self.timings_["slopes"] += perf_counter() - started
        return (
            beta,
            slopes,
            {
                "inner_iterations": inner + 1,
                "lsmr_stop": stop,
                "lsmr_iterations": iterations,
                "beta_delta": float(delta),
            },
        )
