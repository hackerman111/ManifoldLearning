import numpy as np

from ..ADP_Config import ADP_Config
from ..ADP_Solver import ADP_solver
from ..ADP_Statistic import calculate_statistics
from ..calculus import (
    calculate_rho_k,
    calculate_weight,
    generate_proj,
    initialize_beta_local,
    pairwise_distance2,
    search_bandwidth,
)
from .solvers.LSMR import solve as solve_lsmr


class ADP_single_index:
    def __init__(
        self,
        config: ADP_Config | None = None,
        solver: ADP_solver | None = None,
    ):
        self.config = config or ADP_Config()
        self.solver = solver or ADP_solver(
            solve_lsmr,
            max_steps=2,
            tol=1e-6,
        )

    def fit(self, X, Y):
        X, Y = _prepare_inputs(X, Y)
        config = self.config
        n, d = X.shape

        # Локальные имена короче self.config.N_loc и не дублируют состояние модели.
        N_loc = config.N_loc
        N_lin = config.N_lin or 2 * d
        N_J = config.N_J or n
        N_phi = config.N_phi or min(N_loc, d)
        kernel = config.kernel
        a = config.a
        batch_size = config.batch_size
        lambda_penalty = config.lambda_penalty
        local_ridge = config.local_ridge
        outer_steps = config.outer_steps
        index_init = config.index_init
        seed = config.seed
        configured_h_min = config.h_min

        if N_loc > n:
            raise ValueError("N_loc cannot exceed n")
        if N_lin > n:
            raise ValueError("N_lin cannot exceed n")
        if index_init == "local" and N_lin <= d + 1:
            raise ValueError("N_lin must exceed d + 1 for local initialization")
        if not np.ceil(n / N_loc) <= N_J <= n:
            raise ValueError("N_J must lie between ceil(n / N_loc) and n")

        scale = float(np.mean(np.std(X, axis=0)))
        h_min = configured_h_min or max(
            10.0 * scale / n,
            np.finfo(float).eps,
        )

        rng = np.random.default_rng(seed)
        if N_J == n:
            centers = X.copy()
        else:
            centers = X[rng.choice(n, size=N_J, replace=False)]
        distance2 = pairwise_distance2(X, centers)

        if index_init == "random":
            beta_init = rng.normal(size=d)
            beta_init /= np.linalg.norm(beta_init)
        else:
            beta_init = initialize_beta_local(
                X,
                Y,
                centers,
                distance2,
                N_lin,
                kernel,
                local_ridge,
            )
        initial_beta = beta_init.copy()

        h0 = search_bandwidth(distance2, N_loc, kernel, lower=h_min)
        h = h0
        rho = 1.0
        localization_beta = np.zeros(d)
        trace = []
        solver_diagnostics = []
        coefficients = None
        stop_reason = None
        k = 0

        while True:
            if outer_steps is not None and k >= outer_steps:
                raise RuntimeError("outer_steps exhausted before reaching h_min")

            directions = generate_proj(
                rng,
                N_J,
                N_phi,
                localization_beta,
                rho,
            )
            weights = calculate_weight(
                X,
                centers,
                localization_beta,
                h,
                rho,
                kernel,
                block_size=batch_size,
            )
            statistics = calculate_statistics(
                X,
                Y,
                weights,
                directions,
                batch_size=batch_size,
            )
            result = self.solver.fit(
                statistics,
                beta_init,
                lambda_penalty=lambda_penalty,
                local_ridge=local_ridge,
            )

            beta = np.asarray(result.index, dtype=float)
            if beta.shape != (d,):
                raise RuntimeError("single-index solver must return shape (d,)")
            beta_norm = np.linalg.norm(beta)
            if not np.isfinite(beta_norm) or beta_norm == 0:
                raise RuntimeError("single-index solver returned an invalid index")
            beta /= beta_norm
            if np.dot(beta, beta_init) < 0:
                beta = -beta

            diagnostics = dict(result.diagnostics)
            solver_diagnostics.append(diagnostics)
            trace.append(
                {
                    **diagnostics,
                    "k": k,
                    "h": float(h),
                    "rho": float(rho),
                    "mean_mass": float(np.mean(statistics.mass)),
                    "beta": beta.copy(),
                }
            )
            coefficients = result.coefficients

            next_h = h / a
            if next_h < h_min:
                stop_reason = "h_min"
                trace[-1]["stop_reason"] = stop_reason
                break

            beta_init = beta
            next_rho = calculate_rho_k(
                X,
                centers,
                beta_init,
                next_h,
                N_loc,
                kernel,
                distance2=distance2,
            )
            if next_rho is None:
                stop_reason = "local_mass_limit"
                trace[-1]["stop_reason"] = stop_reason
                break

            h = float(next_h)
            rho = next_rho
            localization_beta = beta_init
            k += 1

        self.beta_init_ = initial_beta
        self.beta_ = beta
        self.projector_ = np.outer(beta, beta)
        self.coefficients_ = coefficients
        self.trace_ = trace
        self.solver_diagnostics_ = solver_diagnostics
        self.centers_ = centers
        self.h0_ = float(h0)
        self.h_k_ = float(h)
        self.rho_k_ = float(rho)
        self.stop_reason_ = stop_reason
        self.n_features_in_ = d
        self.effective_parameters_ = {
            "N_loc": N_loc,
            "N_lin": N_lin,
            "N_J": N_J,
            "N_phi": N_phi,
            "h_min": float(h_min),
        }
        return self

    def transform(self, X) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X must have shape (n, d) with the fitted d")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X @ self.beta_

    def score_direction(self, beta_true) -> float:
        self._check_fitted()
        beta_true = np.asarray(beta_true, dtype=float)
        if beta_true.shape != self.beta_.shape:
            raise ValueError("beta_true must have shape (d,)")
        norm = np.linalg.norm(beta_true)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("beta_true must be finite and non-zero")
        return float(abs(np.dot(self.beta_, beta_true / norm)))

    def _check_fitted(self) -> None:
        if not hasattr(self, "beta_"):
            raise RuntimeError("model is not fitted")


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
