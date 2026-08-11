from dataclasses import dataclass, field
from math import sqrt

import numpy as np

from ..ADP_Config import ADP_Config
from ..ADP_Solver import ADP_solver
from ..ADP_Statistic import calculate_statistics
from ..engine import utils
from ..engine.calculus import (
    calculate_alpha_k,
    calculate_multi_weight,
    generate_multi_proj,
    initialize_basis_local,
    initialize_basis_pilot,
    initialize_basis_random,
    pairwise_distance2,
    search_bandwidth,
)
from ..engine.logger import finish_tracking, start_tracking, track_stage
from ..single_index.solvers.LSMR import solve as solve_lsmr


@dataclass(slots=True)
class ADP_multi_index_result:
    beta_init: np.ndarray
    beta_final: np.ndarray | None = None
    beta_true: np.ndarray | None = None
    trace: list[dict] = field(default_factory=list)
    stop_reason: str | None = None

    dist_init: float | None = None
    dist_final: float | None = None

    def Calculate_dist(self):
        if self.beta_true is None or self.beta_final is None:
            raise RuntimeError("beta_true and beta_final are required")

        true_projector, shape = _projector(self.beta_true, "beta_true")
        init_projector, _ = _projector(self.beta_init, "beta_init", shape)
        final_projector, _ = _projector(self.beta_final, "beta_final", shape)
        scale = sqrt(2.0 * shape[1])
        self.dist_init = float(
            np.linalg.norm(true_projector - init_projector, ord="fro") / scale
        )
        self.dist_final = float(
            np.linalg.norm(true_projector - final_projector, ord="fro") / scale
        )

    def Set_beta_init(self, beta_init):
        self.beta_init = beta_init

    def Set_beta_true(self, beta_true):
        self.beta_true = beta_true

    def Set_beta_final(self, beta_final):
        self.beta_final = beta_final

    def Set_stop_reason(self, stop_reason):
        self.stop_reason = stop_reason


class ADP_multi_index:
    def __init__(
        self,
        index_dim: int,
        config: ADP_Config | None = None,
        solver: ADP_solver | None = None,
    ):
        if isinstance(index_dim, bool) or not isinstance(index_dim, (int, np.integer)):
            raise TypeError("index_dim must be an integer")
        if index_dim < 1:
            raise ValueError("index_dim must be positive")

        self.index_dim = int(index_dim)
        self.config = config or ADP_Config()
        self.solver = solver or ADP_solver(solve_lsmr, tol=1e-7 / 2)
        self.ADP_multi_index_result = ADP_multi_index_result

    def fit(self, X, Y, *, progress=None):
        tracker = start_tracking()
        try:
            return self._fit(X, Y, tracker, progress)
        finally:
            self.profile_ = finish_tracking(tracker)

    def _fit(self, X, Y, tracker, progress):
        with track_stage(tracker, "initialization"):
            X, Y = utils._prepare_xy(X, Y)
            config = self.config
            n, d = X.shape
            m = self.index_dim
            if m >= d:
                raise ValueError("index_dim must be smaller than d")

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
            seed = config.seed
            smart_weights = config.smart_weights

            utils._check_model_sizes(n, d, N_loc, N_lin, N_J, config.index_init)

            scale = float(np.mean(np.std(X, axis=0)))
            h_min = config.h_min or max(
                10.0 * scale / n,
                np.finfo(float).eps,
            )

            rng = np.random.default_rng(seed)
            if N_J == n:
                centers = X.copy()
            else:
                centers = X[rng.choice(n, size=N_J, replace=False)]
            distance2 = pairwise_distance2(X, centers)

            if config.index_init == "random":
                basis_init = initialize_basis_random(rng, d, m)
            elif config.index_init == "pilot":
                basis_init = initialize_basis_pilot(X, Y, m, seed=seed)
            else:
                basis_init = initialize_basis_local(
                    X,
                    Y,
                    centers,
                    distance2,
                    N_lin,
                    kernel,
                    local_ridge,
                    m,
                )

            initial_basis = basis_init.copy()
            result = self.ADP_multi_index_result(beta_init=initial_basis)
            h = search_bandwidth(distance2, N_loc, kernel, lower=h_min)
            alpha = 1.0
            localization_basis = basis_init
            localization_eigenvalues = np.ones(m)
            solver_diagnostics = []
            k = 0

        while True:
            if outer_steps is not None and k >= outer_steps:
                raise RuntimeError("outer_steps exhausted before reaching h_min")

            with track_stage(tracker, "directions"):
                directions = generate_multi_proj(
                    rng,
                    N_J,
                    N_phi,
                    localization_basis,
                    localization_eigenvalues,
                    alpha,
                )
            with track_stage(tracker, "statistics"):
                weights = calculate_multi_weight(
                    X,
                    centers,
                    localization_basis,
                    localization_eigenvalues,
                    h,
                    alpha,
                    kernel,
                    block_size=batch_size,
                    distance2=distance2,
                    smart=smart_weights,
                )
                statistics = calculate_statistics(
                    X,
                    Y,
                    weights,
                    directions,
                    batch_size=batch_size,
                )
            with track_stage(tracker, "solver"):
                solver_result = self.solver.fit(
                    statistics,
                    basis_init,
                    lambda_penalty=lambda_penalty,
                    local_ridge=local_ridge,
                )

            basis = _orthonormal_basis(solver_result.index, (d, m))
            diagnostics = dict(solver_result.diagnostics)
            if "eigenvalues" not in diagnostics:
                raise RuntimeError("multi-index solver must return eigenvalues")
            eigenvalues = np.asarray(diagnostics["eigenvalues"], dtype=float)
            if (
                eigenvalues.shape != (m,)
                or not np.all(np.isfinite(eigenvalues))
                or np.any(eigenvalues < 0)
            ):
                raise RuntimeError("multi-index solver returned invalid eigenvalues")

            solver_diagnostics.append(diagnostics)
            result.trace.append(
                {
                    **diagnostics,
                    "k": k,
                    "h": float(h),
                    "alpha": float(alpha),
                    "mean_mass": float(np.mean(statistics.mass)),
                    "basis": basis.copy(),
                    "eigenvalues": eigenvalues.copy(),
                    "distance_initial": _subspace_distance(initial_basis, basis),
                }
            )
            if progress is not None:
                progress(dict(result.trace[-1]))

            with track_stage(tracker, "update"):
                next_h = h / a
                if next_h < h_min:
                    stop_reason = "h_min"
                    result.trace[-1]["stop_reason"] = stop_reason
                    break

                basis_init = basis
                next_alpha = calculate_alpha_k(
                    X,
                    centers,
                    basis,
                    eigenvalues,
                    next_h,
                    N_loc,
                    kernel,
                    distance2=distance2,
                    smart=smart_weights,
                )
                if next_alpha is None:
                    stop_reason = "local_mass_limit"
                    result.trace[-1]["stop_reason"] = stop_reason
                    break

                h = float(next_h)
                alpha = next_alpha
                localization_basis = basis
                localization_eigenvalues = eigenvalues
                k += 1

        result.Set_beta_final(basis)
        result.Set_stop_reason(stop_reason)
        self.result_ = result
        self.basis_ = result.beta_final
        self.beta_ = self.basis_
        self.eigenvalues_ = eigenvalues.copy()
        self.coefficients_ = solver_result.coefficients
        self.trace_ = result.trace
        self.solver_diagnostics_ = solver_diagnostics
        self.effective_parameters_ = {
            "N_loc": N_loc,
            "N_lin": N_lin,
            "N_J": N_J,
            "N_phi": N_phi,
            "h_min": float(h_min),
            "index_dim": m,
            "smart_weights": smart_weights,
        }
        self.n_features_in_ = d
        return self

    def transform(self, X):
        self._check_fitted()
        return utils._prepare_transform(X, self.n_features_in_) @ self.basis_

    def _check_fitted(self) -> None:
        if not hasattr(self, "basis_"):
            raise RuntimeError("model is not fitted")


def _orthonormal_basis(value, shape):
    basis = np.asarray(value, dtype=float)
    if basis.shape != shape or not np.all(np.isfinite(basis)):
        raise RuntimeError(f"multi-index solver must return shape {shape}")
    basis, triangular = np.linalg.qr(basis, mode="reduced")
    if np.min(np.abs(np.diag(triangular))) <= np.finfo(float).eps:
        raise RuntimeError("multi-index solver returned a rank-deficient basis")
    columns = np.arange(basis.shape[1])
    signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), columns])
    basis *= np.where(signs == 0, 1.0, signs)
    return basis


def _projector(value, name, expected_shape=None):
    basis = np.asarray(value, dtype=float)
    if basis.ndim != 2 or 0 in basis.shape:
        raise ValueError(f"{name} must have non-empty shape (d, m)")
    if expected_shape is not None and basis.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if not np.all(np.isfinite(basis)):
        raise ValueError(f"{name} must contain only finite values")

    left, singular_values, _ = np.linalg.svd(basis, full_matrices=False)
    threshold = np.finfo(float).eps * max(basis.shape) * singular_values[0]
    if singular_values[-1] <= threshold:
        raise ValueError(f"{name} must have full column rank")
    return left @ left.T, basis.shape


def _subspace_distance(left, right):
    left_projector, shape = _projector(left, "left")
    right_projector, _ = _projector(right, "right", shape)
    return float(
        np.linalg.norm(left_projector - right_projector, ord="fro")
        / sqrt(2.0 * shape[1])
    )
