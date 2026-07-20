from __future__ import annotations

import math

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from ...backends.numpy_backend import NumpyBackend
from ...common.utils import normalize_rows, unit_vector
from .types import RunOutcome, SingleIndexJob


def run_correctness(job: SingleIndexJob) -> RunOutcome:
    handlers = {
        "C01": _c01_direct_statistics,
        "C02": _c02_constant_shift,
        "C03": _c03_exact_linear,
        "C04": _c04_rotation,
        "C05": _c05_feature_scale,
        "C06": _c06_response_affine,
        "C07": _c07_mass_monotonicity,
        "C08": _c08_exact_als,
        "C09": _c09_cg_dense_reference,
        "C10": _c10_als_objective,
        "C11": _c11_objective_scale,
        "C12": _c12_chunk_equivalence,
    }
    try:
        handler = handlers[job.scenario.scenario_id]
    except KeyError as exc:
        raise ValueError(
            f"no correctness executor for {job.scenario.scenario_id}"
        ) from exc
    primary_error, threshold, extra = handler(job.seeds.data)
    passed = bool(np.isfinite(primary_error) and primary_error <= threshold)
    return RunOutcome(
        metrics={
            "primary_error": float(primary_error),
            "threshold": float(threshold),
            "passed": passed,
            **extra,
        },
        iterations=(),
        solver_iterations=(),
        stop_reason="passed" if passed else "threshold_failed",
    )


def _c01_direct_statistics(seed: int):
    actual, reference = _statistics_case(seed)
    errors = [
        _relative_error(actual[0], reference[0]),
        float(np.max(np.abs(actual[1] - reference[1]))),
        _relative_error(actual[2], reference[2]),
        _relative_error(actual[3], reference[3]),
    ]
    return max(errors), 1e-12, {"component_errors": "|".join(map(str, errors))}


def _c02_constant_shift(seed: int):
    rng = np.random.default_rng(seed)
    X, y, directions, q = _statistics_inputs(rng)
    backend = NumpyBackend("float64", statistics_workers=1)
    first = backend.random_projection_sums(
        X=X,
        y=y,
        centers=X[: directions.shape[0]],
        directions=directions,
        q=q,
        kernel="epanechnikov",
    )[0]
    shifted = backend.random_projection_sums(
        X=X,
        y=y + 100.0,
        centers=X[: directions.shape[0]],
        directions=directions,
        q=q,
        kernel="epanechnikov",
    )[0]
    return _relative_error(first, shifted), 1e-12, {}


def _c03_exact_linear(seed: int):
    X, y, beta = _linear_case(seed)
    estimated = _centered_ols(X, y)
    return _direction_loss(estimated, beta), 1e-12, {}


def _c04_rotation(seed: int):
    X, y, beta = _linear_case(seed)
    q, _ = np.linalg.qr(np.random.default_rng(seed + 1).normal(size=(X.shape[1], X.shape[1])))
    rotated_X = X @ q
    expected = q.T @ beta
    estimated = _centered_ols(rotated_X, y)
    return _direction_loss(estimated, expected), 1e-12, {}


def _c05_feature_scale(seed: int):
    X, y, beta = _linear_case(seed)
    errors = []
    for scale in (1e-3, 0.1, 10.0, 1e3):
        estimated = _centered_ols(scale * X, y)
        errors.append(_direction_loss(estimated, beta))
    return max(errors), 1e-10, {}


def _c06_response_affine(seed: int):
    X, y, beta = _linear_case(seed)
    errors = []
    for offset, scale in ((-10.0, 0.01), (0.0, 10.0), (10.0, 100.0)):
        estimated = _centered_ols(X, offset + scale * y)
        errors.append(_direction_loss(estimated, beta))
    return max(errors), 1e-10, {}


def _c07_mass_monotonicity(seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 5))
    centers = X[:20]
    beta = unit_vector(rng.normal(size=5))
    backend = NumpyBackend("float64", statistics_workers=1)
    norm2 = backend.pairwise_norm2(X, centers)
    projection2 = backend.pairwise_projection2(X, centers, beta)
    mass_h1 = backend.kernel(norm2 / 1.0**2, "epanechnikov").sum(axis=1)
    mass_h2 = backend.kernel(norm2 / 2.0**2, "epanechnikov").sum(axis=1)
    q_rho_low = (0.2**2 * norm2 + projection2) / 2.0**2
    q_rho_high = (0.8**2 * norm2 + projection2) / 2.0**2
    mass_rho_low = backend.kernel(q_rho_low, "epanechnikov").sum(axis=1)
    mass_rho_high = backend.kernel(q_rho_high, "epanechnikov").sum(axis=1)
    violation = max(
        float(np.max(np.maximum(mass_h1 - mass_h2, 0.0))),
        float(np.max(np.maximum(mass_rho_high - mass_rho_low, 0.0))),
    )
    return violation, 1e-12, {}


def _c08_exact_als(seed: int):
    beta, expected, objectives = _als_case(seed, response_scale=1.0)
    return _direction_loss(beta, expected), 1e-10, {"iterations": len(objectives)}


def _c09_cg_dense_reference(seed: int):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    eigenvalues = np.geomspace(1.0, 1e3, 20)
    matrix = q @ np.diag(eigenvalues) @ q.T
    rhs = rng.normal(size=20)
    dense = np.linalg.solve(matrix, rhs)
    operator = LinearOperator(matrix.shape, matvec=lambda vector: matrix @ vector)
    iterative, info = cg(operator, rhs, rtol=1e-12, atol=0.0, maxiter=500)
    error = _relative_error(iterative, dense)
    return error, 1e-9, {"cg_info": int(info)}


def _c10_als_objective(seed: int):
    _, _, objectives = _als_case(seed, response_scale=1.0)
    increases = np.maximum(np.diff(objectives), 0.0)
    error = float(np.max(increases)) if increases.size else 0.0
    return error, 1e-10, {"iterations": len(objectives)}


def _c11_objective_scale(seed: int):
    beta_small, _, _ = _als_case(seed, response_scale=1e-4)
    beta_large, _, _ = _als_case(seed, response_scale=1e4)
    return _direction_loss(beta_small, beta_large), 1e-8, {}


def _c12_chunk_equivalence(seed: int):
    rng = np.random.default_rng(seed)
    X, y, directions, q = _statistics_inputs(rng)
    backend = NumpyBackend("float64", statistics_workers=1)
    centers = X[: directions.shape[0]]
    full = backend.random_projection_sums(
        X=X,
        y=y,
        centers=centers,
        directions=directions,
        q=q,
        kernel="epanechnikov",
    )
    pieces = [
        backend.random_projection_sums(
            X=X,
            y=y,
            centers=centers[start:stop],
            directions=directions[start:stop],
            q=q[start:stop],
            kernel="epanechnikov",
        )
        for start, stop in ((0, 2), (2, directions.shape[0]))
    ]
    chunked = tuple(
        np.concatenate([piece[index] for piece in pieces], axis=0)
        for index in range(4)
    )
    errors = [_relative_error(full[index], chunked[index]) for index in range(4)]
    return max(errors), 1e-12, {}


def _statistics_case(seed: int):
    rng = np.random.default_rng(seed)
    X, y, directions, q = _statistics_inputs(rng)
    centers = X[: directions.shape[0]]
    backend = NumpyBackend("float64", statistics_workers=1)
    actual = backend.random_projection_sums(
        X=X,
        y=y,
        centers=centers,
        directions=directions,
        q=q,
        kernel="epanechnikov",
    )[:4]
    weights = np.maximum(1.0 - q, 0.0)
    imav = np.zeros_like(actual[0])
    s_vec = np.zeros_like(actual[1])
    u_mat = np.zeros_like(actual[2])
    counts = weights.sum(axis=1)
    for center in range(centers.shape[0]):
        differences = X - centers[center]
        projected = differences @ directions[center].T
        imav[center] = (y[:, None] * projected * weights[center, :, None]).sum(axis=0)
        s_vec[center] = (projected * weights[center, :, None]).sum(axis=0)
        for direction in range(directions.shape[1]):
            u_mat[center, direction] = (
                differences
                * projected[:, direction, None]
                * weights[center, :, None]
            ).sum(axis=0)
    return actual, (imav, s_vec, u_mat, counts)


def _statistics_inputs(rng: np.random.Generator):
    X = rng.normal(size=(40, 3))
    y = rng.normal(size=40)
    centers = X[:4]
    directions = normalize_rows(rng.normal(size=(4, 3, 3)))
    norm2 = np.sum((centers[:, None, :] - X[None, :, :]) ** 2, axis=2)
    q = norm2 / 4.0**2
    return X, y, directions, q


def _linear_case(seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(300, 5))
    beta = unit_vector(rng.normal(size=5))
    y = -0.4 + 1.8 * (X @ beta)
    return X, y, beta


def _centered_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X - X.mean(axis=0), y - y.mean(), rcond=None)
    return unit_vector(beta)


def _als_case(seed: int, *, response_scale: float):
    rng = np.random.default_rng(seed)
    centers, directions, dimension = 30, 8, 6
    U = rng.normal(size=(centers, directions, dimension))
    expected = unit_vector(rng.normal(size=dimension))
    slopes_true = rng.normal(size=centers)
    imav = response_scale * slopes_true[:, None] * (U @ expected)
    beta = unit_vector(expected + 0.1 * rng.normal(size=dimension))
    objectives = []
    for _ in range(20):
        projected = U @ beta
        denominator = np.sum(projected * projected, axis=1)
        slopes = np.sum(imav * projected, axis=1) / np.maximum(
            denominator,
            np.finfo(float).tiny,
        )
        flat_u = U.reshape(-1, dimension)
        slope_flat = np.repeat(slopes, directions)
        matrix = flat_u.T @ (slope_flat[:, None] ** 2 * flat_u)
        rhs = flat_u.T @ (slope_flat * imav.reshape(-1))
        raw = np.linalg.solve(matrix + 1e-12 * np.eye(dimension), rhs)
        norm = np.linalg.norm(raw)
        beta = raw / norm
        slopes = slopes * norm
        residual = imav - slopes[:, None] * (U @ beta)
        objectives.append(float(np.sum(residual * residual)))
    return beta, expected, objectives


def _direction_loss(left: np.ndarray, right: np.ndarray) -> float:
    cosine_abs = float(
        np.clip(abs(unit_vector(left) @ unit_vector(right)), 0.0, 1.0)
    )
    return float(1.0 - cosine_abs)


def _relative_error(left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(right)), np.finfo(float).eps)
    return float(np.linalg.norm(np.asarray(left) - np.asarray(right)) / denominator)
