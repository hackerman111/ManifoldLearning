"""Генерация данных и независимые воспроизводимые потоки случайности."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields

import numpy as np

from ADP.cli.experiment_utils import (
    fail as _fail,
    standardize as _standardize,
    unit as _unit,
    validate_multi_link,
    validate_single_link,
)

from .models import ExperimentPoint, LinkName


@dataclass(frozen=True, slots=True)
class _SeedBundle:
    beta: int
    features: int
    noise: int
    centers: int
    directions: int
    init: int
    outliers: int
    outlier_noise: int
    gamma: int
    misspecification: int


@dataclass(frozen=True, slots=True)
class _GeneratedData:
    X: np.ndarray
    Y: np.ndarray
    beta: np.ndarray


def _make_seed_bundle(
    selector: str,
    point: ExperimentPoint,
    seed: int,
    *,
    common_random_fields: tuple[str, ...] = (),
) -> _SeedBundle:
    if selector == "custom":
        return _SeedBundle(*(seed for _ in fields(_SeedBundle)))
    parameters = asdict(point)
    for name in (
        "normalize_link_by_sigma_x",
        "basis_pool_dim",
        *common_random_fields,
    ):
        parameters.pop(name, None)
    payload = json.dumps(
        {
            "experiment": selector,
            "parameters": parameters,
            "seed": seed,
            "seed_design": "paired-within-experiment-v1",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    entropy = int(hashlib.sha256(payload).hexdigest(), 16)
    state = np.random.SeedSequence(entropy).generate_state(10)
    return _SeedBundle(*(int(value) for value in state))


def _generate_data(
    selector: str,
    point: ExperimentPoint,
    seeds: _SeedBundle,
    run_seed: int,
) -> _GeneratedData:
    if selector == "custom":
        rng = np.random.default_rng(run_seed)
        X = rng.normal(size=(point.n, point.d))
        beta = _unit(rng.normal(size=point.d), "beta")
        Y = np.sin(X @ beta) + point.sigma_eps * rng.normal(size=point.n)
        return _GeneratedData(X, Y, beta)

    X = _features(point, seeds.features)
    link_divisor = point.sigma_x if point.normalize_link_by_sigma_x else 1.0
    if point.mode == "single":
        beta = _unit(
            np.random.default_rng(seeds.beta).normal(size=point.d),
            "beta",
        )
        index = X @ beta
        signal_values = _link(
            index / link_divisor,
            point.link,
            scale=point.link_scale,
        )
        noise_index = index
    elif point.mode == "multi":
        basis_pool_dim = point.basis_pool_dim or point.index_dim
        basis_pool, _ = np.linalg.qr(
            np.random.default_rng(seeds.beta).normal(size=(point.d, basis_pool_dim)),
            mode="reduced",
        )
        beta = _orient_columns(basis_pool)[:, : point.index_dim]
        projected = X @ beta
        signal_values = _multi_link(
            projected / link_divisor,
            point.link,
            point.link_scale,
        )
        noise_index = np.linalg.norm(projected, axis=1)
    else:
        radial = X[:, :2]
        radii = np.linalg.norm(radial, axis=1)
        if np.any(radii <= np.finfo(float).eps):
            raise RuntimeError("radial manifold data contain an undefined direction")
        beta = np.zeros((point.n, point.d, 1))
        beta[:, :2, 0] = radial / radii[:, None]
        signal_values = 0.5 * point.link_scale * np.square(radii)
        noise_index = radii
    signal = _standardize(signal_values, f"{point.link} link")
    noise = _noise(point, noise_index, seeds.noise)
    noise = _outliers(point, noise, seeds.outliers, seeds.outlier_noise)
    Y = signal + noise
    if point.delta > 0 and point.mode == "single":
        gamma = _gamma(beta, seeds.gamma, seeds.misspecification)
        gamma_index = X @ gamma
        Y += point.delta * _standardize(
            gamma_index + 0.5 * gamma_index**2,
            "misspecification link",
        )
    return _GeneratedData(np.asarray(X), np.asarray(Y), beta)


def _features(point: ExperimentPoint, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (point.n, point.d)
    if point.tau is not None:
        # ESTIMATOR/data design: точная формула из TeX, без переименования tau в corr.
        common = rng.normal(size=(point.n, 1))
        values = point.tau * common + (1 - point.tau) * rng.normal(size=shape)
    elif point.x_distribution == "gaussian":
        coordinates = np.arange(point.d)
        covariance = point.rho_corr ** np.abs(
            np.subtract.outer(coordinates, coordinates)
        )
        values = rng.normal(size=shape) @ np.linalg.cholesky(covariance).T
    elif point.x_distribution == "uniform":
        values = rng.uniform(-math.sqrt(3), math.sqrt(3), size=shape)
    else:
        values = rng.standard_t(df=5, size=shape) * math.sqrt(3 / 5)
    return np.asarray(point.sigma_x * values, dtype=float)


def _noise(point: ExperimentPoint, index: np.ndarray, seed: int) -> np.ndarray:
    if point.sigma_eps == 0:
        return np.zeros(point.n)
    rng = np.random.default_rng(seed)
    if point.heteroscedastic:
        scale = point.sigma_eps * np.sqrt((0.25 + index**2) / 1.25)
        return np.asarray(scale * rng.normal(size=point.n))
    if point.noise_distribution == "gaussian":
        values = rng.normal(size=point.n)
    elif point.noise_distribution == "student_t5":
        values = rng.standard_t(df=5, size=point.n) * math.sqrt(3 / 5)
    else:
        values = rng.standard_t(df=3, size=point.n) * math.sqrt(1 / 3)
    return np.asarray(point.sigma_eps * values)


def _outliers(
    point: ExperimentPoint,
    noise: np.ndarray,
    index_seed: int,
    noise_seed: int,
) -> np.ndarray:
    if point.outlier_fraction == 0:
        return noise
    count = min(point.n, math.ceil(point.outlier_fraction * point.n))
    indices = np.random.default_rng(index_seed).permutation(point.n)[:count]
    result = noise.copy()
    result[indices] = np.random.default_rng(noise_seed).normal(
        scale=point.outlier_scale * point.sigma_eps,
        size=count,
    )
    return result


def _gamma(beta: np.ndarray, seed: int, orientation_seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    for _ in range(256):
        candidate = rng.normal(size=beta.size)
        candidate -= beta * float(candidate @ beta)
        norm = np.linalg.norm(candidate)
        if np.isfinite(norm) and norm > np.finfo(float).eps:
            sign = (
                -1.0
                if np.random.default_rng(orientation_seed).integers(2) == 0
                else 1.0
            )
            return np.asarray(sign * candidate / norm)
    return _fail("cannot generate a direction orthogonal to beta")


def _link(index: np.ndarray, name: LinkName, *, scale: float = 1.0) -> np.ndarray:
    if name == "linear":
        return index
    if name == "quadratic":
        return index + 0.5 * index**2
    if name == "square":
        return index**2
    if name == "cubic":
        return index**3
    if name == "quartic":
        return index**4
    if name == "sin":
        return np.sin(1.5 * index)
    if name == "tanh":
        return np.tanh(2 * index)
    if name == "sin_scaled":
        return np.sin(scale * index)
    if name == "cos_scaled":
        return np.cos(scale * index)
    if name == "x_sin":
        return index * np.sin(scale * index)
    if name == "tanh_scaled":
        return np.tanh(scale * index)
    if name == "absolute":
        return np.abs(index)
    if name == "relu":
        return np.maximum(index, 0.0)
    if name == "gaussian_bump":
        return np.exp(-0.5 * np.square(scale * index))
    validate_single_link(name)
    return index * np.sin(math.sqrt(5) * index)


def _multi_link(
    projected: np.ndarray,
    name: LinkName,
    scale: float,
) -> np.ndarray:
    validate_multi_link(projected, name)
    if name == "multi_additive":
        values = projected[:, 0] ** 2 + np.sin(scale * projected[:, 1])
    else:
        values = projected[:, 0] * np.sin(scale * projected[:, 1])
    if projected.shape[1] > 2:
        # ESTIMATOR/data design: каждая дополнительная координата участвует явно.
        denominators = np.arange(3, projected.shape[1] + 1)
        values = values + np.sum(projected[:, 2:] ** 2 / denominators, axis=1)
    return np.asarray(values)


def _orient_columns(basis: np.ndarray) -> np.ndarray:
    columns = np.arange(basis.shape[1])
    signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), columns])
    return np.asarray(basis * np.where(signs == 0, 1.0, signs))
