from collections.abc import Callable, Iterator

import numpy as np

from ADP.engine import utils
from ADP.engine.utils import _prepare_multi_localization


def calculate_weight(
    X: np.ndarray,
    centers: np.ndarray,
    beta: np.ndarray,
    h: float,
    rho: float,
    kernel: Callable,
    block_size: int = 128,
    *,
    distance2: np.ndarray | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    X, centers, beta = utils._prepare_weight_data(
        X, centers, beta, h, rho, kernel, block_size
    )

    if distance2 is not None:
        distance2 = utils._prepare_distance2(distance2, centers, len(X))

    x_proj = X @ beta
    center_proj = centers @ beta
    for start in range(0, len(centers), block_size):
        C = centers[start : start + block_size]
        if distance2 is None:
            x_norm2 = np.einsum("nd,nd->n", X, X)
            c_norm2 = np.einsum("bd,bd->b", C, C)
            distance2_block = c_norm2[:, None] + x_norm2[None, :] - 2.0 * C @ X.T
            np.maximum(distance2_block, 0.0, out=distance2_block)
        else:
            distance2_block = distance2[start : start + block_size]

        projection_diff = (
            center_proj[start : start + block_size, None] - x_proj[None, :]
        )
        argument = (rho**2 * distance2_block + projection_diff**2) / h**2
        yield start, kernel(argument)


def calculate_weight_from_adp(config, data, beta, h, rho):
    return calculate_weight(
        data.X,
        data.x_j,
        beta,
        h,
        rho,
        config.kernel,
        block_size=config.batch_size,
    )


def calculate_multi_weight(
    X: np.ndarray,
    centers: np.ndarray,
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    h: float,
    alpha: float,
    kernel: Callable,
    block_size: int = 128,
    *,
    distance2: np.ndarray | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    X, centers, basis, eigenvalues = _prepare_multi_localization(
        X,
        centers,
        basis,
        eigenvalues,
        h,
        alpha,
        kernel,
    )
    utils._check_batch_size(block_size)
    if distance2 is not None:
        distance2 = utils._prepare_distance2(distance2, centers, len(X))

    for start in range(0, len(centers), block_size):
        C = centers[start : start + block_size]
        if distance2 is None:
            x_norm2 = np.einsum("nd,nd->n", X, X)
            c_norm2 = np.einsum("bd,bd->b", C, C)
            distance2_block = c_norm2[:, None] + x_norm2[None, :] - 2.0 * C @ X.T
            np.maximum(distance2_block, 0.0, out=distance2_block)
        else:
            distance2_block = distance2[start : start + block_size]

        orthogonal2, principal2 = _multi_components(
            X,
            C,
            basis,
            eigenvalues,
            distance2_block,
        )
        argument = (alpha**2 * orthogonal2 + principal2) / h**2
        yield start, kernel(argument)


def _multi_components(X, centers, basis, eigenvalues, distance2):
    orthogonal2 = distance2.copy()
    principal2 = np.zeros_like(distance2)
    for vector, eigenvalue in zip(basis.T, eigenvalues, strict=True):
        difference = (centers @ vector)[:, None] - (X @ vector)[None, :]
        component2 = np.square(difference)
        orthogonal2 -= component2
        principal2 += eigenvalue * component2
    np.maximum(orthogonal2, 0.0, out=orthogonal2)
    return orthogonal2, principal2
