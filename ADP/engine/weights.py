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
    estimator: str = "legacy",
) -> Iterator[tuple[int, np.ndarray]]:
    X, centers, beta = utils._prepare_weight_data(
        X, centers, beta, h, rho, kernel, block_size
    )

    if estimator not in {"new", "legacy"}:
        raise ValueError("estimator must be 'new' or 'legacy'")
    if estimator == "new":
        beta_norm = np.linalg.norm(beta)
        if beta_norm == 0:
            raise ValueError("beta must be non-zero for the new estimator")
        beta = beta / beta_norm
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
        projection2 = np.square(projection_diff)
        if estimator == "new":
            # ESTIMATOR: T^2=h^-2[rho^2(I-bb^T)+bb^T].
            orthogonal2 = distance2_block - projection2
            # NUMERICAL: roundoff must not make an orthogonal square negative.
            np.maximum(orthogonal2, 0.0, out=orthogonal2)
            argument = (projection2 + rho**2 * orthogonal2) / h**2
        else:
            argument = (rho**2 * distance2_block + projection2) / h**2
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
        estimator=config.estimator,
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
    tensor: str = "orthogonal",
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
    if tensor not in {"orthogonal", "full"}:
        raise ValueError("tensor must be 'orthogonal' or 'full'")
    if distance2 is not None:
        distance2 = utils._prepare_distance2(distance2, centers, len(X))

    # NUMERICAL: общий сдвиг до проекций, чтобы не вычитать большие числа.
    origin = X[0]
    # EXACT: проекции и нормы данных не зависят от блока центров.
    projected_X = (X - origin) @ basis
    projected_centers = (centers - origin) @ basis
    if distance2 is None:
        X = X - origin
        centers = centers - origin
    x_norm2 = np.einsum("nd,nd->n", X, X) if distance2 is None else None
    for start in range(0, len(centers), block_size):
        C = centers[start : start + block_size]
        if distance2 is None:
            c_norm2 = np.einsum("bd,bd->b", C, C)
            assert x_norm2 is not None
            distance2_block = c_norm2[:, None] + x_norm2[None, :] - 2.0 * C @ X.T
            np.maximum(distance2_block, 0.0, out=distance2_block)
        else:
            distance2_block = distance2[start : start + block_size]

        orthogonal2, principal2 = _projected_multi_components(
            projected_X,
            projected_centers[start : start + block_size],
            eigenvalues,
            distance2_block,
        )
        residual2 = orthogonal2 if tensor == "orthogonal" else distance2_block
        argument = (alpha**2 * residual2 + principal2) / h**2
        yield start, kernel(argument)


def _multi_components(X, centers, basis, eigenvalues, distance2):
    origin = X[0]
    return _projected_multi_components(
        (X - origin) @ basis,
        (centers - origin) @ basis,
        eigenvalues,
        distance2,
    )


def _projected_multi_components(
    projected_X: np.ndarray,
    projected_centers: np.ndarray,
    eigenvalues: np.ndarray,
    distance2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    orthogonal2 = distance2.copy()
    principal2 = np.zeros_like(distance2)
    for q, eigenvalue in enumerate(eigenvalues):
        difference = projected_centers[:, q, None] - projected_X[None, :, q]
        component2 = np.square(difference)
        orthogonal2 -= component2
        principal2 += eigenvalue * component2
    np.maximum(orthogonal2, 0.0, out=orthogonal2)
    return orthogonal2, principal2
