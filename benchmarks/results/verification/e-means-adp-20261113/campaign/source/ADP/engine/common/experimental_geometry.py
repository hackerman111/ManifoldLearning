"""P1: точный фильтр носителя через низкоразмерный KD-tree."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree

from ...core.ADP_Config import epanechnikov
from .calculus import _compact_alpha
from .utils import _prepare_multi_localization


@dataclass(frozen=True, slots=True)
class ProjectedComponents:
    """Кандидаты при alpha=0; память O(E0), без ambient pair tensor."""

    indptr: np.ndarray
    indices: np.ndarray
    principal: np.ndarray
    residual: np.ndarray
    shape: tuple[int, int]
    h: float
    projected_X: np.ndarray

    def weights(self, alpha: float) -> csr_matrix:
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must lie in [0, 1]")
        values = epanechnikov((alpha**2 * self.residual + self.principal) / self.h**2)
        weights = csr_matrix(
            (values, self.indices.copy(), self.indptr.copy()), shape=self.shape
        )
        weights.eliminate_zeros()
        return weights

    def alpha(self, target: int) -> float | None:
        if not np.isfinite(target) or target <= 0:
            raise ValueError("target must be positive")
        return _compact_alpha(
            [(self.residual, self.principal)], 1.0 / self.h**2, target * self.shape[0]
        )


def projected_components(
    X: np.ndarray,
    centers: np.ndarray,
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    h: float,
    distance2: np.ndarray,
    *,
    tensor: str = "orthogonal",
) -> ProjectedComponents:
    """p<h² необходимо: p>=0, residual>=0 в обеих поддержанных геометриях.

    Даже при нулевых eigenvalues фильтр безопасен, но может вернуть все пары.
    Query расширен на roundoff; окончательный вес вычисляется по исходной p.
    """
    X, centers, basis, eigenvalues = _prepare_multi_localization(
        X, centers, basis, eigenvalues, h, 1.0, epanechnikov
    )
    if tensor not in {"orthogonal", "full"}:
        raise ValueError("tensor must be orthogonal or full")
    if distance2.shape != (len(centers), len(X)):
        raise ValueError("distance2 shape must be (J, n)")
    Z = (X - X[0]) @ basis
    C = (centers - X[0]) @ basis
    scaled_Z, scaled_C = Z * np.sqrt(eigenvalues), C * np.sqrt(eigenvalues)
    tree = KDTree(scaled_Z)
    scale = max(
        float(np.max(np.abs(scaled_Z))), float(np.max(np.abs(scaled_C))), h, 1.0
    )
    radius = h + 64 * np.finfo(float).eps * scale * np.sqrt(basis.shape[1])
    indices, principals, residuals = [], [], []
    indptr = [0]
    for j, point in enumerate(scaled_C):
        idx = np.asarray(
            tree.query_ball_point(point, radius, return_sorted=True), dtype=np.intp
        )
        principal = np.zeros(len(idx))
        residual = distance2[j, idx].copy()
        for q, value in enumerate(eigenvalues):
            component = np.square(C[j, q] - Z[idx, q])
            principal += value * component
            if tensor == "orthogonal":
                residual -= component
        np.maximum(residual, 0.0, out=residual)
        active = principal / h**2 < 1.0
        indices.append(idx[active])
        principals.append(principal[active])
        residuals.append(residual[active])
        indptr.append(indptr[-1] + int(active.sum()))
    return ProjectedComponents(
        np.asarray(indptr, dtype=np.intp),
        np.concatenate(indices),
        np.concatenate(principals),
        np.concatenate(residuals),
        (len(centers), len(X)),
        h,
        Z,
    )
