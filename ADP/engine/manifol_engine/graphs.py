"""Построение manifold-графа и начальных локальных проекторов."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from ...core.manifold import ADP_Manifold_utils as model_utils
from .utils import orient_rows


def build_manifold_graph(
    model: Any,
    centers: np.ndarray,
    projectors: np.ndarray | None,
    eigenvalues: np.ndarray | None,
    h: float,
    alpha: float,
) -> csr_matrix:
    """Построить CSR: row — target ``l``, column — source ``j``."""
    indptr = [0]
    index_chunks: list[np.ndarray] = []
    data_chunks: list[np.ndarray] = []
    count = 0
    for start in range(0, len(centers), model.batch_size):
        weights = model._weight_block(
            centers, centers, projectors, eigenvalues, h, alpha, start
        )
        for row in weights:
            indices = np.flatnonzero(row)
            model_utils.require_graph_row(indices)
            values = row[indices]
            index_chunks.append(indices)
            data_chunks.append(values)
            count += len(indices)
            indptr.append(count)
    indices = np.concatenate(index_chunks).astype(np.intp, copy=False)
    data = np.concatenate(data_chunks)
    return csr_matrix(
        (data, indices, np.asarray(indptr, dtype=np.intp)),
        shape=(len(centers), len(centers)),
    )


def initialize_projectors(
    gradients: np.ndarray,
    gradient_mass: np.ndarray,
    graph: csr_matrix,
    m: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Инициализировать локальные bases через weighted-gradient SVD."""
    J, d = gradients.shape
    projectors = np.empty((J, m, d))
    eigenvalues = np.empty((J, m))
    for l in range(J):
        begin, end = graph.indptr[l : l + 2]
        sources = graph.indices[begin:end]
        weights = graph.data[begin:end]
        values = np.sqrt(gradient_mass[sources] * weights)[:, None] * gradients[sources]
        _, singular_values, right_vectors = np.linalg.svd(values, full_matrices=False)
        model_utils.require_identified(singular_values, values.shape, m, l)
        projectors[l] = orient_rows(right_vectors[:m].copy())
        spectrum = np.square(singular_values[:m])
        eigenvalues[l] = spectrum / spectrum[0]
    return projectors, eigenvalues
