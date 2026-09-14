from __future__ import annotations

import numpy as np
import pytest

from ADP.core.ADP_Config import epanechnikov
from ADP.engine.calculus import search_bandwidth
from ADP.engine.initialize import (
    _principal_gradient_basis_with_spectrum,
    initialize_basis_local_with_spectrum,
)


@pytest.mark.parametrize("ridge", [0.0, 1e-8, 1e3])
@pytest.mark.parametrize("offset", [0.0, 1e9])
def test_support_initialization_matches_dense(ridge: float, offset: float) -> None:
    """Нулевые строки удаляются без изменения weighted LS и gradient PCA."""
    rng = np.random.default_rng(218)
    n, d, J, m = 80, 5, 9, 2
    X = rng.normal(size=(n, d))
    X[:, -1] = X[:, 0] + 1e-7 * X[:, -1]
    Y = np.sin(X[:, 0]) + X[:, 1] ** 2
    X += offset
    centers = X[:J].copy()
    distance2 = np.square(X[None, :, :] - centers[:, None, :]).sum(axis=2)
    h = search_bandwidth(distance2, 12, epanechnikov, lower=np.finfo(float).eps)
    weights = epanechnikov(distance2 / h**2)
    assert np.count_nonzero(weights) < weights.size
    ridge_rows = np.zeros((d, d + 1))
    ridge_rows[:, 1:] = np.sqrt(ridge) * np.eye(d)
    gradients = np.empty((J, d))
    for j in range(J):
        design = np.column_stack((np.ones(n), X - centers[j]))
        root = np.sqrt(weights[j])
        augmented = np.vstack((root[:, None] * design, ridge_rows))
        target = np.concatenate((root * Y, np.zeros(d)))
        gradients[j] = np.linalg.lstsq(augmented, target, rcond=None)[0][1:]
    expected, spectrum = _principal_gradient_basis_with_spectrum(
        gradients, m, mass=weights.sum(axis=1)
    )
    actual, actual_spectrum = initialize_basis_local_with_spectrum(
        X, Y, centers, distance2, 12, epanechnikov, ridge, m, mass_weighted=True
    )
    # При ridge=0 condition number порядка 1e8: сравниваем подпространства.
    np.testing.assert_allclose(actual @ actual.T, expected @ expected.T, atol=1e-7)
    np.testing.assert_allclose(actual_spectrum, spectrum, rtol=1e-7, atol=1e-10)


def test_support_keeps_original_rank_cutoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Screening не подменяет rcond=None порогом меньшей матрицы."""
    original = np.linalg.lstsq
    shapes = []

    def record(a, b, rcond=None):
        shapes.append(a.shape)
        assert rcond == np.finfo(float).eps * 84
        return original(a, b, rcond=rcond)

    monkeypatch.setattr(np.linalg, "lstsq", record)
    rng = np.random.default_rng(712)
    X = rng.normal(size=(80, 4))
    Y = X[:, 0] + X[:, 1] ** 2
    centers = X[:8]
    distance2 = np.square(X[None] - centers[:, None]).sum(axis=2)
    initialize_basis_local_with_spectrum(
        X, Y, centers, distance2, 10, epanechnikov, 1e-8, 2
    )
    assert len(shapes) == len(centers)
    assert all(rows < 84 for rows, _ in shapes)
