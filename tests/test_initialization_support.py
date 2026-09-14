from __future__ import annotations

import numpy as np
import pytest

from ADP.core.ADP_Config import epanechnikov
from ADP.engine.calculus import search_bandwidth
from ADP.engine.initialize import (
    _centered_ridge_gradient,
    _principal_gradient_basis_with_spectrum,
    _select_loo_ridge,
    initialize_basis_local_with_spectrum,
)


@pytest.mark.parametrize("ridge", [0.0, 1e-8, 1e3])
@pytest.mark.parametrize("offset", [0.0, 1e9])
@pytest.mark.parametrize("d", [5, 40])
def test_support_initialization_matches_dense(
    ridge: float, offset: float, d: int
) -> None:
    """Нулевые строки удаляются без изменения weighted LS и gradient PCA."""
    rng = np.random.default_rng(218)
    n, J, m = 80, 9, 2
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
        X, Y, centers, distance2, 10, epanechnikov, 0.0, 2
    )
    assert len(shapes) == len(centers)
    assert all(rows < 84 for rows, _ in shapes)


def test_ridge_loo_matches_explicit_deleted_fits() -> None:
    """PRESS учитывает свободный intercept и неравные локальные веса."""
    rng = np.random.default_rng(71)
    X = rng.normal(size=(16, 4))
    Y = X[:, 0] + 0.8 * rng.normal(size=16)
    w = rng.uniform(0.05, 1.0, size=16)
    root = np.sqrt(w)
    mean = (w / w.sum()) @ X
    yc = (Y - (w / w.sum()) @ Y) * root
    left, values, _ = np.linalg.svd((X - mean) * root[:, None], full_matrices=False)
    minimum = 1e-8
    candidates = np.unique(
        np.append(np.maximum(minimum, values[0] ** 2 * np.logspace(-6, 2, 9)), minimum)
    )
    scores = []
    for ridge in candidates:
        score = 0.0
        penalty = np.column_stack((np.zeros(4), np.sqrt(ridge) * np.eye(4)))
        for i in range(len(X)):
            keep = np.arange(len(X)) != i
            design = np.column_stack((np.ones(keep.sum()), X[keep]))
            fit = np.linalg.lstsq(
                np.vstack((root[keep, None] * design, penalty)),
                np.concatenate((root[keep] * Y[keep], np.zeros(4))),
                rcond=None,
            )[0]
            score += w[i] * (Y[i] - fit[0] - X[i] @ fit[1:]) ** 2
        scores.append(score)
    chosen = _select_loo_ridge(left, values, left.T @ yc, yc, root, minimum)
    assert chosen == candidates[np.argmin(scores)]
    actual = _centered_ridge_gradient(X, Y, root, minimum, 1e-14, ridge_selection="loo")
    expected = _centered_ridge_gradient(X, Y, root, chosen, 1e-14)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_centered_ridge_rank_guard_and_response_offset() -> None:
    rng = np.random.default_rng(132)
    X = rng.normal(size=(30, 5))
    Y = np.round(rng.normal(size=30) * 8) / 8
    root = np.ones(30)
    expected = _centered_ridge_gradient(X, Y, root, 1e-8, 1e-14)
    shifted = _centered_ridge_gradient(X, Y + 1e12, root, 1e-8, 1e-14)
    np.testing.assert_allclose(shifted, expected, atol=1e-12)
    assert _centered_ridge_gradient(X, Y, root, 1e-40, 1e-14) is None
    assert _centered_ridge_gradient(X + 1e12, Y, root, 1e-8, 1e-14) is None


def test_ridge_loo_rejects_undefined_deletion() -> None:
    with pytest.raises(RuntimeError, match="no local ridge candidate"):
        _select_loo_ridge(
            np.eye(2),
            np.ones(2),
            np.ones(2),
            np.ones(2),
            np.array([1.0, 1e-20]),
            1e-8,
        )


def test_local_cv_cli_is_reproducible() -> None:
    from ADP.cli.main import _run, build_parser

    args = build_parser().parse_args(
        [
            "--mode",
            "multi",
            "--n",
            "100",
            "--d",
            "5",
            "--index-dim",
            "2",
            "--N_loc",
            "10",
            "--N_J",
            "12",
            "--N_lin",
            "16",
            "--N_phi",
            "5",
            "--outer-steps",
            "2",
            "--solver-max-steps",
            "2",
            "--index-init",
            "local-cv",
        ]
    )
    first, _, _, metadata = _run(args)
    second, _, _, _ = _run(args)
    np.testing.assert_allclose(first @ first.T, np.eye(2), atol=1e-12)
    np.testing.assert_array_equal(first, second)
    assert metadata["outer_iterations"] == 2
