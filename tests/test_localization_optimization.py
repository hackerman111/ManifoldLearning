from __future__ import annotations

import numpy as np
import pytest

from ADP.core.ADP_Config import epanechnikov
from ADP.engine.calculus import (
    calculate_alpha_k,
    pairwise_distance2,
    search_bandwidth,
)
from ADP.engine.weights import calculate_multi_weight


def _reference_kernel(value: np.ndarray) -> np.ndarray:
    # Другой callable намеренно включает общий dense search.
    return epanechnikov(value)


@pytest.mark.parametrize("tensor", ["orthogonal", "full"])
@pytest.mark.parametrize("target", [0.5, 5, 30, 80])
@pytest.mark.parametrize("block_size", [1, 4, 32])
def test_compact_alpha_matches_dense(
    tensor: str, target: float, block_size: int
) -> None:
    rng = np.random.default_rng(107)
    X = rng.normal(size=(70, 8))
    centers = X[:9].copy()
    basis, _ = np.linalg.qr(rng.normal(size=(8, 3)))
    eigenvalues = np.array([1.0, 0.1, 1e-12])
    for h in (0.3, 1.5, 10.0):
        args = (X, centers, basis, eigenvalues, h, target)
        actual = calculate_alpha_k(
            *args, epanechnikov, tensor=tensor, block_size=block_size
        )
        expected = calculate_alpha_k(*args, _reference_kernel, tensor=tensor)
        if expected is None:
            assert actual is None
        else:
            assert actual == pytest.approx(expected, abs=1.5e-8)


@pytest.mark.parametrize("target", [0.1, 1, 10, 49.9])
def test_compact_bandwidth_matches_dense(target: float) -> None:
    rng = np.random.default_rng(217)
    distance2 = rng.lognormal(0.0, 4.0, size=(7, 50))
    distance2[0, :3] = 0
    actual = search_bandwidth(distance2, target, epanechnikov, lower=1e-3)
    expected = search_bandwidth(distance2, target, _reference_kernel, lower=1e-3)
    assert actual == pytest.approx(expected, rel=1e-12)


def test_bandwidth_rejects_invalid_distances_and_impossible_mass() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        search_bandwidth(np.array([[-1.0, 2.0]]), 1, epanechnikov, lower=1)
    with pytest.raises(ValueError, match="cannot exceed n"):
        search_bandwidth(np.ones((2, 3)), 4, epanechnikov, lower=1)


@pytest.mark.parametrize("tensor", ["orthogonal", "full"])
@pytest.mark.parametrize("batch_size", [1, 3, 20])
def test_shifted_multi_weights_match_direct(tensor: str, batch_size: int) -> None:
    rng = np.random.default_rng(841)
    # Двоичные дроби позволяют отделить cancellation от округления входа.
    X = np.round(rng.normal(size=(40, 5)) * 8) / 8 + 1e12
    centers = X[:7] + 0.125
    basis, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    spectrum = np.array([1.0, 0.03])
    delta = X[None] - centers[:, None]
    distance2 = np.square(delta).sum(axis=2)
    np.testing.assert_allclose(pairwise_distance2(X, centers), distance2, atol=1e-13)
    coordinates = delta @ basis
    principal2 = (coordinates**2 * spectrum).sum(axis=2)
    residual2 = distance2.copy()
    if tensor == "orthogonal":
        residual2 -= np.square(coordinates).sum(axis=2)
    expected = epanechnikov((0.2**2 * residual2 + principal2) / 1.7**2)
    for cached in (None, distance2):
        actual = np.vstack(
            [
                block
                for _, block in calculate_multi_weight(
                    X,
                    centers,
                    basis,
                    spectrum,
                    1.7,
                    0.2,
                    epanechnikov,
                    block_size=batch_size,
                    distance2=cached,
                    tensor=tensor,
                )
            ]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
