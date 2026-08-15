import numpy as np

from ADP.ADP_Config import epanechnikov
from ADP.engine.calculus import (
    calculate_multi_weight,
    calculate_weight,
    select_optimal_alpha,
)


def _matrix(blocks):
    return np.vstack([block for _, block in blocks])


def test_smart_weights_preserve_single_and_multi_statistics_support():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(40, 6))
    centers = X[:8]
    distance2 = np.square(centers[:, None] - X).sum(axis=2)

    beta = rng.normal(size=6)
    beta /= np.linalg.norm(beta)
    single = dict(
        X=X,
        centers=centers,
        beta=beta,
        h=2.1,
        rho=0.6,
        kernel=epanechnikov,
        block_size=3,
        distance2=distance2,
    )
    np.testing.assert_allclose(
        _matrix(calculate_weight(**single, smart=True)),
        _matrix(calculate_weight(**single)),
        rtol=0,
        atol=1e-14,
    )

    basis, _ = np.linalg.qr(rng.normal(size=(6, 2)), mode="reduced")
    multi = dict(
        X=X,
        centers=centers,
        basis=basis,
        eigenvalues=np.array([1.0, 0.35]),
        h=2.1,
        alpha=0.6,
        kernel=epanechnikov,
        block_size=3,
        distance2=distance2,
    )
    np.testing.assert_allclose(
        _matrix(calculate_multi_weight(**multi, smart=True)),
        _matrix(calculate_multi_weight(**multi)),
        rtol=0,
        atol=1e-14,
    )

    sparse_X = rng.normal(size=(100, 6))
    sparse_centers = sparse_X[:1]
    sparse_distance2 = np.square(
        sparse_centers[:, None] - sparse_X
    ).sum(axis=2)
    sparse_single = single | {
        "X": sparse_X,
        "centers": sparse_centers,
        "distance2": sparse_distance2,
        "h": 1e-8,
    }
    sparse_multi = multi | {
        "X": sparse_X,
        "centers": sparse_centers,
        "distance2": sparse_distance2,
        "h": 1e-8,
    }
    np.testing.assert_array_equal(
        _matrix(calculate_weight(**sparse_single, smart=True)),
        _matrix(calculate_weight(**sparse_single)),
    )
    np.testing.assert_array_equal(
        _matrix(calculate_multi_weight(**sparse_multi, smart=True)),
        _matrix(calculate_multi_weight(**sparse_multi)),
    )

    shifted_X = np.array([[1e16, 1e16]])
    shifted_centers = np.array([[1e16 + 2, 1e16]])
    cancelled_distance2 = np.zeros((1, 1))
    shifted = dict(
        X=shifted_X,
        centers=shifted_centers,
        beta=np.array([1.0, 0.0]),
        h=np.sqrt(6.0),
        rho=1.0,
        kernel=epanechnikov,
        distance2=cancelled_distance2,
    )
    np.testing.assert_array_equal(
        _matrix(calculate_weight(**shifted, smart=True)),
        _matrix(calculate_weight(**shifted)),
    )


def test_optimal_alpha_is_largest_feasible_epanechnikov_value():
    orthogonal2 = np.array([[0.0, 0.5, 1.0], [0.2, 0.8, 1.4]])
    principal2 = np.array([[0.0, 0.1, 0.2], [0.05, 0.15, 0.25]])
    h = 1.0
    target_mass = 3.0

    alpha = select_optimal_alpha(
        orthogonal2,
        principal2,
        h,
        target_mass,
    )

    def mass(value):
        argument = (value**2 * orthogonal2 + principal2) / h**2
        return float(epanechnikov(argument).sum())

    assert alpha is not None
    np.testing.assert_allclose(mass(alpha), target_mass, atol=1e-12)
    assert mass(min(1.0, alpha + 1e-6)) < target_mass
    assert select_optimal_alpha(
        np.ones((1, 2)),
        np.zeros((1, 2)),
        1.0,
        2.0,
    ) == 0.0
