from dataclasses import replace

import numpy as np
import pytest

from ADP import ADP_Config, ADP_multi_index
from ADP.core.ADP_Config import epanechnikov
from ADP.engine.common.calculus import calculate_alpha_k, pairwise_distance2
from ADP.engine.common.statistic import calculate_statistics
from ADP.engine.common.weights import calculate_multi_weight


def experimental(**kwargs):
    import ADP

    assert hasattr(ADP, "AdaptiveCentersConfig"), "experimental config is missing"
    return ADP.AdaptiveCentersConfig(**kwargs)


def sample(offset=0.0):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(70, 6)) + offset
    C = X[:15].copy()
    B = np.linalg.qr(rng.normal(size=(6, 2)))[0]
    return X, C, B, np.array([1.0, 0.0]), rng


@pytest.mark.parametrize("tensor", ["orthogonal", "full"])
@pytest.mark.parametrize("alpha", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("offset", [0.0, 1e8])
def test_projected_search_preserves_kernel_and_alpha(tensor, alpha, offset):
    experimental()
    from ADP.engine.common.experimental_geometry import projected_components

    X, C, B, lam, _ = sample(offset)
    D = pairwise_distance2(X, C)
    geometry = projected_components(X, C, B, lam, 2.0, D, tensor=tensor)
    actual = geometry.weights(alpha).toarray()
    expected = np.concatenate(
        [
            W
            for _, W in calculate_multi_weight(
                X, C, B, lam, 2.0, alpha, epanechnikov, distance2=D, tensor=tensor
            )
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=3e-14)
    assert np.array_equal(actual > 0, expected > 0)
    target = 3
    old_alpha = calculate_alpha_k(
        X, C, B, lam, 2.0, target, epanechnikov, distance2=D, tensor=tensor
    )
    assert geometry.alpha(target) == old_alpha


def test_sparse_statistics_and_cheap_moments_against_direct_reference():
    experimental()
    from ADP.engine.common.experimental_centers import projected_statistics
    from ADP.engine.common.experimental_geometry import projected_components

    X, C, B, lam, rng = sample(1e8)
    Y = rng.normal(size=len(X))
    W = projected_components(X, C, B, lam, 3.0, pairwise_distance2(X, C)).weights(0.2)
    Phi = rng.normal(size=(len(C), 4, X.shape[1]))
    for normalized in [False, True]:
        dense = calculate_statistics(X, Y, W.toarray(), Phi, normalized=normalized)
        sparse = calculate_statistics(X, Y, W, Phi, normalized=normalized)
        for name in ["I", "U", "mass", "mean", "n_eff", "eta", "S"]:
            np.testing.assert_allclose(
                getattr(sparse, name), getattr(dense, name), rtol=2e-12, atol=2e-12
            )
    Z = (X - X[0]) @ B
    moments = projected_statistics(Z, Y, W, ridge=1e-6)
    for j in range(len(C)):
        w = W.getrow(j).toarray()[0]
        a = w / w.sum()
        centered = Z - a @ Z
        sigma = (centered.T * a) @ centered
        g = centered.T @ (a * (Y - a @ Y))
        np.testing.assert_allclose(moments.covariance[j], sigma, atol=1e-14)
        np.testing.assert_allclose(moments.cross[j], g, atol=1e-14)
        np.testing.assert_allclose(
            moments.slopes[j], np.linalg.solve(sigma + 1e-6 * np.eye(2), g)
        )


def test_configuration_validation():
    for kwargs in [
        {"lambda_floor": -1},
        {"lambda_floor": np.nan},
        {"lambda_floor": 1.1},
        {"budget": 0},
        {"budget": True},
        {"selection": "wrong"},
        {"logdet_ridge": 0},
        {"coverage_weight": -1},
        {"score_ridge": -1},
        {"projection_search": 1},
    ]:
        with pytest.raises((ValueError, TypeError)):
            experimental(**kwargs)


def test_floor_integration_and_opt_out_identity():
    exp = experimental()
    X, _, B, _, rng = sample()
    Y = (X @ B[:, 0]) ** 2 + np.sin(2.5 * X @ B[:, 1]) + 0.01 * rng.normal(size=len(X))
    config = ADP_Config(
        N_J=30, N_phi=4, N_loc=8, outer_steps=3, select_step="last", seed=132
    )
    baseline = ADP_multi_index(2, config).fit(X, Y)
    disabled = ADP_multi_index(2, replace(config, experimental_centers=exp)).fit(X, Y)
    np.testing.assert_array_equal(baseline.basis_, disabled.basis_)
    floor = ADP_multi_index(
        2, replace(config, experimental_centers=experimental(lambda_floor=0.25))
    ).fit(X, Y)
    search = ADP_multi_index(
        2,
        replace(
            config,
            experimental_centers=experimental(
                lambda_floor=0.25, projection_search=True
            ),
        ),
    ).fit(X, Y)
    np.testing.assert_allclose(
        floor.basis_ @ floor.basis_.T, search.basis_ @ search.basis_.T, atol=2e-8
    )
    assert len(search.trace_) == len(floor.trace_)
    for row in search.trace_:
        assert min(row["experimental_centers"]["localization_eigenvalues"]) >= 0.25
    for bad in [
        replace(config, gpu=True),
        replace(config, estimator="legacy"),
        replace(config, kernel=lambda q: np.exp(-q)),
    ]:
        with pytest.raises((ValueError, NotImplementedError)):
            ADP_multi_index(
                2,
                replace(bad, experimental_centers=experimental(projection_search=True)),
            ).fit(X, Y)


def test_greedy_matches_exhaustive_marginals_and_pou_mass():
    experimental()
    from scipy.sparse import csr_matrix

    from ADP.engine.common.experimental_centers import greedy_select, quadrature_mass

    W = csr_matrix([[1.0, 0.5, 0, 0], [0.7, 0.2, 0, 0], [0, 0, 1, 0.4], [0.3, 0, 0, 1]])
    vectors = np.array([[1.0, 0], [1.0, 0.1], [0, 1.0], [0.2, 0.5]])
    selected, diagnostic = greedy_select(
        W, vectors, np.arange(4), budget=3, minimum=1, coverage_weight=1.0, ridge=0.3
    )
    pool = np.asarray(W.sum(axis=0)).ravel()
    all_log = np.linalg.slogdet(np.eye(2) + vectors.T @ vectors / 0.3)[1]
    chosen = []

    def value(S):
        V = vectors[S]
        covered = np.asarray(W[S].sum(axis=0)).ravel() > 0
        return (
            np.linalg.slogdet(np.eye(2) + V.T @ V / 0.3)[1] / all_log
            + pool[covered].sum() / pool.sum()
        )

    for j in selected:
        remaining = [i for i in range(4) if i not in chosen]
        expected = max(remaining, key=lambda i: value([*chosen, i]) - value(chosen))
        assert j == expected
        chosen.append(int(j))
    masses = quadrature_mass(W, selected)
    covered = np.asarray(W[selected].sum(axis=0)).ravel() > 0
    assert masses.sum() == pytest.approx(pool[covered].sum())
    assert diagnostic["mass_coverage"] == pytest.approx(
        pool[covered].sum() / pool.sum()
    )
    assert quadrature_mass(csr_matrix([[1.0], [0.5]]), np.array([0]))[0] == 1.5


@pytest.mark.parametrize("selection", ["random", "score", "logdet", "logdet_coverage"])
def test_selected_centers_keep_full_pool_step_error_and_streams(selection):
    experimental()
    X, _, B, _, _ = sample()
    Y = (X @ B[:, 0]) ** 2 + np.sin(X @ B[:, 1])
    config = ADP_Config(N_J=30, N_phi=4, N_loc=8, outer_steps=2, seed=72)
    full = ADP_multi_index(2, config).fit(X, Y)
    model = ADP_multi_index(
        2,
        replace(
            config,
            experimental_centers=experimental(
                lambda_floor=0.25,
                projection_search=True,
                selection=selection,
                budget=12,
                quadrature=True,
            ),
        ),
    ).fit(X, Y)
    np.testing.assert_array_equal(full.result_.beta_init, model.result_.beta_init)
    for row in model.trace_:
        diag = row["experimental_centers"]
        assert diag["selected_centers"] in (12, 30)
        assert 0 < diag["mass_coverage"] <= 1 + 1e-14
        assert diag["step_error_centers"] == 30
    assert (
        len(model.coefficients_)
        == model.trace_[model.effective_parameters_["selected_iteration"]][
            "experimental_centers"
        ]["selected_centers"]
    )


def test_boundary_support_and_zero_spectrum():
    from ADP.engine.common.experimental_geometry import projected_components

    X = np.array(
        [[0.0, 0.0], [1.0, 0.0], [np.nextafter(1.0, 0.0), 0.0], [0.0, 1.0], [0.0, 2.0]]
    )
    C, B = X[:1], np.array([[1.0], [0.0]])
    D = pairwise_distance2(X, C)
    for eigenvalue in [0.0, 1.0]:
        for alpha in [0.0, 1.0]:
            weights = (
                projected_components(X, C, B, np.array([eigenvalue]), 1.0, D)
                .weights(alpha)
                .toarray()
            )
            expected = epanechnikov(eigenvalue * X[:, 0] ** 2 + alpha**2 * X[:, 1] ** 2)
            np.testing.assert_array_equal(weights[0], expected)


def test_degenerate_surrogate_and_dynamic_greedy_are_explicit():
    from scipy.sparse import csr_matrix

    from ADP.engine.common.experimental_centers import (
        greedy_select,
        projected_statistics,
    )

    W = csr_matrix(np.eye(4))
    Z = np.zeros((4, 2))
    moments = projected_statistics(Z, np.arange(4.0), W, ridge=1e-6)
    assert not moments.feasible.any()
    assert np.isfinite(moments.slopes).all()
    selected, diag = greedy_select(
        W, Z, np.arange(4), budget=None, minimum=2, coverage_weight=1.0, ridge=1.0
    )
    assert len(selected) == 4
    assert diag["thresholds_met"]
    assert diag["information_fraction"] == 1.0


def test_sparse_statistics_reject_invalid_weights_without_mutating_input():
    from scipy.sparse import csr_matrix

    X, C, _, _, rng = sample()
    Phi = rng.normal(size=(len(C), 3, X.shape[1]))
    W = csr_matrix(np.ones((len(C), len(X))))
    before = W.copy()
    calculate_statistics(X, rng.normal(size=len(X)), W, Phi)
    np.testing.assert_array_equal(W.data, before.data)
    for invalid in [np.nan, -1.0]:
        W.data[0] = invalid
        with pytest.raises(ValueError):
            calculate_statistics(X, np.ones(len(X)), W, Phi)


def test_projected_statistics_without_search_and_too_small_budget():
    X, _, B, _, _ = sample()
    Y = (X @ B[:, 0]) ** 2 + np.sin(X @ B[:, 1])
    cfg = ADP_Config(N_J=15, N_phi=4, N_loc=8, outer_steps=1, seed=72)
    baseline = ADP_multi_index(2, cfg).fit(X, Y)
    candidate = ADP_multi_index(
        2, replace(cfg, experimental_centers=experimental(projected_statistics=True))
    ).fit(X, Y)
    np.testing.assert_allclose(
        baseline.basis_ @ baseline.basis_.T,
        candidate.basis_ @ candidate.basis_.T,
        atol=1e-9,
    )
    with pytest.raises(ValueError, match="budget"):
        ADP_multi_index(
            2,
            replace(
                cfg, experimental_centers=experimental(selection="logdet", budget=3)
            ),
        ).fit(X, Y)


@pytest.mark.parametrize("enabled", [False, True])
def test_none_seed_remains_supported(enabled):
    X, _, B, _, _ = sample()
    Y = (X @ B[:, 0]) ** 2 + np.sin(X @ B[:, 1])
    cfg = ADP_Config(
        seed=None,
        N_J=15,
        N_phi=4,
        N_loc=8,
        outer_steps=1,
        experimental_centers=experimental(lambda_floor=0.25) if enabled else None,
    )
    model = ADP_multi_index(2, cfg).fit(X, Y)
    assert np.isfinite(model.basis_).all()


def test_selection_entropy_matches_original_integer_seed_stream():
    for seed in (42, 92799, 92800, 92801, 92802, 93000):
        center_seed = np.random.SeedSequence(seed).spawn(3)[0]
        original = np.random.default_rng(np.random.SeedSequence([seed, 92817]))
        derived = np.random.default_rng(
            np.random.SeedSequence([center_seed.entropy, 92817])
        )
        np.testing.assert_array_equal(
            original.integers(0, 500, 1000), derived.integers(0, 500, 1000)
        )
