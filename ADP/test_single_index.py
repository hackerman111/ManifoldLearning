import numpy as np

from ADP import (
    ADP_Config,
    ADP_SolverResult,
    ADP_single_index,
    ADP_solver,
)
from ADP.ADP_Config import epanechnikov
from ADP.ADP_Statistic import calculate_statistics
from ADP.calculus import (
    calculate_rho_k,
    calculate_weight,
    generate_proj,
    pairwise_distance2,
    search_bandwidth,
)


def test_single_index_recovers_direction_and_preserves_local_mass():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(240, 3))
    beta_true = np.array([1.0, -0.7, 0.4])
    beta_true /= np.linalg.norm(beta_true)
    Y = np.sin(X @ beta_true) + 0.05 * rng.normal(size=len(X))

    config = ADP_Config(
        seed=9,
        N_loc=10,
        N_lin=12,
        N_J=60,
        N_phi=3,
        lambda_penalty=100.0,
        h_min=0.15,
        batch_size=20,
    )
    model = ADP_single_index(config).fit(X, Y)

    assert model.score_direction(beta_true) > 0.99
    assert model.trace_
    assert model.centers_.shape == (config.N_J, X.shape[1])
    assert all(step["mean_mass"] >= config.N_loc - 1e-10 for step in model.trace_)
    assert model.h_k_ / config.a < config.h_min
    np.testing.assert_allclose(model.projector_, np.outer(model.beta_, model.beta_))
    assert model.transform(X[:2]).shape == (2,)
    assert set(model.profile_["stages"]) == {
        "initialization",
        "directions",
        "statistics",
        "solver",
        "update",
    }
    assert model.profile_["total_time_seconds"] > 0
    assert model.profile_["peak_memory_bytes"] > 0
    assert model.profile_["stage_memory_bytes"] == sum(
        stage["memory_bytes"] for stage in model.profile_["stages"].values()
    )
    np.testing.assert_allclose(
        sum(stage["time_fraction"] for stage in model.profile_["stages"].values()),
        1.0,
    )
    np.testing.assert_allclose(
        sum(
            stage["memory_fraction"]
            for stage in model.profile_["stages"].values()
        ),
        1.0,
    )


def test_solver_is_configured_separately_from_model():
    seen = {}

    def identity_solver(
        statistics,
        initial_index,
        *,
        lambda_penalty,
        local_ridge,
        marker,
    ):
        seen.update(
            marker=marker,
            lambda_penalty=lambda_penalty,
            local_ridge=local_ridge,
            directions=statistics.I.shape,
        )
        return ADP_SolverResult(initial_index.copy(), None, {"status": "identity"})

    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 3))
    Y = X[:, 0]
    solver = ADP_solver(identity_solver, marker=17)
    config = ADP_Config(
        seed=4,
        N_loc=5,
        N_J=20,
        N_phi=2,
        h_min=10.0,
        index_init="random",
    )

    model = ADP_single_index(config, solver=solver).fit(X, Y)

    assert seen == {
        "marker": 17,
        "lambda_penalty": config.lambda_penalty,
        "local_ridge": config.local_ridge,
        "directions": (20, 2),
    }
    assert len(model.trace_) == 1


def test_default_parameters_are_resolved_from_data():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(120, 3))
    beta_true = np.array([1.0, 0.5, -0.2])
    beta_true /= np.linalg.norm(beta_true)
    Y = np.sin(X @ beta_true) + 0.05 * rng.normal(size=len(X))

    model = ADP_single_index(
        ADP_Config(lambda_penalty=100.0),
    ).fit(X, Y)

    assert model.effective_parameters_["N_lin"] == 6
    assert model.effective_parameters_["N_J"] == 120
    assert model.effective_parameters_["N_phi"] == 3
    assert model.stop_reason_ in {"h_min", "local_mass_limit"}
    assert model.score_direction(beta_true) > 0.99


def test_weights_reuse_precomputed_distances():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(30, 4))
    centers = X[:12]
    beta = rng.normal(size=4)
    beta /= np.linalg.norm(beta)
    distance2 = pairwise_distance2(X, centers)

    expected = np.vstack(
        [
            block
            for _, block in calculate_weight(
                X, centers, beta, 1.5, 0.4, epanechnikov, block_size=5
            )
        ]
    )
    actual = np.vstack(
        [
            block
            for _, block in calculate_weight(
                X,
                centers,
                beta,
                1.5,
                0.4,
                epanechnikov,
                block_size=5,
                distance2=distance2,
            )
        ]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_rho_search_stops_at_float_precision():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(80, 3))
    centers = X[:40]
    beta = np.array([1.0, -0.5, 0.25])
    beta /= np.linalg.norm(beta)
    distance2 = pairwise_distance2(X, centers)
    h0 = search_bandwidth(distance2, 5, epanechnikov, lower=1e-12)
    calls = 0

    def counting_kernel(value):
        nonlocal calls
        calls += 1
        return epanechnikov(value)

    rho = calculate_rho_k(
        X,
        centers,
        beta,
        h0 / np.sqrt(2),
        5,
        counting_kernel,
        distance2=distance2,
    )

    assert rho is not None and 0 < rho < 1
    assert calls < 40


def test_local_statistics_match_direct_sums():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(120, 4))
    Y = rng.normal(size=len(X))
    centers = X[:40]
    distance2 = pairwise_distance2(X, centers)
    h = search_bandwidth(distance2, 5, epanechnikov, lower=1e-12)
    weights = np.vstack(
        [
            block
            for _, block in calculate_weight(
                X,
                centers,
                np.zeros(4),
                h,
                1.0,
                epanechnikov,
                block_size=16,
                distance2=distance2,
            )
        ]
    )
    directions = generate_proj(rng, len(centers), 3, np.zeros(4), 1.0)

    assert 4 * np.count_nonzero(weights, axis=1).max() <= len(X)
    result = calculate_statistics(X, Y, weights, directions, batch_size=16)

    mass = weights.sum(axis=1)
    mean = weights @ X / mass[:, None]
    centered = X[None, :, :] - mean[:, None, :]
    projections = np.einsum("jpd,jnd->jpn", directions, centered)
    weighted = weights[:, None, :] * projections

    np.testing.assert_allclose(result.I, weighted @ Y, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        result.U,
        weighted @ centered,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(result.mass, mass)
    np.testing.assert_allclose(result.mean, mean)
