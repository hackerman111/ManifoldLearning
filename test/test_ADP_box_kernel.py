from functools import partial

import numpy as np
import pytest

from ADP import (
    ADP_Config,
    ADP_SolverResult,
    ADP_multi_index,
    ADP_single_index,
    ADP_solver,
)
from ADP.ADP_Statistic import calculate_statistics, calculate_statistics_gpu
from ADP.engine.box_kernel import (
    NeighborhoodEngine,
    SparseStatisticsCache,
    box_kernel,
    initialize_basis_local_sparse,
    make_plateau_kernel,
    plateau_kernel,
    search_sparse_bandwidth,
    search_sparse_scale,
    sparse_kernel_parameters,
)
from ADP.engine.calculus import initialize_basis_local, pairwise_distance2


def test_box_and_plateau_kernel_boundaries():
    q = np.array([0.0, 0.5, 0.75, 1.0, 2.0])
    np.testing.assert_array_equal(
        box_kernel(q),
        [1.0, 1.0, 1.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        plateau_kernel(q, tau=0.5),
        [1.0, 1.0, 0.5, 0.0, 0.0],
        atol=1e-15,
    )


def test_plateau_tau_and_kernel_identity_are_validated():
    kernel = make_plateau_kernel(0.4)
    assert isinstance(kernel, partial)
    assert sparse_kernel_parameters(box_kernel) == ("box", None)
    assert sparse_kernel_parameters(kernel) == ("plateau", 0.4)
    with pytest.raises(ValueError, match="tau"):
        make_plateau_kernel(1.0)
    with pytest.raises(ValueError, match="finite"):
        plateau_kernel(np.array([np.nan]))


def test_plateau_is_c2_at_tau_and_support_boundary():
    tau = 0.5
    step = 1e-5
    for point in (tau, 1.0):
        left, center, right = plateau_kernel(
            np.array([point - step, point, point + step]),
            tau=tau,
        )
        first = (right - left) / (2.0 * step)
        second = (right - 2.0 * center + left) / step**2
        assert abs(first) < 1e-6
        assert abs(second) < 1e-3


def _dense_blocks(blocks, rows, columns):
    result = np.zeros((rows, columns))
    for block in blocks:
        for local_row in range(block.rows):
            indices, weights = block.row(local_row)
            result[block.start + local_row, indices] = weights
    return result


def test_single_neighborhood_matches_dense_box_reference():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(11, 4))
    centers = X[[0, 4, 8]]
    beta = rng.normal(size=4)
    beta /= np.linalg.norm(beta)
    h, rho = 1.6, 0.45

    engine = NeighborhoodEngine(X, centers, block_size=2)
    blocks = list(engine.single_blocks(beta, h, rho, box_kernel))
    actual = _dense_blocks(blocks, len(centers), len(X))

    differences = centers[:, None, :] - X[None, :, :]
    distance2 = np.einsum("jnd,jnd->jn", differences, differences)
    projection2 = np.square(differences @ beta)
    expected = box_kernel((rho**2 * distance2 + projection2) / h**2)

    np.testing.assert_array_equal(actual, expected)
    assert sum(block.edge_count for block in blocks) == np.count_nonzero(expected)
    assert all(block.boundary_count == 0 for block in blocks)


def test_multi_neighborhood_matches_dense_plateau_reference():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(13, 5))
    centers = X[[1, 6, 10]]
    basis, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    eigenvalues = np.array([1.0, 0.35])
    h, alpha, tau = 1.8, 0.4, 0.5
    kernel = make_plateau_kernel(tau)

    engine = NeighborhoodEngine(X, centers, block_size=2)
    blocks = list(
        engine.multi_blocks(basis, eigenvalues, h, alpha, kernel)
    )
    actual = _dense_blocks(blocks, len(centers), len(X))

    differences = centers[:, None, :] - X[None, :, :]
    distance2 = np.einsum("jnd,jnd->jn", differences, differences)
    projected = differences @ basis
    projected2 = np.einsum("jnm,jnm->jn", projected, projected)
    principal2 = np.einsum(
        "jnm,m,jnm->jn",
        projected,
        eigenvalues,
        projected,
    )
    q = (alpha**2 * np.maximum(distance2 - projected2, 0.0) + principal2) / h**2
    expected = plateau_kernel(q, tau=tau)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    assert sum(block.edge_count for block in blocks) == np.count_nonzero(q < 1.0)
    assert sum(block.boundary_count for block in blocks) == np.count_nonzero(
        (q > tau) & (q < 1.0)
    )


def test_multi_screen_uses_smallest_localization_eigenvalue():
    X = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 2.0]])
    centers = X[[0]]
    basis = np.array([[1.0], [0.0]])
    eigenvalues = np.array([0.01])
    engine = NeighborhoodEngine(X, centers, block_size=1)

    block = next(
        engine.multi_blocks(
            basis,
            eigenvalues,
            h=1.0,
            alpha=1.0,
            kernel=box_kernel,
        )
    )

    assert block.indices.tolist() == [0, 1]


def _mean_mass(blocks):
    masses = [block.mass for block in blocks]
    return float(np.concatenate(masses).mean())


def test_sparse_bandwidth_returns_smallest_feasible_box_radius():
    X = np.array([[0.0], [1.0], [3.0]])
    engine = NeighborhoodEngine(X, X[[0]], block_size=1)

    h = search_sparse_bandwidth(engine, 2.0, box_kernel, lower=0.1)

    assert _mean_mass(engine.isotropic_blocks(h, box_kernel)) >= 2.0
    assert _mean_mass(
        engine.isotropic_blocks(np.nextafter(h, 0.0), box_kernel)
    ) < 2.0


@pytest.mark.parametrize(
    "kernel",
    [box_kernel, make_plateau_kernel(0.5)],
    ids=["box", "plateau"],
)
def test_sparse_scale_matches_dense_bisection(kernel):
    rng = np.random.default_rng(8)
    X = rng.normal(size=(24, 3))
    centers = X[[0, 7, 15, 20]]
    beta = rng.normal(size=3)
    beta /= np.linalg.norm(beta)
    h = 1.5
    engine = NeighborhoodEngine(X, centers, block_size=2)

    differences = centers[:, None, :] - X[None, :, :]
    distance2 = np.einsum("jnd,jnd->jn", differences, differences)
    projection2 = np.square(differences @ beta)

    def dense_mass(scale):
        return float(
            kernel((scale**2 * distance2 + projection2) / h**2).sum(axis=1).mean()
        )

    target = (dense_mass(0.0) + dense_mass(1.0)) / 2.0
    assert dense_mass(0.0) > dense_mass(1.0)
    actual = search_sparse_scale(
        lambda scale: engine.single_blocks(
            beta,
            h,
            scale,
            kernel,
            record=False,
        ),
        target,
    )
    low, high = 0.0, 1.0
    while high - low > np.sqrt(np.finfo(float).eps):
        middle = (low + high) / 2.0
        if dense_mass(middle) >= target:
            low = middle
        else:
            high = middle

    np.testing.assert_allclose(actual, low, rtol=0, atol=1e-14)


def test_sparse_local_initialization_matches_dense_reference():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(30, 3))
    true_beta = np.array([0.8, -0.5, 0.3])
    true_beta /= np.linalg.norm(true_beta)
    Y = X @ true_beta + 0.01 * rng.normal(size=len(X))
    centers = X[np.arange(0, len(X), 3)]
    ridge = 1e-6
    engine = NeighborhoodEngine(X, centers, block_size=4)

    actual = initialize_basis_local_sparse(
        X,
        Y,
        engine,
        6,
        box_kernel,
        ridge,
        1,
    )
    expected = initialize_basis_local(
        X,
        Y,
        centers,
        pairwise_distance2(X, centers),
        6,
        box_kernel,
        ridge,
        1,
    )

    np.testing.assert_allclose(actual.T @ actual, np.eye(1), atol=1e-12)
    np.testing.assert_allclose(
        actual @ actual.T,
        expected @ expected.T,
        rtol=1e-9,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "kernel",
    [box_kernel, make_plateau_kernel(0.5)],
    ids=["box", "plateau"],
)
def test_sparse_statistics_match_dense_weights(kernel):
    rng = np.random.default_rng(19)
    n, d, J, P = 25, 4, 5, 3
    X = rng.normal(size=(n, d))
    Y = rng.normal(size=n)
    centers = X[[0, 4, 9, 15, 21]]
    directions = rng.normal(size=(J, P, d))
    beta = rng.normal(size=d)
    beta /= np.linalg.norm(beta)
    engine = NeighborhoodEngine(X, centers, block_size=2)
    blocks = list(engine.single_blocks(beta, 1.7, 0.35, kernel))
    dense = _dense_blocks(blocks, J, n)

    expected = calculate_statistics(X, Y, dense, directions, batch_size=2)
    actual = calculate_statistics(
        X,
        Y,
        iter(blocks),
        directions,
        batch_size=2,
        sparse_cache=SparseStatisticsCache(),
    )

    for name in actual:
        np.testing.assert_allclose(
            actual[name],
            expected[name],
            rtol=1e-11,
            atol=1e-11,
        )


def test_sparse_statistics_cache_reuses_only_exact_weighted_support():
    X = np.array([[0.0], [0.75], [2.0]])
    Y = np.array([0.0, 1.0, -1.0])
    centers = X[[0]]
    directions = np.ones((1, 1, 1))
    kernel = make_plateau_kernel(0.5)
    engine = NeighborhoodEngine(X, centers, block_size=1)
    cache = SparseStatisticsCache()

    first = list(engine.isotropic_blocks(1.0, kernel))
    calculate_statistics(
        X, Y, iter(first), directions, sparse_cache=cache
    )
    calculate_statistics(
        X, Y, iter(first), directions, sparse_cache=cache
    )
    assert cache.last_hits == 1

    changed_weights = list(engine.isotropic_blocks(1.1, kernel))
    assert first[0].indices.tolist() == changed_weights[0].indices.tolist()
    calculate_statistics(
        X,
        Y,
        iter(changed_weights),
        directions,
        sparse_cache=cache,
    )
    assert cache.last_hits == 0


def _cuda_available():
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="CUDA device is unavailable")
def test_sparse_gpu_statistics_match_cpu():
    import cupy as cp

    rng = np.random.default_rng(29)
    X = rng.normal(size=(18, 3))
    Y = rng.normal(size=len(X))
    centers = X[[0, 5, 11, 16]]
    directions = rng.normal(size=(len(centers), 2, X.shape[1]))
    engine = NeighborhoodEngine(X, centers, block_size=2)
    blocks = list(
        engine.isotropic_blocks(1.8, make_plateau_kernel(0.5))
    )

    cpu = calculate_statistics(X, Y, iter(blocks), directions)
    gpu = calculate_statistics_gpu(X, Y, iter(blocks), directions)

    np.testing.assert_allclose(cp.asnumpy(gpu.I), cpu.I, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(cp.asnumpy(gpu.U), cpu.U, rtol=1e-11, atol=1e-11)
    for name in ("mass", "mean", "n_eff", "eta"):
        np.testing.assert_allclose(gpu[name], cpu[name], rtol=1e-11, atol=1e-11)


def test_sparse_kernel_cli_and_config():
    from ADP.cli import build_parser, experiment_from_args, parse_kernel

    assert parse_kernel("box") is box_kernel
    assert parse_kernel("plateau") is plateau_kernel
    parser = build_parser()
    experiment = experiment_from_args(
        parser.parse_args(
            ["--kernel", "plateau", "--kernel-tau", "0.3"]
        ),
        parser,
    )
    kernel = experiment.variants["default"].config.kernel
    assert sparse_kernel_parameters(kernel) == ("plateau", 0.3)
    assert ADP_Config(kernel=box_kernel).kernel is box_kernel
    with pytest.raises(ValueError, match="smart_weights"):
        ADP_Config(kernel=box_kernel, smart_weights=True)


def test_kernel_tau_is_not_silently_ignored():
    from ADP.cli import build_parser, experiment_from_args

    parser = build_parser()
    with pytest.raises(SystemExit):
        experiment_from_args(
            parser.parse_args(
                ["--kernel", "epanechnikov", "--kernel-tau", "0.3"]
            ),
            parser,
        )


def test_plateau_kernel_has_stable_experiment_serialization():
    from ADP.experiment_runner import _spec_value

    assert _spec_value(make_plateau_kernel(0.3)) == {
        "function": "ADP.engine.box_kernel:plateau_kernel",
        "args": [],
        "keywords": {"tau": 0.3},
    }


def _unchanged_index(statistics, index, **_parameters):
    index = np.asarray(index)
    diagnostics = (
        {}
        if index.ndim == 1
        else {"eigenvalues": np.ones(index.shape[1])}
    )
    coefficients = np.ones(
        len(statistics.I)
        if index.ndim == 1
        else (len(statistics.I), index.shape[1])
    )
    return ADP_SolverResult(index.copy(), coefficients, diagnostics)


def _forbid_dense_path(*_args, **_kwargs):
    raise AssertionError("dense J x n path was called")


def test_single_fit_uses_sparse_backend_without_dense_path(monkeypatch):
    import importlib

    module = importlib.import_module("ADP.single_index.ADP_single_index")
    monkeypatch.setattr(module, "pairwise_distance2", _forbid_dense_path)
    monkeypatch.setattr(module, "calculate_weight", _forbid_dense_path)
    rng = np.random.default_rng(31)
    X = rng.normal(size=(30, 3))
    Y = X[:, 0] - 0.3 * X[:, 1]
    config = ADP_Config(
        seed=2,
        N_loc=4,
        N_lin=6,
        N_J=8,
        N_phi=2,
        outer_steps=1,
        h_min=1e6,
        kernel=box_kernel,
        index_init="local",
    )

    model = ADP_single_index(config, ADP_solver(_unchanged_index)).fit(X, Y)

    np.testing.assert_allclose(np.linalg.norm(model.beta_), 1.0, atol=1e-12)
    assert model.effective_parameters_["kernel_mode"] == "box"
    assert model.effective_parameters_["kernel_tau"] is None
    assert {
        "support_edges",
        "boundary_edges",
        "support_reuse_hits",
    } <= model.result_.trace[0].keys()


def test_multi_fit_uses_sparse_backend_without_dense_path(monkeypatch):
    import importlib

    module = importlib.import_module("ADP.multi_index.ADP_multi_index")
    monkeypatch.setattr(module, "pairwise_distance2", _forbid_dense_path)
    monkeypatch.setattr(module, "calculate_multi_weight", _forbid_dense_path)
    rng = np.random.default_rng(37)
    X = rng.normal(size=(36, 4))
    Y = np.sin(X[:, 0]) + X[:, 1]
    config = ADP_Config(
        seed=3,
        N_loc=4,
        N_J=9,
        N_phi=2,
        outer_steps=1,
        h_min=1e6,
        kernel=make_plateau_kernel(0.4),
        index_init="pilot",
    )

    model = ADP_multi_index(
        2,
        config,
        ADP_solver(_unchanged_index),
    ).fit(X, Y)

    np.testing.assert_allclose(model.basis_.T @ model.basis_, np.eye(2), atol=1e-12)
    assert model.effective_parameters_["kernel_mode"] == "plateau"
    assert model.effective_parameters_["kernel_tau"] == 0.4
    assert {
        "support_edges",
        "boundary_edges",
        "support_reuse_hits",
    } <= model.result_.trace[0].keys()
