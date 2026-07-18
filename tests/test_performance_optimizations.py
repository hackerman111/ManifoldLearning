from dataclasses import replace
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from adp import ADP, ADPConfig
from adp.backends import numpy_backend
from adp.backends.numpy_backend import NumpyBackend
from adp.common.types import LocalStatistics
from adp.common.utils import kernel_np, pairwise_norm2


def install_fake_cupy(monkeypatch):
    calls = []

    def record(name, fn):
        def wrapped(*args, **kwargs):
            calls.append(name)
            return fn(*args, **kwargs)

        return wrapped

    fake = SimpleNamespace(
        asarray=record("asarray", np.asarray),
        asnumpy=record("asnumpy", np.asarray),
        exp=record("exp", np.exp),
        maximum=record("maximum", np.maximum),
        square=record("square", np.square),
        einsum=record("einsum", np.einsum),
        matmul=record("matmul", np.matmul),
        swapaxes=record("swapaxes", np.swapaxes),
        zeros=record("zeros", np.zeros),
        quantile=record("quantile", np.quantile),
        finfo=np.finfo,
    )
    monkeypatch.setitem(sys.modules, "cupy", fake)
    return calls


def reference_random_projection_sums(X, y, centers, directions, q, kernel):
    backend = NumpyBackend()
    weights = backend.kernel(q, kernel)
    counts = weights.sum(axis=1)
    imav = np.zeros((centers.shape[0], directions.shape[1]))
    s_all = np.zeros_like(imav)
    u_all = np.zeros((centers.shape[0], directions.shape[1], X.shape[1]))
    for j in range(centers.shape[0]):
        safe_count = max(float(counts[j]), np.finfo(float).eps)
        xbar = (weights[j] @ X) / safe_count
        centered = X - xbar
        projected = centered @ directions[j].T
        weighted_projected = projected * weights[j, :, None]
        s_all[j] = weighted_projected.sum(axis=0)
        imav[j] = (weights[j] * y) @ projected
        u_all[j] = weighted_projected.T @ centered
    return imav, s_all, u_all, counts, float(counts.mean())


@pytest.mark.parametrize("kernel", ("epanechnikov", "quartic", "gaussian"))
def test_numpy_and_cupy_kernels_match_squared_distance_convention(monkeypatch, kernel):
    from adp.backends.cupy_backend import CupyBackend

    install_fake_cupy(monkeypatch)
    q = np.array([0.25, 0.5, 1.5])
    expected = kernel_np(q, kernel)

    np.testing.assert_allclose(NumpyBackend().kernel(q, kernel), expected)
    cupy_backend = CupyBackend()
    np.testing.assert_allclose(cupy_backend.to_numpy(cupy_backend.kernel(q, kernel)), expected)


def test_local_mass_mode_selects_mean_or_configured_quantile():
    q = np.array([[0.0, 0.0], [0.0, 100.0]])

    mean_model = ADP.create(
        "new",
        ADPConfig(local_mass_mode="mean", kernel="gaussian", show_progress=False),
    )
    quantile_model = ADP.create(
        "new",
        ADPConfig(
            local_mass_mode="quantile",
            local_mass_quantile=0.25,
            kernel="gaussian",
            show_progress=False,
        ),
    )

    assert mean_model._local_mass_score(q) == pytest.approx(1.5)
    assert quantile_model._local_mass_score(q) == pytest.approx(1.25)


def test_random_initial_beta_mode_is_reproducible_and_data_independent():
    config = ADPConfig(initial_beta_mode="random", random_state=17, show_progress=False)
    first = ADP.create("new", config)
    second = ADP.create("new", config)

    beta_first = first._initial_beta(np.arange(12.0).reshape(4, 3), np.arange(4.0))
    beta_second = second._initial_beta(
        np.array([[100.0, -2.0, 1.0], [0.0, 3.0, 8.0], [4.0, 5.0, -9.0], [2.0, 7.0, 6.0]]),
        np.array([9.0, -1.0, 2.0, 4.0]),
    )

    np.testing.assert_allclose(beta_first, beta_second)
    assert np.linalg.norm(beta_first) == pytest.approx(1.0)


def test_config_rejects_unknown_initial_beta_and_local_mass_modes():
    with pytest.raises(ValueError, match="initial_beta_mode"):
        ADPConfig(initial_beta_mode="bad")
    with pytest.raises(ValueError, match="local_mass_mode"):
        ADPConfig(local_mass_mode="bad")


def test_backend_random_projection_sums_match_weighted_xbar_reference():
    X = np.array(
        [
            [-1.0, 0.5, 0.1],
            [0.2, -0.3, 0.4],
            [0.8, 1.1, -0.7],
            [1.5, -1.2, 0.9],
            [-0.6, 0.7, -1.0],
        ]
    )
    y = np.array([0.4, -0.2, 0.9, 1.4, -0.8])
    centers = np.array([[0.1, 0.2, -0.1], [1.0, -0.8, 0.7]])
    directions = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, -1.0, 0.5]],
        ]
    )
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    q = pairwise_norm2(X, centers) / 4.0

    actual = NumpyBackend().random_projection_sums(
        X=X,
        y=y,
        centers=centers,
        directions=directions,
        q=q,
        kernel="gaussian",
    )
    expected = reference_random_projection_sums(X, y, centers, directions, q, "gaussian")

    for actual_part, expected_part in zip(actual, expected):
        np.testing.assert_allclose(actual_part, expected_part, rtol=1e-12, atol=1e-12)


def test_cupy_backend_covers_gpu_md_pairwise_kernel_and_local_mass(monkeypatch):
    from adp.backends.cupy_backend import CupyBackend

    calls = install_fake_cupy(monkeypatch)
    X = np.array(
        [
            [-1.0, 0.5, 0.1],
            [0.2, -0.3, 0.4],
            [0.8, 1.1, -0.7],
            [1.5, -1.2, 0.9],
            [-0.6, 0.7, -1.0],
        ]
    )
    centers = np.array([[0.1, 0.2, -0.1], [1.0, -0.8, 0.7]])
    backend = CupyBackend()

    norm2 = backend.pairwise_norm2(X, centers)
    mass = backend.local_mass_score(norm2 / 4.0, "gaussian")

    np.testing.assert_allclose(backend.to_numpy(norm2), pairwise_norm2(X, centers))
    assert np.isclose(mass, reference_random_projection_sums(X, np.ones(X.shape[0]), centers, np.ones((2, 1, 3)), norm2 / 4.0, "gaussian")[3].mean())
    assert "einsum" in calls
    assert "exp" in calls
    assert "asnumpy" in calls


def test_cupy_backend_statistics_match_numpy_with_fake_cupy(monkeypatch):
    install_fake_cupy(monkeypatch)
    rng = np.random.default_rng(17)
    X = rng.normal(size=(18, 4))
    y = rng.normal(size=18)
    centers = rng.normal(size=(5, 4))
    beta = np.array([1.0, 0.2, -0.1, 0.3])
    beta = beta / np.linalg.norm(beta)
    directions = rng.normal(size=(5, 3, 4))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    config = ADPConfig(
        n_centers=5,
        n_directions=3,
        chunk_size=3,
        kernel="gaussian",
        show_progress=False,
    )
    numpy_model = ADP.create("new", config)
    cupy_model = ADP.create("new", replace(config, backend="cupy"))

    numpy_stats = numpy_model._compute_statistics(X, y, centers, 1.7, beta, directions, None)
    cupy_stats = cupy_model._compute_statistics(X, y, centers, 1.7, beta, directions, None)

    np.testing.assert_allclose(cupy_stats.imav, numpy_stats.imav, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cupy_stats.S, numpy_stats.S, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cupy_stats.U, numpy_stats.U, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cupy_stats.N, numpy_stats.N, rtol=1e-12, atol=1e-12)
    assert cupy_model.backend.name == "cupy"


def test_cupy_backend_reuses_prepared_statistics_inputs(monkeypatch):
    from adp.backends.cupy_backend import CupyBackend

    calls = install_fake_cupy(monkeypatch)
    backend = CupyBackend()
    X = np.arange(12.0).reshape(4, 3)
    y = np.arange(4.0)
    centers = X[:2].copy()
    directions = np.ones((2, 3, 3))

    first = backend.prepare_statistics_inputs(X, y, centers, directions)
    second = backend.prepare_statistics_inputs(X, y, centers, directions)

    assert all(left is right for left, right in zip(first, second))
    assert calls.count("asarray") == 4

    backend.clear_device_cache()
    backend.prepare_statistics_inputs(X, y, centers, directions)
    assert calls.count("asarray") == 8


def test_cupy_statistics_transfers_chunk_results_only_at_finalize(monkeypatch):
    from adp.backends.cupy_backend import CupyBackend

    calls = install_fake_cupy(monkeypatch)
    rng = np.random.default_rng(23)
    X = rng.normal(size=(12, 4))
    y = rng.normal(size=12)
    centers = rng.normal(size=(5, 4))
    beta = np.array([1.0, 0.2, -0.1, 0.3])
    beta /= np.linalg.norm(beta)
    directions = rng.normal(size=(5, 3, 4))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    model = ADP.create(
        "new",
        ADPConfig(
            n_centers=5,
            n_directions=3,
            chunk_size=2,
            kernel="gaussian",
            backend="cupy",
            show_progress=False,
        ),
    )

    stats = model._compute_statistics(X, y, centers, 1.7, beta, directions, None)

    assert stats.imav.shape == (5, 3)
    assert calls.count("asnumpy") == 4


def test_cupy_backend_fails_fast_without_cuda_device(monkeypatch):
    from adp.backends.cupy_backend import CupyBackend

    fake = SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(getDeviceCount=lambda: 0),
        )
    )
    monkeypatch.setitem(sys.modules, "cupy", fake)

    with pytest.raises(RuntimeError, match="CUDA"):
        CupyBackend()


@pytest.mark.parametrize("dtype", ("float64", "float32"))
def test_numpy_kernel_argument_matches_isotropic_and_anisotropic_formulas(dtype):
    backend = NumpyBackend(dtype)
    norm2 = np.array([[1.0, 4.0], [9.0, 16.0]], dtype=dtype)
    projection2 = np.array([[0.25, 1.0], [2.25, 4.0]], dtype=dtype)

    isotropic = backend.kernel_argument(norm2, h=2.0)
    anisotropic = backend.kernel_argument(
        norm2,
        h=2.0,
        projection2=projection2,
        anisotropy=0.5,
    )

    np.testing.assert_allclose(isotropic, norm2 / 4.0)
    np.testing.assert_allclose(anisotropic, (0.25 * norm2 + projection2) / 4.0)
    assert isotropic.dtype == np.dtype(dtype)
    assert anisotropic.dtype == np.dtype(dtype)


def test_kernel_argument_requires_projection_for_anisotropy():
    with pytest.raises(ValueError, match="projection2"):
        NumpyBackend().kernel_argument(np.ones((2, 3)), h=1.0, anisotropy=0.5)


def test_cupy_kernel_argument_matches_numpy_with_fake_cupy(monkeypatch):
    from adp.backends.cupy_backend import CupyBackend

    install_fake_cupy(monkeypatch)
    norm2 = np.array([[1.0, 4.0], [9.0, 16.0]])
    projection2 = np.array([[0.25, 1.0], [2.25, 4.0]])
    backend = CupyBackend()

    actual = backend.kernel_argument(
        norm2,
        h=2.0,
        projection2=projection2,
        anisotropy=0.5,
    )

    np.testing.assert_allclose(
        backend.to_numpy(actual),
        (0.25 * norm2 + projection2) / 4.0,
    )


def test_cupy_backend_releases_cached_arrays_and_memory_pools(monkeypatch):
    from adp.backends.cupy_backend import CupyBackend

    class Pool:
        def __init__(self):
            self.free_calls = 0

        def free_all_blocks(self):
            self.free_calls += 1

    device_pool = Pool()
    pinned_pool = Pool()
    fake = SimpleNamespace(
        asarray=np.asarray,
        get_default_memory_pool=lambda: device_pool,
        get_default_pinned_memory_pool=lambda: pinned_pool,
    )
    monkeypatch.setitem(sys.modules, "cupy", fake)
    backend = CupyBackend()
    backend.prepare_statistics_inputs(
        np.zeros((2, 3)),
        np.zeros(2),
        np.zeros((1, 3)),
        np.zeros((1, 2, 3)),
    )

    backend.release_device_memory()

    assert backend._device_cache == {}
    assert device_pool.free_calls == 1
    assert pinned_pool.free_calls == 1


def test_cupy_fit_releases_device_memory_after_training(monkeypatch):
    install_fake_cupy(monkeypatch)
    model = ADP.create(
        "new",
        ADPConfig(
            backend="cupy",
            n_centers=4,
            n_directions=2,
            min_neighbors=3,
            outer_steps=1,
            inner_steps=1,
            show_progress=False,
            random_state=29,
        ),
    )
    data = model.generate_data(n=16, d=3, noise=0.01, link="linear")
    release_calls = []
    monkeypatch.setattr(model.backend, "release_device_memory", lambda: release_calls.append(True))

    model.fit(data.X, data.y, centers=data.centers, directions=data.directions, beta0=data.beta)

    assert release_calls == [True]


def test_isotropic_bandwidth_uses_lower_quantile_local_mass():
    X = np.linspace(-0.5, 0.5, 30)[:, None]
    centers = np.array([[-0.2], [0.0], [0.2], [8.0]])
    model = ADP.create(
        "new",
        ADPConfig(
            min_neighbors=5.0,
            kernel="gaussian",
            local_mass_mode="quantile",
            local_mass_quantile=0.25,
            scale_search_steps=28,
            scale_expand_steps=20,
            show_progress=False,
        ),
    )

    h = model._select_isotropic_bandwidth(X, centers)
    masses = model.backend.kernel(pairwise_norm2(X, centers) / (h * h), "gaussian").sum(axis=1)

    assert np.quantile(masses, 0.25) >= model.config.min_neighbors - 1e-5
    assert masses.mean() > model.config.min_neighbors


def test_solve_beta_uses_diagonal_cg_preconditioner(monkeypatch):
    import adp.variants.random_projection as random_projection

    captured = {}

    def fake_cg(operator, rhs, **kwargs):
        captured["M"] = kwargs.get("M")
        return rhs, 0

    monkeypatch.setattr(random_projection, "cg", fake_cg)
    model = ADP.create("new", ADPConfig(show_progress=False))
    stats = LocalStatistics(
        variant="new",
        imav=np.array([[1.0, 0.5], [0.4, -0.3]]),
        centers=np.zeros((2, 3)),
        h=1.0,
        weights_mean=4.0,
        S=np.zeros((2, 2)),
        U=np.array(
            [
                [[1.0, 0.2, 0.0], [0.1, 0.8, 0.3]],
                [[0.4, 0.0, 0.7], [0.3, 0.5, 0.2]],
            ]
        ),
    )

    model._solve_beta(
        stats=stats,
        intercepts=np.zeros(2),
        slopes=np.array([1.0, 0.7]),
        prior=np.array([1.0, 0.0, 0.0]),
        lambda_penalty=0.2,
    )

    assert captured["M"] is not None
    np.testing.assert_allclose(captured["M"].matvec(np.ones(3)), captured["M"] @ np.ones(3))
    assert np.all(np.isfinite(captured["M"].matvec(np.ones(3))))


def test_solve_beta_accepts_warm_start_for_cg(monkeypatch):
    import adp.variants.random_projection as random_projection

    captured = {}

    def fake_cg(operator, rhs, **kwargs):
        captured["x0"] = kwargs.get("x0")
        return rhs, 0

    monkeypatch.setattr(random_projection, "cg", fake_cg)
    model = ADP.create("new", ADPConfig(show_progress=False))
    stats = LocalStatistics(
        variant="new",
        imav=np.array([[0.2, 0.1], [0.5, -0.4]]),
        centers=np.zeros((2, 3)),
        h=1.0,
        weights_mean=4.0,
        S=np.zeros((2, 2)),
        U=np.array(
            [
                [[0.9, 0.1, 0.0], [0.2, 0.7, 0.2]],
                [[0.3, 0.1, 0.8], [0.4, 0.4, 0.1]],
            ]
        ),
    )
    warm_start = np.array([0.2, 0.9, 0.1])

    model._solve_beta(
        stats=stats,
        intercepts=np.zeros(2),
        slopes=np.array([1.0, 0.8]),
        prior=np.array([1.0, 0.0, 0.0]),
        lambda_penalty=0.2,
        x0=warm_start,
    )

    np.testing.assert_allclose(captured["x0"], warm_start)


def test_solve_beta_uses_flattened_gemv_system_without_einsum(monkeypatch):
    import adp.variants.random_projection as random_projection

    model = ADP.create("new", ADPConfig(tol=1e-11, show_progress=False))
    stats = LocalStatistics(
        variant="new",
        imav=np.array([[1.0, 0.5], [0.4, -0.3]]),
        centers=np.zeros((2, 3)),
        h=1.0,
        weights_mean=4.0,
        S=np.zeros((2, 2)),
        U=np.array(
            [
                [[1.0, 0.2, 0.0], [0.1, 0.8, 0.3]],
                [[0.4, 0.0, 0.7], [0.3, 0.5, 0.2]],
            ]
        ),
    )
    slopes = np.array([1.0, 0.7])
    prior = np.array([1.0, 0.0, 0.0])
    lambda_penalty = 0.2

    def fail_einsum(*args, **kwargs):
        raise AssertionError("_solve_beta should use flattened BLAS products")

    monkeypatch.setattr(random_projection.np, "einsum", fail_einsum)
    beta = model._solve_beta(
        stats=stats,
        intercepts=np.zeros(2),
        slopes=slopes,
        prior=prior,
        lambda_penalty=lambda_penalty,
    )

    u_flat = stats.U.reshape(-1, stats.U.shape[-1])
    slope_flat = np.repeat(slopes, stats.U.shape[1])
    regularization = lambda_penalty + model.config.ridge
    lhs = u_flat.T @ (slope_flat[:, None] ** 2 * u_flat)
    lhs += regularization * np.eye(u_flat.shape[1])
    rhs = u_flat.T @ (slope_flat * stats.imav.reshape(-1)) + lambda_penalty * prior
    expected = np.linalg.solve(lhs, rhs)
    np.testing.assert_allclose(beta, expected, rtol=1e-8, atol=1e-8)


def test_compact_kernel_projection_sums_avoid_dense_3d_matmul(monkeypatch):
    X = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [4.0, 4.0],
            [5.0, 5.0],
        ]
    )
    y = np.array([1.0, 0.5, -1.0, 0.2])
    centers = np.array([[0.0, 0.0], [5.0, 5.0]])
    directions = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, -1.0]],
        ]
    )
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    q = pairwise_norm2(X, centers)
    original_matmul = numpy_backend.np.matmul

    def guarded_matmul(a, b, *args, **kwargs):
        if getattr(a, "ndim", 0) == 3 or getattr(b, "ndim", 0) == 3:
            raise AssertionError("compact kernels should not build dense 3D projected blocks")
        return original_matmul(a, b, *args, **kwargs)

    monkeypatch.setattr(numpy_backend.np, "matmul", guarded_matmul)

    actual = NumpyBackend().random_projection_sums(
        X=X,
        y=y,
        centers=centers,
        directions=directions,
        q=q,
        kernel="epanechnikov",
    )
    expected = reference_random_projection_sums(X, y, centers, directions, q, "epanechnikov")

    for actual_part, expected_part in zip(actual, expected):
        np.testing.assert_allclose(actual_part, expected_part, rtol=1e-12, atol=1e-12)


def test_projection_cache_keeps_only_latest_beta_projection():
    model = ADP.create("new", ADPConfig(show_progress=False))
    X = np.arange(12.0).reshape(6, 2)
    centers = X[:3]

    model._cached_pairwise_projection2(X, centers, np.array([1.0, 0.0]))
    model._cached_pairwise_projection2(X, centers, np.array([0.0, 1.0]))

    projection_keys = [key for key in model._pairwise_cache if key[0] == "proj2"]
    assert len(projection_keys) == 1


def test_default_bandwidth_search_budget_is_short():
    config = ADPConfig(show_progress=False)

    assert config.scale_expand_steps <= 16
    assert config.scale_search_steps <= 12
    assert config.anisotropy_search_steps <= 12


def test_objective_check_every_reduces_full_objective_passes(monkeypatch):
    model = ADP.create(
        "new",
        ADPConfig(
            inner_steps=5,
            objective_check_every=3,
            show_progress=False,
        ),
    )
    stats = LocalStatistics(
        variant="new",
        imav=np.array([[1.0, 0.2], [0.3, -0.1]]),
        centers=np.zeros((2, 3)),
        h=1.0,
        weights_mean=4.0,
        S=np.zeros((2, 2)),
        U=np.array(
            [
                [[1.0, 0.2, 0.0], [0.1, 0.8, 0.3]],
                [[0.4, 0.0, 0.7], [0.3, 0.5, 0.2]],
            ]
        ),
    )
    calls = {"objective": 0}

    def no_convergence_beta(stats, intercepts, slopes, prior, lambda_penalty, x0=None):
        return np.asarray(x0, dtype=float) + np.array([0.2, -0.1, 0.05])

    def counted_objective(*args, **kwargs):
        calls["objective"] += 1
        return 10.0 - calls["objective"]

    monkeypatch.setattr(model, "_solve_beta", no_convergence_beta)
    monkeypatch.setattr(model, "_objective", counted_objective)

    _, _, _, history = model._alternating_solve(
        stats,
        beta_start=np.array([1.0, 0.0, 0.0]),
        lambda_penalty=0.2,
        outer=0,
        outer_started=0.0,
    )

    assert len(history) == 5
    assert calls["objective"] < len(history)
    assert np.isfinite(history[-1].objective)


def test_float32_config_preserves_statistics_dtype():
    model = ADP.create(
        "new",
        ADPConfig(
            n_centers=8,
            n_directions=3,
            min_neighbors=4,
            outer_steps=1,
            inner_steps=2,
            dtype="float32",
            show_progress=False,
            random_state=11,
        ),
    )
    data = model.generate_data(n=40, d=4, noise=0.01, link="linear")

    result = model.fit(data.X, data.y, beta0=data.beta)

    assert result.statistics.imav.dtype == np.float32
    assert result.statistics.S.dtype == np.float32
    assert result.statistics.U.dtype == np.float32
    assert result.statistics.directions is None
    assert result.statistics.n_directions == 3


@pytest.mark.parametrize("kernel", ("epanechnikov", "quartic"))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (("float64", 1e-11, 1e-12), ("float32", 2e-5, 2e-6)),
)
def test_fused_compact_statistics_match_reference_and_make_s_exact_zero(
    kernel,
    dtype,
    rtol,
    atol,
):
    rng = np.random.default_rng(43)
    X = rng.normal(size=(30, 5)).astype(dtype)
    y = rng.normal(size=30).astype(dtype)
    centers = rng.normal(size=(4, 5)).astype(dtype)
    directions = rng.normal(size=(4, 3, 5)).astype(dtype)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    q = (pairwise_norm2(X, centers) / 8.0).astype(dtype)
    backend = NumpyBackend(dtype)

    actual = backend.random_projection_sums(
        X=X,
        y=y,
        centers=centers,
        directions=directions,
        q=q,
        kernel=kernel,
    )
    expected = reference_random_projection_sums(X, y, centers, directions, q, kernel)

    np.testing.assert_allclose(actual[0], expected[0], rtol=rtol, atol=atol)
    np.testing.assert_array_equal(actual[1], np.zeros_like(actual[1]))
    np.testing.assert_allclose(actual[2], expected[2], rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual[3], expected[3], rtol=rtol, atol=atol)
    assert actual[0].dtype == np.dtype(dtype)
    assert actual[1].dtype == np.dtype(dtype)
    assert actual[2].dtype == np.dtype(dtype)
    assert actual[3].dtype == np.dtype(dtype)


def test_fused_compact_statistics_keep_empty_center_zero():
    X = np.array([[0.0, 0.0], [0.2, 0.1], [1.0, -0.5]])
    y = np.array([1.0, -0.5, 0.25])
    centers = np.array([[10.0, 10.0], [0.0, 0.0]])
    directions = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    q = np.array([[4.0, 5.0, 6.0], [0.0, 0.25, 2.0]])

    imav, s_vec, u_mat, counts, _ = NumpyBackend().random_projection_sums(
        X=X,
        y=y,
        centers=centers,
        directions=directions,
        q=q,
        kernel="epanechnikov",
    )

    np.testing.assert_array_equal(imav[0], np.zeros_like(imav[0]))
    np.testing.assert_array_equal(s_vec[0], np.zeros_like(s_vec[0]))
    np.testing.assert_array_equal(u_mat[0], np.zeros_like(u_mat[0]))
    assert counts[0] == 0.0
    assert counts[1] > 0.0


def test_statistics_workers_must_be_positive():
    with pytest.raises(ValueError, match="statistics_workers"):
        ADPConfig(statistics_workers=0)

    with pytest.raises(ValueError, match="statistics_workers"):
        ADPConfig(statistics_workers=1.5)


def test_numpy_backend_receives_statistics_workers():
    model = ADP.create(
        "new",
        ADPConfig(statistics_workers=3, show_progress=False),
    )

    assert model.backend.statistics_workers == 3


def test_parallel_compact_statistics_match_serial(monkeypatch):
    monkeypatch.setattr(numpy_backend, "PARALLEL_STATISTICS_MIN_WORK", 0)
    rng = np.random.default_rng(59)
    X = rng.normal(size=(40, 6))
    y = rng.normal(size=40)
    centers = rng.normal(size=(6, 6))
    directions = rng.normal(size=(6, 4, 6))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    q = pairwise_norm2(X, centers) / 10.0

    serial = NumpyBackend(statistics_workers=1).random_projection_sums(
        X=X,
        y=y,
        centers=centers,
        directions=directions,
        q=q,
        kernel="epanechnikov",
    )
    parallel = NumpyBackend(statistics_workers=2).random_projection_sums(
        X=X,
        y=y,
        centers=centers,
        directions=directions,
        q=q,
        kernel="epanechnikov",
    )

    for serial_part, parallel_part in zip(serial, parallel):
        np.testing.assert_allclose(
            parallel_part,
            serial_part,
            rtol=1e-12,
            atol=1e-12,
        )


def test_parallel_compact_statistics_bounds_submitted_centers(monkeypatch):
    observed = {}

    class RecordingExecutor:
        def __init__(self, *, max_workers):
            observed["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def map(self, function, iterable, *, buffersize=None):
            observed["buffersize"] = buffersize
            return map(function, iterable)

    monkeypatch.setattr(numpy_backend, "PARALLEL_STATISTICS_MIN_WORK", 0)
    monkeypatch.setattr(numpy_backend, "ThreadPoolExecutor", RecordingExecutor)
    X = np.arange(24.0).reshape(8, 3)
    y = np.arange(8.0)
    centers = X[:4]
    directions = np.ones((4, 2, 3))
    q = np.zeros((4, 8))

    NumpyBackend(statistics_workers=2).random_projection_sums(
        X=X,
        y=y,
        centers=centers,
        directions=directions,
        q=q,
        kernel="epanechnikov",
    )

    assert observed == {"max_workers": 2, "buffersize": 2}


def test_opt_in_fit_telemetry_records_real_cg_and_local_diagnostics():
    model = ADP.create(
        "new",
        ADPConfig(
            n_centers=8,
            n_directions=4,
            min_neighbors=4,
            outer_steps=1,
            inner_steps=2,
            tol=1e-12,
            record_telemetry=True,
            record_solver_trace=True,
            random_state=71,
            show_progress=False,
        ),
    )
    data = model.generate_data(n=48, d=4, noise=0.05, link="quadratic")

    result = model.fit(
        data.X,
        data.y,
        beta0=np.array([0.2, -0.4, 0.7, 0.1]),
    )

    assert len(result.outer_telemetry) == 1
    assert len(result.local_telemetry) == result.statistics.centers.shape[0]
    assert result.outer_telemetry[0]["service_overhead_sec"] >= 0.0
    assert not math.isnan(result.outer_telemetry[0]["condition_median"])
    assert result.statistics.weight_sum2 is not None
    assert result.statistics.weight_nonzero is not None
    assert result.statistics.min_weight is not None
    assert result.statistics.max_weight is not None
    assert result.history
    for step in result.history:
        assert step.objective_before is not None
        assert step.objective_after is not None
        assert step.pre_normalization_beta_norm > 0.0
        assert step.gradient_norm >= 0.0
        assert step.linear_residual_norm >= 0.0
        assert step.gradient_norm == 2.0 * step.linear_residual_norm
        assert step.relative_linear_residual >= 0.0
        assert step.linear_solver_iterations == len(step.solver_residual_trace)
        assert step.linear_solver_status in {"converged", "max_iterations"}
        assert step.inner_iteration_time_sec >= 0.0
        assert step.beta is not None
        assert np.linalg.norm(step.beta) == pytest.approx(1.0)
        assert all(value >= 0.0 for value in step.solver_residual_trace)


def test_disabled_fit_telemetry_keeps_optional_payloads_empty():
    model = ADP.create(
        "new",
        ADPConfig(
            n_centers=6,
            n_directions=3,
            outer_steps=1,
            inner_steps=1,
            random_state=73,
            show_progress=False,
        ),
    )
    data = model.generate_data(n=30, d=3, noise=0.01, link="linear")

    result = model.fit(data.X, data.y, beta0=data.beta)

    assert result.outer_telemetry == []
    assert result.local_telemetry == []
    assert result.statistics.weight_sum2 is None
    assert result.statistics.weight_nonzero is None
    assert result.statistics.min_weight is None
    assert result.statistics.max_weight is None
