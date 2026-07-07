import numpy as np

from adp import ADP, ADPConfig
from adp.backends import numpy_backend
from adp.backends.numpy_backend import NumpyBackend
from adp.common.types import LocalStatistics
from adp.common.utils import pairwise_norm2


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


def test_isotropic_bandwidth_uses_lower_quantile_local_mass():
    X = np.linspace(-0.5, 0.5, 30)[:, None]
    centers = np.array([[-0.2], [0.0], [0.2], [8.0]])
    model = ADP.create(
        "new",
        ADPConfig(
            min_neighbors=5.0,
            kernel="gaussian",
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
    assert result.statistics.directions.dtype == np.float32
