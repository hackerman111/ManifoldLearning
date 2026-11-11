"""Check the PDF's metric operators against residuals and finite differences."""

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from ADP.core.ADP_Statistic import ADP_Statistics
from experiments import full_ridge_batched as backend


def example(kind="full"):
    rng = np.random.default_rng(1286026)
    j, d, m = 7, 6, 2
    raw = rng.normal(size=(j, d, d))
    covariance = raw @ raw.swapaxes(1, 2) + np.eye(d)
    cross = rng.normal(size=(j, d))
    basis = np.linalg.qr(rng.normal(size=(d, m)))[0]
    if kind == "singular":
        covariance[:, 3:] = 0
        covariance[:, :, 3:] = 0
        cross[:, 3:] = 0
    elif kind == "zero":
        covariance[:] = 0
        cross[:] = 0
    stats = ADP_Statistics(
        cross,
        covariance,
        np.arange(1, j + 1, dtype=float),
        np.zeros((j, d)),
        np.ones(j),
        np.empty((j, 0)),
        np.zeros(j),
    )
    return basis, stats


def polar(matrix):
    u, _, vt = np.linalg.svd(matrix, full_matrices=False)
    return u @ vt


def test_sc_and_vp_match_augmented_residual_jacobians():
    from experiments.anisotropic_metrics import TangentGeometry

    b, stats = example()
    p = backend.build(stats, b)
    state = backend.profile(b, p)
    geometry = TangentGeometry(b, p, state)
    v = geometry.complement
    q, m = v.shape[1], b.shape[1]
    sc, vp = np.zeros((q * m, q * m)), np.zeros((q * m, q * m))
    for c, h, weight in zip(p.covariance, p.cross, p.pi, strict=True):
        values, vectors = np.linalg.eigh(c)
        root = (vectors * np.sqrt(values)) @ vectors.T
        target = np.linalg.solve(root, h)

        def residual(basis, root=root, target=target):
            w = np.vstack((root @ basis, np.sqrt(p.tau) * np.eye(m)))
            g = np.linalg.solve(w.T @ w, w.T @ np.r_[target, np.zeros(m)])
            return np.r_[target, np.zeros(m)] - w @ g

        w = np.vstack((root @ b, np.sqrt(p.tau) * np.eye(m)))
        g = np.linalg.solve(w.T @ w, w.T @ np.r_[target, np.zeros(m)])
        eliminated, derivatives = [], []
        for coordinate in np.eye(q * m):
            delta = v @ coordinate.reshape(q, m)
            dz = np.r_[root @ delta @ g, np.zeros(m)]
            eliminated.append(dz - w @ np.linalg.solve(w.T @ w, w.T @ dz))
            eps = 1e-5
            derivatives.append(
                (residual(polar(b + eps * delta)) - residual(polar(b - eps * delta)))
                / (2 * eps)
            )
        sc += weight * np.array(eliminated) @ np.array(eliminated).T
        vp += weight * np.array(derivatives) @ np.array(derivatives).T
    np.testing.assert_allclose(geometry.matrix("schur"), sc, atol=1e-10)
    np.testing.assert_allclose(geometry.matrix("varpro"), vp, rtol=1e-8, atol=1e-9)
    assert np.linalg.eigvalsh(vp - sc).min() >= -1e-10


@pytest.mark.parametrize("kind", ["schur", "varpro", "hessian"])
def test_dense_operator_matches_matrix_free_and_hessian_difference(kind):
    from experiments.anisotropic_metrics import TangentGeometry

    b, stats = example()
    p = backend.build(stats, b)
    geometry = TangentGeometry(b, p, backend.profile(b, p))
    delta = geometry.complement @ np.arange(8).reshape(4, 2) / 10
    dense = geometry.matrix(kind)
    actual = geometry.complement.T @ geometry.apply(delta, kind)
    np.testing.assert_allclose(
        actual.ravel(), dense @ (geometry.complement.T @ delta).ravel(), atol=1e-10
    )
    np.testing.assert_allclose(dense, dense.T, atol=1e-11)
    if kind == "hessian":
        eps = 2e-4
        fp, f0, fm = [
            backend.profile(polar(b + t * delta), p).value for t in (eps, 0, -eps)
        ]
        numeric = (fp - 2 * f0 + fm) / eps**2
        assert numeric == pytest.approx(
            2 * np.sum(delta * geometry.apply(delta, kind)), rel=3e-5, abs=1e-6
        )


@pytest.mark.parametrize("kind", ["schur", "varpro"])
def test_cg_and_dense_damped_steps_agree(kind):
    from experiments.anisotropic_metrics import TangentGeometry

    b, stats = example("singular")
    p = backend.build(stats, b)
    state = backend.profile(b, p)
    geometry = TangentGeometry(b, p, state)
    rhs = state.gradient - b @ (b.T @ state.gradient)
    direct, _ = geometry.solve(rhs, kind, damping=0.02, method="dense")
    iterative, info = geometry.solve(rhs, kind, damping=0.02, method="cg")
    np.testing.assert_allclose(direct, iterative, rtol=1e-8, atol=1e-9)
    assert info["relative_residual"] <= 1e-8
    assert np.sum(rhs * direct) > 0


@pytest.mark.parametrize(
    "arm", ["euclidean", "left", "two_sided", "schur", "varpro", "kernel"]
)
def test_directions_are_horizontal_descent_and_basis_equivariant(arm):
    from experiments.anisotropic_metrics import Metric

    b, stats = example()
    p = backend.build(stats, b)
    state = backend.profile(b, p)
    tensor = np.diag(np.arange(1, 7))
    metric = Metric(b, p, state, arm, metric_tensor=tensor)
    direction, _ = metric.direction(b, state)
    np.testing.assert_allclose(b.T @ direction, 0, atol=1e-11)
    assert np.sum(state.gradient * direction) < 0
    rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
    br = b @ rotation
    sr = backend.profile(br, p)
    other = Metric(br, p, sr, arm, metric_tensor=tensor)
    rotated, _ = other.direction(br, sr)
    np.testing.assert_allclose(rotated, direction @ rotation, rtol=1e-8, atol=1e-9)
    if arm in ("left", "two_sided", "kernel"):
        mb = metric.feature @ b
        schur = metric.feature - mb @ np.linalg.solve(b.T @ mb, mb.T)
        right = metric.right(state) if arm == "two_sided" else np.eye(2)
        test = (np.eye(6) - b @ b.T) @ np.arange(12).reshape(6, 2)
        lhs = np.sum((-direction) * (schur @ test @ right))
        assert lhs == pytest.approx(np.sum(state.gradient * test), rel=1e-10)


@pytest.mark.parametrize("kind", ["full", "singular", "zero"])
@pytest.mark.parametrize(
    "arm", ["euclidean", "left", "two_sided", "schur", "varpro", "kernel"]
)
def test_common_solver_preserves_profile_ridge_and_monotonicity(kind, arm):
    from experiments.anisotropic_metrics import solve

    b, stats = example(kind)
    p = backend.build(stats, b)
    result = solve(b.T, stats, arm, max_steps=5, metric_tensor=np.eye(6))
    np.testing.assert_allclose(result.index @ result.index.T, np.eye(2), atol=1e-12)
    state = backend.profile(result.index.T, p)
    diag = result.diagnostics
    assert diag["penalty"]["local_ridge"] == p.tau
    assert diag["profile_objective"] == pytest.approx(state.value, abs=1e-10)
    assert all(y <= x + 1e-12 for x, y in pairwise(diag["profile_history"]))
    expected = (
        np.linalg.norm(
            state.gradient - result.index.T @ (result.index @ state.gradient)
        )
        / p.scale
    )
    assert diag["riemannian_gradient"] == pytest.approx(expected)


def test_invariant_ridge_preserves_profile_under_generalized_representation():
    from experiments.anisotropic_metrics import invariant_profile

    b, stats = example()
    p = backend.build(stats, b)
    change = np.array([[2.0, 0.3], [0.0, 0.7]])
    z = b @ change
    state = invariant_profile(z, p)
    expected = backend.profile(b, p)
    assert state.value == pytest.approx(expected.value, abs=1e-12)
    np.testing.assert_allclose(
        z @ state.coefficients.T, b @ expected.coefficients.T, atol=1e-11
    )
    dz = np.arange(12).reshape(6, 2) / 10
    eps = 1e-5
    numeric = (
        invariant_profile(z + eps * dz, p).value
        - invariant_profile(z - eps * dz, p).value
    ) / (2 * eps)
    assert numeric == pytest.approx(np.sum(state.gradient * dz), rel=1e-7)


def test_zero_ridge_rank_drop_is_rejected_and_bad_inputs_are_explicit():
    from experiments.anisotropic_metrics import Metric, TangentGeometry, solve

    b, stats = example("singular")
    p = backend.build(stats, b)
    p.tau = 0
    bad = np.eye(6)[:, 4:]
    dummy = replace(backend.profile(b, p), coefficients=np.zeros((7, 2)))
    with pytest.raises(ValueError, match="rank|regular"):
        TangentGeometry(bad, p, dummy)
    with pytest.raises(ValueError):
        Metric(b, p, dummy, "kernel", metric_tensor=np.full((6, 6), np.nan))
    with pytest.raises(ValueError):
        solve(b.T * 2, stats, "euclidean")
    with pytest.raises(ValueError):
        solve(b.T, stats, "typo")
    with pytest.raises(ValueError):
        solve(b.T, stats, "schur", max_steps=-1)


@pytest.mark.parametrize("kind", ["orthogonal", "full"])
def test_kernel_tensor_matches_actual_adp_weight_quadratic(kind):
    from experiments.anisotropic_metrics import localization_tensor

    b, _ = example()
    alpha, eigenvalues = 0.2, np.array([0.3, 0.8])
    tensor = localization_tensor(b, alpha, eigenvalues, kind)
    x = np.arange(6) / 5
    projected = b.T @ x
    residual = x @ x - projected @ projected if kind == "orthogonal" else x @ x
    expected = alpha**2 * residual + np.sum(eigenvalues * projected**2)
    assert np.sum((tensor @ x) ** 2) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "arm", ["euclidean", "left", "two_sided", "schur", "varpro", "kernel"]
)
def test_numerical_nullspace_keeps_whitened_profile_fallback(arm):
    from experiments.anisotropic_metrics import solve

    b, stats = example()
    b = np.eye(6)[:, 2:4] + 1e-9 * np.eye(6)[:, :2]
    b = np.linalg.qr(b)[0]
    stats = replace(
        stats,
        U=np.tile(np.diag([2.0, 1.0, 0, 0, 0, 0]), (7, 1, 1)),
        I=np.tile([1.0, 0, 0, 0, 0, 0], (7, 1)),
    )
    p = backend.build(stats, b)
    assert p.info["whitened_fallback"]
    result = solve(b.T, stats, arm, max_steps=3, metric_tensor=np.eye(6))
    assert result.diagnostics["penalty"]["full_moments"]["whitened_fallback"]
    expected = backend.profile(result.index.T, p)
    assert result.diagnostics["profile_objective"] == pytest.approx(
        expected.value, abs=1e-10
    )
