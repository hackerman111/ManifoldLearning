"""Dense references and invariants for the isolated full-moment variants."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import block_diag

from experiments import full_moment_variants as variants


def fixture(seed=37, *, singular=False):
    rng = np.random.default_rng(seed)
    j, n, d, m = 6, 30, 7, 2
    x, y = rng.normal(size=(n, d)), rng.normal(size=n)
    weights = rng.uniform(0.1, 1, size=(j, n))
    if singular:
        weights[:, 5:] = 0
    directions = np.zeros((j, 3, d))
    stats = variants.calculate_statistics(x, y, iter([(0, weights)]), directions)
    basis = np.linalg.qr(rng.normal(size=(d, m)))[0]
    return rng, stats, basis


@pytest.mark.parametrize("arm", ["full0", "metric", "coherence"])
@pytest.mark.parametrize("singular", [False, True])
def test_profile_gradient_and_basis_invariance(arm, singular):
    rng, stats, basis = fixture(singular=singular)
    p = variants.build(stats, basis, arm, smoothness=0.01)
    state = variants.profile(basis, p)
    direction = rng.normal(size=basis.shape)
    direction -= basis @ (basis.T @ direction)
    eps = 2e-6
    plus = np.linalg.qr(basis + eps * direction)[0]
    minus = np.linalg.qr(basis - eps * direction)[0]
    derivative = (
        variants.profile(plus, p).value - variants.profile(minus, p).value
    ) / (2 * eps)
    assert derivative == pytest.approx(
        np.sum(state.gradient * direction), abs=2e-6, rel=2e-5
    )
    rotation = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    assert variants.profile(basis @ rotation, p).value == pytest.approx(
        state.value, abs=2e-8
    )
    assert state.normal_residual < 2e-7


@pytest.mark.parametrize("singular", [False, True])
def test_profiles_match_dense_least_squares(singular):
    _, stats, basis = fixture(singular=singular)
    for arm in ("full0", "metric"):
        p = variants.build(stats, basis, arm)
        state = variants.profile(basis, p)
        expected = 0.0
        for group, root in zip(p.groups, p.metric_roots, strict=True):
            design = block_diag(*(np.sqrt(p.pi[k]) * p.root[k] @ basis for k in group))
            target = np.concatenate([np.sqrt(p.pi[k]) * p.target[k] for k in group])
            design, target = root @ design, root @ target
            g = np.linalg.lstsq(design, target, rcond=1e-12)[0]
            expected += np.linalg.norm(design @ g - target) ** 2
        assert state.value == pytest.approx(expected, abs=1e-11)


def test_exact_model_has_zero_loss_for_every_variant_and_no_tangent_drift():
    _, stats, basis = fixture()
    g = np.tile(np.array([0.7, -0.2]), (6, 1))
    exact = replace(stats, I=(stats.U @ basis @ g[..., None]).squeeze(-1))
    for arm in ("full0", "metric", "coherence"):
        p = variants.build(exact, basis, arm)
        state = variants.profile(basis, p)
        assert state.value < 1e-20
        assert np.linalg.norm(state.gradient) < 1e-10
    result = variants.solve(basis.T, exact, "tangent")
    np.testing.assert_allclose(
        result.index.T @ result.index, basis @ basis.T, atol=1e-10
    )


def test_overlap_covariance_is_computed_from_shared_observations():
    _, stats, basis = fixture()
    p = variants.build(stats, basis, "metric")
    a, b = p.groups[0]
    psi = []
    for k in (a, b):
        indices, weights, z = stats.windows[k]
        matrix = np.zeros((basis.shape[0], 30))
        matrix[:, indices] = np.sqrt(p.pi[k]) * p.inverse_root[k] @ (z.T * weights)
        psi.append(matrix)
    expected = np.vstack(psi) @ np.vstack(psi).T
    np.testing.assert_allclose(p.noise_covariances[0], expected, atol=1e-13)


def test_zero_and_low_rank_windows_are_supported():
    _, stats, basis = fixture(singular=True)
    zero = replace(stats, I=np.zeros_like(stats.I), U=np.zeros_like(stats.U))
    for arm in ("full0", "metric", "coherence"):
        p = variants.build(zero, basis, arm)
        state = variants.profile(basis, p)
        assert state.value == 0
        assert np.isfinite(state.gradient).all()


def test_underdetermined_windows_with_joint_graph_are_supported():
    rng, stats, basis = fixture()
    direction = rng.normal(size=(6, 7, 1))
    covariance = direction @ direction.swapaxes(1, 2)
    g = np.tile(np.array([0.7, -0.2]), (6, 1))
    exact = replace(
        stats, U=covariance, I=(covariance @ basis @ g[..., None]).squeeze(-1)
    )
    for arm in ("full0", "metric", "coherence"):
        p = variants.build(exact, basis, arm, smoothness=0)
        state = variants.profile(basis, p)
        assert state.value < 1e-12
        assert state.normal_residual < 2e-7
        assert p.info["incoming_design_rank_histogram"][1] == 6


def test_compressed_step_matches_dense_joint_linearization():
    _, stats, basis = fixture()
    p = variants.build(stats, basis, "full0")
    state = variants.profile(basis, p)
    h, score = variants.tangent_system(basis, state, p)
    complement = np.linalg.qr(basis, mode="complete")[0][:, basis.shape[1] :]
    q, m = h.shape[0], basis.shape[1]
    columns = []
    for coordinate in range(q):
        delta = complement @ np.eye(q)[coordinate].reshape(complement.shape[1], m)
        columns.append(
            (
                np.sqrt(p.pi)[:, None]
                * (p.root @ delta @ state.coefficients[..., None]).squeeze(-1)
            ).ravel()
        )
    t = np.array(columns).T
    d = block_diag(*(np.sqrt(p.pi[k]) * p.root[k] @ basis for k in range(len(p.pi))))
    residual = (
        np.sqrt(p.pi)[:, None]
        * (p.target - (p.root @ basis @ state.coefficients[..., None]).squeeze(-1))
    ).ravel()
    design = np.c_[t, d]
    damping = 0.05
    augmented = np.r_[
        design, np.c_[np.sqrt(damping) * np.eye(q), np.zeros((q, d.shape[1]))]
    ]
    expected = np.linalg.lstsq(augmented, np.r_[residual, np.zeros(q)], rcond=1e-12)[0][
        :q
    ]
    np.testing.assert_allclose(
        np.linalg.solve(h + damping * np.eye(q), score), expected, atol=1e-10
    )


def test_optimizer_only_tangent_preserves_objective_descent():
    _, stats, basis = fixture()
    result = variants.solve(basis.T, stats, "tangent", max_steps=4)
    assert np.max(np.diff(result.diagnostics["profile_history"])) <= 1e-10
    assert result.diagnostics["penalty"]["local_ridge"] == 0


def test_rank_drop_is_not_reported_as_stationary():
    x = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
    stats = variants.calculate_statistics(
        x, x[:, 0], iter([(0, np.ones((2, 2)))]), np.zeros((2, 2, 3))
    )
    basis = np.array([[0.0], [1.0], [0.0]])
    with pytest.raises(RuntimeError, match="nonregular"):
        variants.solve(basis.T, stats, "full0")
