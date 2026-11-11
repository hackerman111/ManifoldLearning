"""Independent checks for the opt-in low-rank Grassmann corrections."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.grassman_accuracy_2026_10_22 import lowrank as lr
from experiments.grassman_accuracy_2026_10_22.candidates import reference

_gradient, _profile = reference._gradient, reference._profile


def _horizontal(Y, X):
    return X - Y @ (Y.T @ X)


@pytest.mark.parametrize("m,k", [(1, 1), (2, 1), (2, 2), (4, 2), (4, 4)])
def test_frozen_correction_matches_dense_action_and_adjoint(m, k):
    rng = np.random.default_rng(3100 + 10 * m + k)
    d, J, p = 8, 5, 3
    Y, _ = np.linalg.qr(rng.normal(size=(d, m)))
    U = rng.normal(size=(J, p, d))
    g = rng.normal(size=(J, m))
    mass = rng.uniform(0.2, 1.5, size=J)
    right = rng.normal(size=(k, m))
    op = lr.FrozenCorrection(Y, U, g, mass, right=right)

    # Form the small dense matrix independently, one coordinate at a time.
    eye = np.eye(d * k)
    dense = np.column_stack([op.matvec(eye[:, j]) for j in range(d * k)])
    x = rng.normal(size=d * k)
    v = rng.normal(size=J * p)
    L = _horizontal(Y, x.reshape(d, k))
    h = g @ right.T
    direct = np.stack(
        [np.sqrt(mass[j]) * (U[j] @ L) @ h[j] for j in range(J)]
    ).ravel()
    assert np.allclose(op.matvec(x), direct, rtol=2e-12, atol=2e-12)
    assert np.allclose(op.matvec(x), dense @ x, rtol=2e-12, atol=2e-12)
    assert np.allclose(op.rmatvec(v), dense.T @ v, rtol=2e-12, atol=2e-12)
    assert np.allclose(np.dot(op.matvec(x), v), np.dot(x, op.rmatvec(v)), atol=2e-12)
    assert np.shares_memory(op.U, U)


def test_frozen_correction_default_right_is_identity():
    rng = np.random.default_rng(411)
    d, m, J, p = 7, 3, 4, 2
    Y, _ = np.linalg.qr(rng.normal(size=(d, m)))
    U, g = rng.normal(size=(J, p, d)), rng.normal(size=(J, m))
    mass, x = rng.uniform(0.1, 2, J), rng.normal(size=d * m)
    implicit = lr.FrozenCorrection(Y, U, g, mass)
    explicit = lr.FrozenCorrection(Y, U, g, mass, right=np.eye(m))
    assert np.allclose(implicit.matvec(x), explicit.matvec(x), atol=1e-12)


def test_coarse_correction_matches_dense_ridge_solve():
    rng = np.random.default_rng(512)
    d, m, k, J, p = 9, 2, 3, 7, 2
    Y, _ = np.linalg.qr(rng.normal(size=(d, m)))
    raw = _horizontal(Y, rng.normal(size=(d, k)))
    V, _ = np.linalg.qr(raw)
    V = _horizontal(Y, V)
    V, _ = np.linalg.qr(V)
    U = rng.normal(size=(J, p, d))
    g, residual = rng.normal(size=(J, m)), rng.normal(size=(J, p))
    mass, damping = rng.uniform(0.2, 1.3, J), 0.07
    Z, info = lr.coarse_correction(Y, U, g, residual, mass, V, damping)

    # Design maps row-major vec(C) to the weighted U V C g predictions.
    A = np.concatenate(
        [
            np.sqrt(mass[j])
            * np.kron((U[j] @ V), g[j][None, :])
            for j in range(J)
        ],
        axis=0,
    )
    b = (np.sqrt(mass)[:, None] * residual).reshape(-1)
    assert info["stop"] == "coarse"
    mu = damping
    expected = np.linalg.solve(A.T @ A + mu * np.eye(k * m), A.T @ b)
    assert Z.shape == (d, m)
    assert np.allclose(Z, V @ expected.reshape(k, m), rtol=2e-8, atol=2e-9)
    assert np.allclose(Y.T @ Z, 0.0, atol=2e-10)


def test_rank_k_tangent_can_represent_full_correction_and_rank_one_cannot():
    rng = np.random.default_rng(613)
    d, m, J, p = 10, 2, 18, 3
    Y, _ = np.linalg.qr(rng.normal(size=(d, m)))
    U = rng.normal(size=(J, p, d))
    g = rng.normal(size=(J, m))
    mass = rng.uniform(0.3, 1.2, J)
    target = _horizontal(Y, rng.normal(size=(d, m)))
    residual = np.stack([(U[j] @ target) @ g[j] for j in range(J)])
    # A full horizontal basis makes the unrestricted least-squares solution
    # attainable when k=m; rank one is strictly restricted for this target.
    from scipy.linalg import null_space

    V = null_space(Y.T)
    Zfull, _ = lr.coarse_correction(Y, U, g, residual, mass, V, 0.0)
    assert np.allclose(Y.T @ Zfull, 0, atol=2e-10)
    assert np.allclose(Zfull, target, rtol=2e-8, atol=2e-9)
    singular = np.linalg.svd(target, compute_uv=False)
    assert np.linalg.matrix_rank(target, tol=1e-8) == m
    assert singular[1] > 1e-8  # no rank-one tangent can equal this target

    I = rng.normal(size=(J, p))
    state = _profile(U @ Y, I, mass, 0.0)
    G = _gradient(Y, U, state, mass)
    z1, _ = lr.correction(Y, U, state, mass, G, mode="lora", rank_k=1)
    zk, _ = lr.correction(Y, U, state, mass, G, mode="lora", rank_k=m)
    assert np.linalg.matrix_rank(z1, tol=1e-8) <= 1
    assert np.linalg.matrix_rank(zk, tol=1e-8) <= m
    assert np.linalg.norm(zk) > 0
    assert np.allclose(Y.T @ z1, 0, atol=2e-9)
    assert np.allclose(Y.T @ zk, 0, atol=2e-9)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_correction_is_equivariant_to_response_and_mass_units(scale):
    rng = np.random.default_rng(714)
    d, m, J, p = 8, 2, 12, 2
    Y, _ = np.linalg.qr(rng.normal(size=(d, m)))
    U = rng.normal(size=(J, p, d))
    mass, I = rng.uniform(0.3, 1.1, J), rng.normal(size=(J, p))
    state = _profile(U @ Y, I, mass, 0.0)
    G = _gradient(Y, U, state, mass)
    base, _ = lr.correction(Y, U, state, mass, G, mode="lora", rank_k=2)

    scaled_state = _profile(U @ Y, I * scale, mass, 0.0)
    scaled_G = _gradient(Y, U, scaled_state, mass)
    changed, _ = lr.correction(
        Y, U, scaled_state, mass, scaled_G, mode="lora", rank_k=2
    )
    assert np.allclose(changed, base, rtol=2e-5, atol=2e-8)


def test_lora_exposes_krylov_stops_and_enrichment_does_not_worsen_frozen_objective():
    rng = np.random.default_rng(765)
    d, m, J, p = 9, 2, 16, 3
    Y, _ = np.linalg.qr(rng.normal(size=(d, m)))
    U, I = rng.normal(size=(J, p, d)), rng.normal(size=(J, p))
    mass = rng.uniform(0.2, 1.3, J)
    state = _profile(U @ Y, I, mass, 0.0)
    G = _gradient(Y, U, state, mass)
    _, lora_info = lr.correction(
        Y, U, state, mass, G, mode="lora", rank_k=1, rounds=2, lsmr_steps=2
    )
    assert lora_info["lsmr"]
    assert all(0 < stop[1] <= 2 for stop in lora_info["lsmr"])
    assert all(len(stop) == 3 for stop in lora_info["lsmr"])

    _, svd_info = lr.correction(Y, U, state, mass, G, mode="svd", rank_k=1)
    _, enriched_info = lr.correction(
        Y, U, state, mass, G, mode="enriched", rank_k=1
    )
    assert enriched_info["objective"] <= svd_info["objective"] + 1e-9


def test_lowrank_solver_variants_keep_profile_and_final_certificate_consistent():
    from experiments.grassman_accuracy_2026_10_22.candidates import solver_for
    rng = np.random.default_rng(815)
    J, p, d, m = 32, 4, 8, 2
    B, _ = np.linalg.qr(rng.normal(size=(d, m)))
    U, I = rng.normal(size=(J, p, d)), rng.normal(size=(J, p))
    mass = rng.uniform(0.3, 1.2, J)
    for name in (
        "lora1_10", "lora_k10", "lora_k20", "svd1_10", "svd_k10",
        "svd_k20", "enriched10", "enriched20",
    ):
        result = solver_for(name)(B.T, U, I, mass=mass, max_steps=3, tol=1e-12)
        Y = result.index.T
        assert np.allclose(Y.T @ Y, np.eye(m), atol=2e-8), name
        state = _profile(U @ Y, I, mass, 0.0)
        grad = _gradient(Y, U, state, mass)
        history = result.diagnostics["profile_history"]
        assert np.all(np.diff(history) <= 1e-10), name
        assert np.isclose(history[-1], state.value, rtol=2e-8, atol=2e-9), name
        reported = result.diagnostics["gradient_norm"]
        assert np.isclose(reported, np.linalg.norm(grad), rtol=2e-7, atol=2e-9), name


def test_correction_reports_workspace_bounded_operator_fallback():
    rng = np.random.default_rng(916)
    d, m, J, p = 12, 2, 10, 2
    Y, _ = np.linalg.qr(rng.normal(size=(d, m)))
    U = rng.normal(size=(J, p, d))
    mass = rng.uniform(0.2, 1.0, J)
    I = rng.normal(size=(J, p))
    state = _profile(U @ Y, I, mass, 0.0)
    G = _gradient(Y, U, state, mass)
    Z, info = lr.correction(
        Y, U, state, mass, G, mode="lora", rank_k=2, workspace_bytes=1
    )
    assert np.all(np.isfinite(Z))
    assert info["stop"] == "workspace"


def test_polar_preserves_frozen_predictions_with_unequal_mass_and_singular_center():
    from scipy.linalg import polar, sqrtm

    rng = np.random.default_rng(1017)
    d, m, J, p = 7, 2, 6, 3
    Y, _ = np.linalg.qr(rng.normal(size=(d, m)))
    U = rng.normal(size=(J, p, d))
    # One zero statistic center and one locally rank-deficient center.
    U[0] = 0
    U[1] = rng.normal(size=(p, 1)) @ rng.normal(size=(1, d))
    assert np.linalg.matrix_rank(U[1] @ Y) < m
    mass = np.geomspace(1e-4, 1e4, J)
    g = rng.normal(size=(J, m))
    residual = rng.normal(size=(J, p))
    I = residual + np.einsum("jpm,jm->jp", U @ Y, g)
    Delta = _horizontal(Y, rng.normal(size=(d, m))) * 0.08
    X = Y + Delta
    Q, _ = polar(X)
    S = np.asarray(sqrtm(X.T @ X)).real
    adjusted = np.einsum("ab,jb->ja", S, g)
    frozen = residual - np.einsum("jpm,jm->jp", U @ Delta, g)
    polar_residual = I - np.einsum("jpm,jm->jp", U @ Q, adjusted)
    assert np.allclose(polar_residual, frozen, rtol=2e-10, atol=2e-10)
    assert np.allclose(
        np.sum(mass[:, None] * polar_residual**2),
        np.sum(mass[:, None] * frozen**2), rtol=2e-10, atol=2e-10,
    )
    profiled = _profile(U @ Q, I, mass, 0.0)
    assert profiled.value <= np.sum(mass[:, None] * frozen**2) + 1e-9


def test_lowrank_wrapper_constant_center_recursion_and_ridge_guard():
    from experiments.grassman_accuracy_2026_10_22.candidates import solver_for

    rng = np.random.default_rng(1118)
    J, p, d, m = 9, 3, 6, 2
    B, _ = np.linalg.qr(rng.normal(size=(d, m)))
    U, I = rng.normal(size=(J, p, d)), rng.normal(size=(J, p))
    U[0] = 0
    mass = np.geomspace(0.1, 10.0, J)
    result = solver_for("svd1_10")(B.T, U, I, mass=mass, tol=1e-12)
    assert result.diagnostics["constant_centers"] == 1
    assert np.allclose(result.coefficients[0], 0)
    history = result.diagnostics["profile_history"]
    assert np.all(np.diff(history) <= 1e-10)
    assert np.isclose(history[-1], result.diagnostics["profile_objective"])

    Y = result.index.T
    state = _profile(U @ Y, I, mass, 0.2)
    G = _gradient(Y, U, state, mass)
    proposal, info, evals = lr.step(
        Y, U, I, state, mass, G, 0.2, 0.5, 1024**2, "svd", 1
    )
    assert proposal is None and info["stop"] == "ridge_guard" and evals == 0


def test_lowrank_public_helpers_reject_strided_statistics_without_copying():
    rng = np.random.default_rng(1219)
    d, m, J, p = 7, 2, 5, 3
    Y, _ = np.linalg.qr(rng.normal(size=(d, m)))
    U = np.asfortranarray(rng.normal(size=(J, p, d)))
    assert not U.flags.c_contiguous
    g, mass = rng.normal(size=(J, m)), np.ones(J)
    V = np.linalg.qr(_horizontal(Y, rng.normal(size=(d, 1))))[0]
    with pytest.raises(ValueError, match="contiguous"):
        lr.FrozenCorrection(Y, U, g, mass)
    with pytest.raises(ValueError, match="contiguous"):
        lr.coarse_correction(Y, U, g, rng.normal(size=(J, p)), mass, V, 0.1)
    state = _profile(np.ascontiguousarray(U) @ Y, rng.normal(size=(J, p)), mass, 0.0)
    G = _gradient(Y, np.ascontiguousarray(U), state, mass)
    with pytest.raises(ValueError, match="contiguous"):
        lr.correction(Y, U, state, mass, G, mode="svd")
