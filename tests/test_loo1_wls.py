"""TDD verification suite for LOO1-WLS-ADP (estimator='loo1_wls')."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest
from scipy.linalg import subspace_angles

from ADP import (
    ADP_Config,
    ADP_multi_index,
    ADP_single_index,
    LOO1WLSConfig,
    calculate_alpha_k,
    epanechnikov,
    pairwise_distance2,
    search_bandwidth,
)
from ADP.engine.common.initialize import initialize_basis_local_with_spectrum
from ADP.engine.common.loo1_wls import (
    LOO1Neighborhoods,
    build_loo1_neighborhoods,
    calculate_alpha_k_streaming,
    evaluate_loo1_wls,
    initialize_basis_local_streaming,
    pairwise_distance2_blocks,
    search_bandwidth_streaming,
    subcritical_eigenvalues,
)
from ADP.engine.common.weights import calculate_multi_weight
from ADP.solver.loo1_wls import precondition_gradient, solve as solve_loo1_wls
from ADP.solver.LSMR import HPAOResult


def _edr(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Compute Grassmann subspace distance sqrt(mean(sin(theta)^2))."""
    q_est = np.linalg.qr(estimate)[0]
    q_true = np.linalg.qr(truth)[0]
    angles = subspace_angles(q_est, q_true)
    return float(np.sqrt(np.mean(np.square(np.sin(angles)))))


def _orthonormal_basis(rng: np.random.Generator, d: int, m: int) -> np.ndarray:
    """Return a (d, m) column-orthonormal matrix."""
    q, _ = np.linalg.qr(rng.standard_normal((d, m)), mode="reduced")
    return q


def _gaussian_kernel(value: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.asarray(value, dtype=float))


def _naive_reference_loo1(
    Y: np.ndarray,
    X: np.ndarray,
    neighborhoods: LOO1Neighborhoods,
    *,
    ridge: float,
    leverage_cap: float,
    smooth_cap_eps: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Independent per-center loop reference for Theorem 1.3 and chunk equivalence."""
    _, d = X.shape
    _, m = Y.shape
    obj = 0.0
    grad_E = np.zeros((d, m), dtype=float)
    active_coeffs = np.zeros((len(neighborhoods.active_centers), m), dtype=float)
    active_mass = np.zeros(len(neighborhoods.active_centers), dtype=float)

    for idx in range(len(neighborhoods.active_centers)):
        s = int(neighborhoods.indptr[idx])
        e = int(neighborhoods.indptr[idx + 1])
        cols = neighborhoods.col_idx[s:e]
        a = neighborhoods.a[s:e]
        yc = neighborhoods.yc[s:e]
        mass_j = float(neighborhoods.mass[idx])
        active_mass[idx] = mass_j

        X_loc = X[cols]
        z_raw = X_loc @ Y
        z_mean = a @ z_raw
        Z = z_raw - z_mean[None, :]

        G = (Z.T * a[None, :]) @ Z + ridge * np.eye(m)
        rhs = Z.T @ (a * yc)
        c = np.linalg.solve(G, rhs)
        active_coeffs[idx] = c

        r = yc - Z @ c
        U = np.linalg.solve(G, Z.T).T  # (k, m): rows are G^{-1} z_i^c
        lev = a + a * np.sum(Z * U, axis=1)

        if smooth_cap_eps == 0.0:
            ell = np.minimum(lev, leverage_cap)
            d_ell = (lev < leverage_cap).astype(float)
        else:
            arg = (leverage_cap - lev) / smooth_cap_eps
            softplus = np.logaddexp(0.0, arg)
            ell = leverage_cap - smooth_cap_eps * softplus
            d_ell = 1.0 / (1.0 + np.exp(-arg))

        denom = 1.0 - ell
        obj += mass_j * float(np.sum(a * np.square(r) / denom))

        omega = a / denom
        r_bar = 2.0 * omega * r
        h_bar = (a * np.square(r) / np.square(denom)) * d_ell
        c_bar = -(Z.T @ r_bar)
        v = np.linalg.solve(G, c_bar)

        M = (U.T * (a * h_bar)[None, :]) @ U
        dZ = (
            -np.outer(r_bar, c)
            + np.outer(a * r, v)
            - np.outer(a * (Z @ v), c)
            + 2.0 * (a * h_bar)[:, None] * U
            - 2.0 * (a[:, None] * (Z @ M))
        )
        dz_raw = dZ - np.outer(a, dZ.sum(axis=0))
        grad_E += X_loc.T @ (mass_j * dz_raw)

    return obj, grad_E, active_coeffs, active_mass


# ---------------------------------------------------------------------------
# 1. Config & API boundary validation
# ---------------------------------------------------------------------------


def test_loo1_wls_config_defaults_frozen_and_slots() -> None:
    cfg = LOO1WLSConfig()
    assert cfg.spectrum_exponent == 0.5
    assert cfg.ridge == 1e-4
    assert cfg.leverage_cap == 0.85
    assert cfg.smooth_cap_eps == 0.0
    assert cfg.preconditioner_reg == 1e-3
    assert cfg.min_neighbors is None
    assert cfg.center_chunk == 512
    assert cfg.stream_distances is None
    assert cfg.stream_distance_threshold == 4_000_000
    assert cfg.max_steps == 25
    assert cfg.tol == 1e-6
    assert cfg.armijo_c1 == 1e-4
    assert cfg.max_line_search == 14

    assert hasattr(LOO1WLSConfig, "__slots__")
    assert len(fields(LOO1WLSConfig)) == 13
    with pytest.raises(FrozenInstanceError):
        cfg.ridge = 0.1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"spectrum_exponent": "0.5"}, TypeError),
        ({"spectrum_exponent": True}, TypeError),
        ({"spectrum_exponent": -0.1}, ValueError),
        ({"spectrum_exponent": float("nan")}, ValueError),
        ({"spectrum_exponent": float("inf")}, ValueError),
        ({"ridge": "1e-4"}, TypeError),
        ({"ridge": False}, TypeError),
        ({"ridge": -1e-6}, ValueError),
        ({"ridge": float("nan")}, ValueError),
        ({"leverage_cap": "0.85"}, TypeError),
        ({"leverage_cap": True}, TypeError),
        ({"leverage_cap": 0.0}, ValueError),
        ({"leverage_cap": 1.0}, ValueError),
        ({"leverage_cap": 1.2}, ValueError),
        ({"smooth_cap_eps": True}, TypeError),
        ({"smooth_cap_eps": -0.01}, ValueError),
        ({"preconditioner_reg": True}, TypeError),
        ({"preconditioner_reg": 0.0}, ValueError),
        ({"preconditioner_reg": -1e-3}, ValueError),
        ({"min_neighbors": True}, TypeError),
        ({"min_neighbors": 2.5}, TypeError),
        ({"min_neighbors": 0}, ValueError),
        ({"center_chunk": True}, TypeError),
        ({"center_chunk": 1.5}, TypeError),
        ({"center_chunk": 0}, ValueError),
        ({"stream_distances": 1}, TypeError),
        ({"stream_distances": "yes"}, TypeError),
        ({"stream_distance_threshold": True}, TypeError),
        ({"stream_distance_threshold": 0}, ValueError),
        ({"max_steps": True}, TypeError),
        ({"max_steps": 0}, ValueError),
        ({"tol": True}, TypeError),
        ({"tol": 0.0}, ValueError),
        ({"armijo_c1": True}, TypeError),
        ({"armijo_c1": 0.0}, ValueError),
        ({"armijo_c1": 1.0}, ValueError),
        ({"max_line_search": True}, TypeError),
        ({"max_line_search": 0}, ValueError),
    ],
)
def test_loo1_wls_config_validation(
    kwargs: dict[str, object], exc: type[Exception]
) -> None:
    with pytest.raises(exc):
        LOO1WLSConfig(**kwargs)  # type: ignore[arg-type]


def test_adp_config_loo1_wls_validation_and_unsupported_modes() -> None:
    cfg = ADP_Config(estimator="loo1_wls")
    assert cfg.estimator == "loo1_wls"
    assert cfg.loo1_wls is None

    custom = LOO1WLSConfig(spectrum_exponent=2.0 / 3.0)
    cfg2 = ADP_Config(estimator="loo1_wls", loo1_wls=custom)
    assert cfg2.loo1_wls == custom

    with pytest.raises(ValueError):
        ADP_Config(estimator="new", loo1_wls=LOO1WLSConfig())
    with pytest.raises(ValueError):
        ADP_Config(estimator="legacy", loo1_wls=LOO1WLSConfig())
    with pytest.raises(TypeError):
        ADP_Config(estimator="loo1_wls", loo1_wls="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ADP_Config(estimator="unknown")

    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 4))
    y = X[:, 0] + 0.1 * rng.standard_normal(40)

    with pytest.raises(NotImplementedError):
        ADP_single_index(ADP_Config(estimator="loo1_wls")).fit(X, y)

    with pytest.raises(NotImplementedError):
        ADP_multi_index(2, ADP_Config(estimator="loo1_wls", gpu=True)).fit(X, y)


def test_build_neighborhoods_empty_raises_runtime_error() -> None:
    rng = np.random.default_rng(2)
    X = rng.standard_normal((12, 4))
    Y = rng.standard_normal(12)
    weights = [(0, np.zeros((3, 12), dtype=float))]
    with pytest.raises(RuntimeError):
        build_loo1_neighborhoods(X, Y, iter(weights), min_neighbors=3, total_centers=3)


# ---------------------------------------------------------------------------
# 2. Theorem 1.1': Exact 3-term regularized & clipped heteroskedastic identity
# ---------------------------------------------------------------------------


def test_theorem_1_1_prime_exact_three_term_identity_and_boxcar_zero_bias() -> None:
    rng = np.random.default_rng(11)
    k, d, m = 18, 6, 3
    X_loc = rng.standard_normal((k, d))
    Y = _orthonormal_basis(rng, d, m)
    Z_raw = X_loc @ Y

    # Case A: arbitrary non-uniform weights, heteroskedastic sigma_i^2, ridge > 0,
    # and leverage cap chosen tightly so clipping is genuinely active.
    raw_w = rng.uniform(0.2, 2.5, size=k)
    raw_w[0] *= 5.0  # induce high leverage on observation 0
    a = raw_w / raw_w.sum()
    sigma2 = rng.uniform(0.1, 1.8, size=k)
    tau = 0.15
    ell_max = 0.35

    Z = Z_raw - (a @ Z_raw)[None, :]
    G = (Z.T * a[None, :]) @ Z + tau * np.eye(m)
    G_inv = np.linalg.inv(G)
    # Full hat matrix H = 1 a^T + Z G^{-1} Z^T diag(a)
    H = np.outer(np.ones(k), a) + (Z @ G_inv @ Z.T) * a[None, :]
    h_diag = np.diag(H)
    assert np.any(h_diag > ell_max), "clipping must be active in Theorem 1.1' test"
    h_tilde = np.minimum(h_diag, ell_max)

    # Exact noise covariance of residual eta = (I - H) eps:
    # E[eta_i^2] = sum_l (I - H)_{il}^2 sigma_l^2.
    I_minus_H = np.eye(k) - H
    exp_eta2 = np.square(I_minus_H) @ sigma2

    lhs = float(np.sum(a * exp_eta2 / (1.0 - h_tilde)) - np.sum(a * sigma2))

    # 3-term identity RHS:
    # Term 1 (clipping): - sum_i a_i sigma_i^2 (h_i - h_tilde_i) / (1 - h_tilde_i)
    term1_clip = -float(np.sum(a * sigma2 * (h_diag - h_tilde) / (1.0 - h_tilde)))
    # Term 2 (ridge): - tau sum_i a_i^2 sigma_i^2 ||G^{-1} z_i^c||^2
    GZ = Z @ G_inv
    term2_ridge = -tau * float(
        np.sum(np.square(a) * sigma2 * np.sum(np.square(GZ), axis=1))
    )
    # Term 3 (O(l_max^2) remainder):
    # sum_i a_i [h_tilde_i / (1 - h_tilde_i)] (sum_l H_{il}^2 sigma_l^2 - sigma_i^2 h_i)
    sum_H2_sigma = np.square(H) @ sigma2
    term3_rem = float(
        np.sum(a * (h_tilde / (1.0 - h_tilde)) * (sum_H2_sigma - sigma2 * h_diag))
    )
    rhs = term1_clip + term2_ridge + term3_rem
    assert abs(lhs - rhs) < 1e-13

    # Test Case B: Boxcar weights (a_i = 1/k), homoskedastic noise, tau = 0,
    # unclipped -> exact zero bias (< 1e-14).
    a_box = np.full(k, 1.0 / k)
    sigma2_hom = np.full(k, 0.75)
    Z_box = Z_raw - Z_raw.mean(axis=0, keepdims=True)
    G_box = (Z_box.T * a_box[None, :]) @ Z_box
    H_box = (
        np.outer(np.ones(k), a_box)
        + (Z_box @ np.linalg.solve(G_box, Z_box.T)) * a_box[None, :]
    )
    h_box = np.diag(H_box)
    assert np.max(h_box) < 0.85
    exp_eta2_box = np.square(np.eye(k) - H_box) @ sigma2_hom
    bias_box = float(
        np.sum(a_box * exp_eta2_box / (1.0 - h_box)) - np.sum(a_box * sigma2_hom)
    )
    assert abs(bias_box) < 1e-14


# ---------------------------------------------------------------------------
# 3. Theorem 1.3: O(m)-invariance, FD Riemannian gradients, chunk equivalence
# ---------------------------------------------------------------------------


def _make_sample_neighborhoods(
    seed: int = 21, *, n: int = 45, d: int = 6, m: int = 3, n_centers: int = 15
) -> tuple[np.ndarray, np.ndarray, LOO1Neighborhoods]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    truth = _orthonormal_basis(rng, d, m)
    z = X @ truth
    y = (
        np.sin(z[:, 0])
        + 0.4 * z[:, 1] ** 2
        + 0.25 * z[:, 2]
        + 0.1 * rng.standard_normal(n)
    )
    centers = X[:n_centers].copy()
    Y0 = _orthonormal_basis(rng, d, m)
    weights = calculate_multi_weight(
        X,
        centers,
        Y0,
        np.ones(m),
        h=2.8,
        alpha=0.9,
        kernel=epanechnikov,
        block_size=5,
        tensor="orthogonal",
    )
    nbh = build_loo1_neighborhoods(
        X, y, weights, min_neighbors=m + 2, total_centers=n_centers
    )
    return X, Y0, nbh


def test_theorem_1_3_om_invariance() -> None:
    X, Y, nbh = _make_sample_neighborhoods(seed=22)
    rng = np.random.default_rng(23)
    Q, _ = np.linalg.qr(rng.standard_normal((Y.shape[1], Y.shape[1])))

    obj1, grad1, coeffs1, mass1, info1 = evaluate_loo1_wls(
        Y, X, nbh, ridge=1e-3, leverage_cap=0.8, smooth_cap_eps=0.0, center_chunk=5
    )
    obj2, grad2, coeffs2, mass2, info2 = evaluate_loo1_wls(
        Y @ Q, X, nbh, ridge=1e-3, leverage_cap=0.8, smooth_cap_eps=0.0, center_chunk=5
    )

    assert abs(obj1 - obj2) < 1e-14
    np.testing.assert_allclose(mass1, mass2, atol=1e-14)
    np.testing.assert_allclose(grad1 @ Q, grad2, atol=1e-13)
    np.testing.assert_allclose(coeffs1 @ Q, coeffs2, atol=1e-13)
    assert info1["active_centers"] == info2["active_centers"]
    assert info1["nnz_weights"] == info2["nnz_weights"]


@pytest.mark.parametrize("smooth_cap_eps", [0.0, 0.02])
def test_theorem_1_3_fd_riemannian_gradient_and_chunk_equivalence(
    smooth_cap_eps: float,
) -> None:
    X, Y, nbh = _make_sample_neighborhoods(seed=31)
    ridge = 1e-3
    # For hard cap (smooth_cap_eps == 0.0), choose cap 0.55 so clipping is active
    # yet away from exact kinks at Y.
    leverage_cap = 0.55

    obj, grad_E, active_coeffs, active_mass, info = evaluate_loo1_wls(
        Y,
        X,
        nbh,
        ridge=ridge,
        leverage_cap=leverage_cap,
        smooth_cap_eps=smooth_cap_eps,
        center_chunk=5,
    )
    assert set(info.keys()) >= {
        "max_leverage",
        "clipped_fraction",
        "active_centers",
        "nnz_weights",
    }
    assert info["active_centers"] == len(nbh.active_centers)
    assert info["nnz_weights"] == len(nbh.col_idx)
    assert info["clipped_fraction"] > 0.0

    # Equivalence against independent naive per-center reference loop
    ref_obj, ref_grad_E, ref_coeffs, ref_mass = _naive_reference_loo1(
        Y,
        X,
        nbh,
        ridge=ridge,
        leverage_cap=leverage_cap,
        smooth_cap_eps=smooth_cap_eps,
    )
    assert abs(obj - ref_obj) < 1e-12
    np.testing.assert_allclose(grad_E, ref_grad_E, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(active_coeffs, ref_coeffs, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(active_mass, ref_mass, atol=1e-12, rtol=1e-12)

    # Central FD Riemannian gradient check along 4 random tangent directions
    grad_R = grad_E - Y @ (Y.T @ grad_E)
    rng = np.random.default_rng(32)
    delta = 1e-6
    for _ in range(4):
        raw = rng.standard_normal(Y.shape)
        tangent = raw - Y @ (Y.T @ raw)
        tangent /= np.linalg.norm(tangent)

        def _retract(step: float, tangent_dir: np.ndarray = tangent) -> np.ndarray:
            u, _, vt = np.linalg.svd(Y + step * tangent_dir, full_matrices=False)
            return u @ vt

        f_plus = evaluate_loo1_wls(
            _retract(delta),
            X,
            nbh,
            ridge=ridge,
            leverage_cap=leverage_cap,
            smooth_cap_eps=smooth_cap_eps,
            center_chunk=5,
        )[0]
        f_minus = evaluate_loo1_wls(
            _retract(-delta),
            X,
            nbh,
            ridge=ridge,
            leverage_cap=leverage_cap,
            smooth_cap_eps=smooth_cap_eps,
            center_chunk=5,
        )[0]
        fd_deriv = (f_plus - f_minus) / (2.0 * delta)
        analytic_deriv = float(np.sum(grad_R * tangent))
        assert abs(fd_deriv - analytic_deriv) < 1e-8


# ---------------------------------------------------------------------------
# 4. Theorem 2.1: Subcritical eigenvalues & 1D quadratic phase transition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exponent", [0.0, 0.5, 2.0 / 3.0, 2.0])
def test_subcritical_eigenvalues_values_and_orthonormality(exponent: float) -> None:
    rng = np.random.default_rng(41)
    coefficients = rng.standard_normal((25, 3))
    mass = rng.uniform(0.5, 3.0, size=25)

    vt, eigenvalues = subcritical_eigenvalues(coefficients, mass, exponent=exponent)
    assert vt.shape == (3, 3)
    assert eigenvalues.shape == (3,)
    np.testing.assert_allclose(vt @ vt.T, np.eye(3), atol=1e-14)
    assert eigenvalues[0] == pytest.approx(1.0, abs=1e-14)

    weighted = np.sqrt(mass)[:, None] * coefficients
    _, s, ref_vt = np.linalg.svd(weighted, full_matrices=False)
    expected_eigs = (
        np.ones(3)
        if exponent == 0.0
        else np.power(np.maximum(s / s[0], 1e-12), exponent)
    )
    np.testing.assert_allclose(eigenvalues, expected_eigs, atol=1e-13)
    # Check subspace alignment with SVD right singular vectors
    np.testing.assert_allclose(
        vt.T @ np.diag(eigenvalues) @ vt,
        ref_vt.T @ np.diag(expected_eigs) @ ref_vt,
        atol=1e-13,
    )


def test_theorem_2_1_quadratic_phase_transition() -> None:
    """1D weak-direction ratio map r_{k+1} = c * ell_{k,2}:
    beta=0.5 makes 0 repelling (non-collapse to fixed point r*=c^2, ell*=c),
    while beta=2.0 makes 0 attracting (quadratic collapse to ~0).
    """
    c = 0.7
    mass = np.array([1.0, 1.0])

    # beta = 0.5 starting near zero (ell_0 = 1e-5) -> repelled to r*=c^2, ell*=c
    ell_sub = np.array([1.0, 1e-5])
    r_next = 0.0
    for _ in range(30):
        r_next = c * ell_sub[1]
        coeffs = np.diag([1.0, r_next])
        _, ell_sub = subcritical_eigenvalues(coeffs, mass, exponent=0.5)
    assert ell_sub[1] == pytest.approx(c, rel=1e-5)
    assert r_next == pytest.approx(c**2, rel=1e-5)

    # beta = 2.0 starting from r_0 = 0.5 -> collapses to floor
    ell_quad = np.array([1.0, 0.5])
    for _ in range(15):
        r_next = c * ell_quad[1]
        coeffs = np.diag([1.0, r_next])
        _, ell_quad = subcritical_eigenvalues(coeffs, mass, exponent=2.0)
    assert ell_quad[1] < 1e-10


# ---------------------------------------------------------------------------
# 5. Condition-number guard on precondition_gradient
# ---------------------------------------------------------------------------


def test_precondition_gradient_condition_number_guard() -> None:
    rng = np.random.default_rng(51)
    d, m = 7, 3
    grad_R = rng.standard_normal((d, m))
    # Construct coefficients and mass so R = sum_j N_j c_j c_j^T = diag(3, 0, 0).
    coefficients = np.array([[math.sqrt(3.0), 0.0, 0.0], [0.0, 0.0, 0.0]])
    mass = np.array([1.0, 2.0])

    reg = 1e-5
    direction, cond_bound = precondition_gradient(
        grad_R, coefficients, mass, preconditioner_reg=reg
    )
    assert cond_bound <= 1.0 + 1.0 / reg + 1e-6
    assert cond_bound == pytest.approx(100_001.0, rel=1e-9, abs=1e-6)

    tau = reg * 3.0  # reg * max(1.0, tr(R))
    R_reg = np.diag([3.0 + tau, tau, tau])
    expected_dir = grad_R @ np.linalg.inv(R_reg)
    np.testing.assert_allclose(direction, expected_dir, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# 6. Streaming distance2-free exactness
# ---------------------------------------------------------------------------


def test_streaming_helpers_match_dense_to_1e11() -> None:
    rng = np.random.default_rng(61)
    n, d, m, n_centers = 85, 6, 3, 25
    # Large offset exercises shifted Gram stability and near-zero self-distances.
    X = rng.standard_normal((n, d)) + 1e4
    truth = _orthonormal_basis(rng, d, m)
    z = X @ truth
    Y = (
        np.sin(z[:, 0])
        + 0.5 * z[:, 1] ** 2
        + 0.3 * z[:, 2]
        + 0.05 * rng.standard_normal(n)
    )
    centers = X[:n_centers].copy()
    centers[0] = X[0]  # exact zero distance check

    d2_dense = pairwise_distance2(X, centers)
    blocks = list(pairwise_distance2_blocks(X, centers, block_size=7))
    reconstructed = np.vstack([blk for _, blk in blocks])
    assert [start for start, _ in blocks] == list(range(0, n_centers, 7))
    np.testing.assert_allclose(reconstructed, d2_dense, atol=1e-12, rtol=1e-12)

    for kernel in (epanechnikov, _gaussian_kernel):
        h_dense = search_bandwidth(d2_dense, 18.0, kernel, lower=0.5)
        h_stream = search_bandwidth_streaming(
            X, centers, 18.0, kernel, lower=0.5, block_size=7
        )
        assert abs(h_dense - h_stream) < 1e-11

        eigs = np.array([1.0, 0.7, 0.4])
        for tensor in ("orthogonal", "full"):
            alpha_dense = calculate_alpha_k(
                X,
                centers,
                truth,
                eigs,
                h_dense,
                12,
                kernel,
                distance2=d2_dense,
                tensor=tensor,
            )
            alpha_stream = calculate_alpha_k_streaming(
                X,
                centers,
                truth,
                eigs,
                h_dense,
                12,
                kernel,
                tensor=tensor,
                block_size=7,
            )
            assert alpha_dense is not None and alpha_stream is not None
            assert abs(alpha_dense - alpha_stream) < 1e-11

    for mass_weighted in (False, True):
        for ridge_sel in ("fixed", "loo"):
            b_dense, s_dense = initialize_basis_local_with_spectrum(
                X,
                Y,
                centers,
                d2_dense,
                16,
                epanechnikov,
                1e-4,
                m,
                mass_weighted=mass_weighted,
                ridge_selection=ridge_sel,
            )
            b_stream, s_stream = initialize_basis_local_streaming(
                X,
                Y,
                centers,
                16,
                epanechnikov,
                1e-4,
                m,
                mass_weighted=mass_weighted,
                ridge_selection=ridge_sel,
                block_size=7,
            )
            np.testing.assert_allclose(b_stream, b_dense, atol=1e-11, rtol=1e-11)
            np.testing.assert_allclose(s_stream, s_dense, atol=1e-11, rtol=1e-11)


def test_full_model_streaming_matches_dense_to_1e11() -> None:
    rng = np.random.default_rng(62)
    X = rng.standard_normal((120, 6))
    truth = _orthonormal_basis(rng, 6, 2)
    z = X @ truth
    y = np.sin(z[:, 0]) + 0.5 * z[:, 1] ** 2 + 0.1 * rng.standard_normal(120)

    cfg_dense = ADP_Config(
        seed=7,
        N_J=40,
        N_loc=20,
        N_lin=15,
        h_min=0.6,
        outer_steps=3,
        estimator="loo1_wls",
        loo1_wls=LOO1WLSConfig(stream_distances=False, center_chunk=13),
    )
    cfg_stream = ADP_Config(
        seed=7,
        N_J=40,
        N_loc=20,
        N_lin=15,
        h_min=0.6,
        outer_steps=3,
        estimator="loo1_wls",
        loo1_wls=LOO1WLSConfig(stream_distances=True, center_chunk=13),
    )
    m_dense = ADP_multi_index(2, cfg_dense).fit(X, y)
    m_stream = ADP_multi_index(2, cfg_stream).fit(X, y)

    np.testing.assert_allclose(m_stream.beta_, m_dense.beta_, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(
        m_stream.eigenvalues_, m_dense.eigenvalues_, atol=1e-11, rtol=1e-11
    )
    np.testing.assert_allclose(
        m_stream.coefficients_, m_dense.coefficients_, atol=1e-11, rtol=1e-11
    )
    assert len(m_stream.trace_) == len(m_dense.trace_)
    for r_s, r_d in zip(m_stream.trace_, m_dense.trace_, strict=True):
        assert abs(float(r_s["err"]) - float(r_d["err"])) < 1e-11
        assert abs(float(r_s["h"]) - float(r_d["h"])) < 1e-11


# ---------------------------------------------------------------------------
# 7. Solver direct contract & end-to-end ADP_multi_index tests
# ---------------------------------------------------------------------------


def test_solve_loo1_wls_contract_with_inactive_centers() -> None:
    rng = np.random.default_rng(71)
    n, d, m, n_centers = 50, 5, 2, 12
    X = rng.standard_normal((n, d))
    truth = _orthonormal_basis(rng, d, m)
    z = X @ truth
    y = np.sin(z[:, 0]) + 0.4 * z[:, 1] ** 2 + 0.05 * rng.standard_normal(n)

    # Center 0 has zero weights (inactive); remaining centers stay active
    centers = X[:n_centers].copy()
    w_blocks = list(
        calculate_multi_weight(
            X,
            centers,
            truth,
            np.ones(m),
            h=2.5,
            alpha=1.0,
            kernel=epanechnikov,
            block_size=6,
        )
    )
    w_blocks[0][1][0, :] = 0.0
    nbh = build_loo1_neighborhoods(
        X, y, iter(w_blocks), min_neighbors=m + 2, total_centers=n_centers
    )
    assert len(nbh.active_centers) == n_centers - 1

    init = _orthonormal_basis(rng, d, m).T  # (m, d) row-orthonormal
    res = solve_loo1_wls(init, X, nbh, config=LOO1WLSConfig(max_steps=10))
    assert isinstance(res, HPAOResult)
    assert res.index.shape == (m, d)
    np.testing.assert_allclose(res.index @ res.index.T, np.eye(m), atol=1e-12)
    assert res.coefficients.shape == (n_centers, m)
    np.testing.assert_array_equal(res.coefficients[0], np.zeros(m))


@pytest.mark.parametrize("m", [1, 2, 3])
@pytest.mark.parametrize("training_set", ["all", "exclude_centers"])
@pytest.mark.parametrize("select_step", ["best", "last"])
@pytest.mark.parametrize("index_init", ["local", "random"])
def test_end_to_end_loo1_wls_configurations(
    m: int,
    training_set: str,
    select_step: str,
    index_init: str,
) -> None:
    rng = np.random.default_rng(100 + m)
    n, d = 140, 6
    X = rng.standard_normal((n, d))
    truth = _orthonormal_basis(rng, d, m)
    z = X @ truth
    y = sum(np.sin(z[:, q]) if q % 2 == 0 else 0.5 * z[:, q] ** 2 for q in range(m))
    y += 0.1 * rng.standard_normal(n)

    cfg = ADP_Config(
        seed=12,
        N_J=45,
        N_loc=22,
        N_lin=18,
        h_min=0.7,
        outer_steps=2,
        estimator="loo1_wls",
        training_set=training_set,
        select_step=select_step,
        index_init=index_init,
        center_displacement=0.1,
        loo1_wls=LOO1WLSConfig(max_steps=8),
    )
    model = ADP_multi_index(m, cfg).fit(X, y)
    assert model.beta_.shape == (d, m)
    np.testing.assert_allclose(model.beta_.T @ model.beta_, np.eye(m), atol=1e-11)
    assert model.coefficients_.shape == (45, m)
    assert model.eigenvalues_.shape == (m,)
    assert model.eigenvalues_[0] == pytest.approx(1.0, abs=1e-12)
    assert len(model.trace_) >= 1
    assert model.effective_parameters_["N_phi"] == 0
    assert model.transform(X).shape == (n, m)


def test_subspace_recovery_linear_weak_and_mixture() -> None:
    # 1) Linear (m=1, d=8)
    rng_lin = np.random.default_rng(201)
    X_lin = rng_lin.standard_normal((300, 8))
    truth_lin = _orthonormal_basis(rng_lin, 8, 1)
    y_lin = (X_lin @ truth_lin)[:, 0] + 0.15 * rng_lin.standard_normal(300)
    m_lin = ADP_multi_index(
        1,
        ADP_Config(
            seed=201,
            N_J=180,
            N_loc=25,
            N_lin=20,
            h_min=0.5,
            outer_steps=4,
            estimator="loo1_wls",
        ),
    ).fit(X_lin, y_lin)
    assert _edr(m_lin.beta_, truth_lin) < 0.15

    # 2) Weak(8, 2): strong first index + weak symmetric quadratic second index
    rng_weak = np.random.default_rng(202)
    X_weak = rng_weak.standard_normal((500, 8))
    truth_weak = _orthonormal_basis(rng_weak, 8, 2)
    zw = X_weak @ truth_weak
    y_weak = (
        np.sin(zw[:, 0])
        + 0.55 * (zw[:, 1] ** 2 - 1.0)
        + 0.12 * rng_weak.standard_normal(500)
    )
    m_weak = ADP_multi_index(
        2,
        ADP_Config(
            seed=202,
            N_J=250,
            N_loc=30,
            N_lin=24,
            h_min=0.5,
            outer_steps=5,
            estimator="loo1_wls",
        ),
    ).fit(X_weak, y_weak)
    assert _edr(m_weak.beta_, truth_weak) < 0.15

    # 3) Mixture(8, 2): non-Gaussian bimodal symmetric mixture X
    rng_mix = np.random.default_rng(203)
    X_mix = (
        rng_mix.standard_normal((500, 8)) + rng_mix.choice([-1.0, 1.0], size=(500, 8))
    ) / math.sqrt(2.0)
    truth_mix = _orthonormal_basis(rng_mix, 8, 2)
    zm = X_mix @ truth_mix
    y_mix = (
        np.sin(zm[:, 0])
        + 0.6 * (zm[:, 1] ** 2 - 1.0)
        + 0.12 * rng_mix.standard_normal(500)
    )
    m_mix = ADP_multi_index(
        2,
        ADP_Config(
            seed=203,
            N_J=250,
            N_loc=30,
            N_lin=24,
            h_min=0.5,
            outer_steps=5,
            estimator="loo1_wls",
        ),
    ).fit(X_mix, y_mix)
    assert _edr(m_mix.beta_, truth_mix) < 0.15


def test_estimator_new_backward_compatibility() -> None:
    rng = np.random.default_rng(301)
    X = rng.standard_normal((120, 6))
    truth = _orthonormal_basis(rng, 6, 2)
    z = X @ truth
    y = np.sin(z[:, 0]) + 0.5 * z[:, 1] ** 2 + 0.1 * rng.standard_normal(120)

    default_cfg = ADP_Config(seed=301, N_J=35, N_phi=5, N_loc=18, outer_steps=2)
    explicit_new = ADP_Config(
        seed=301, N_J=35, N_phi=5, N_loc=18, outer_steps=2, estimator="new"
    )
    assert default_cfg.estimator == "new"
    assert default_cfg.loo1_wls is None

    m1 = ADP_multi_index(2, default_cfg).fit(X, y)
    m2 = ADP_multi_index(2, explicit_new).fit(X, y)
    np.testing.assert_array_equal(m1.beta_, m2.beta_)
    np.testing.assert_array_equal(m1.eigenvalues_, m2.eigenvalues_)
    np.testing.assert_array_equal(m1.coefficients_, m2.coefficients_)
