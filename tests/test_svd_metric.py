"""Dense independent references for spectral proximal metrics."""

import numpy as np
import pytest

from ADP import ADP_Config, ADP_multi_index, ADP_solver
from ADP.cli.main import _synthetic_data
from ADP.solver import SVD as svd
from ADP.solver._multi_operator import forward


@pytest.mark.parametrize("target", ["matrix", "correction"])
@pytest.mark.parametrize(
    "power,floor,alpha,tensor",
    [
        (0.5, 0.0, 0.2, "full"),
        (1.0, 0.0, 0.03, "full"),
        (1.0, 0.1, 0.001, "full"),
        (1.0, 0.1, 0.03, "orthogonal"),
    ],
)
@pytest.mark.parametrize("direct", [None, 128])
@pytest.mark.parametrize("search", ["alternating", "gradient"])
def test_metric_dense_whitening_reference(
    target, power, floor, alpha, tensor, direct, search
):
    rng = np.random.default_rng(192)
    m, d, J, p = 3, 8, 12, 4
    P = np.linalg.qr(rng.normal(size=(d, m)))[0].T
    U = rng.normal(size=(J, p, d))
    I, g = rng.normal(size=(J, p)), rng.normal(size=(J, m))
    mass = np.geomspace(0.01, 100, J)
    values = np.array([1.0, 0.4, 0.05])
    kernel = P.T @ np.diag(values) @ P + alpha**2 * (
        np.eye(d) if tensor == "full" else np.eye(d) - P.T @ P
    )
    eig, Q = np.linalg.eigh(kernel)
    spectrum = floor + (1 - floor) * (eig / eig.max()) ** power
    A = (Q * spectrum) @ Q.T
    root = (Q * np.sqrt(spectrum)) @ Q.T
    inverse_root = (Q / np.sqrt(spectrum)) @ Q.T
    kwargs = {
        "rank": 2,
        "lambda_penalty": 0.4,
        "inner_tol": 1e-9,
        "inner_maxiter": 50,
        "rank_tol": 1e-12,
        "lsmr_maxiter": 500,
        "warm_start": True,
        "adaptive_krylov": False,
        "precondition_v": False,
        "direct_max_dimension": direct,
        "low_rank_target": target,
        "rank_one_search": search,
    }
    B, diagnostics = svd.solve_fixed_coefficients(
        P,
        U,
        I,
        g,
        mass,
        metric_power=power,
        metric_floor=floor,
        metric_alpha=alpha,
        metric_eigenvalues=values,
        metric_tensor=tensor,
        **kwargs,
    )
    reference, _, _, _ = svd._solve_fixed_coefficients_validated(
        P @ root, U @ inverse_root, I, g, mass, orthonormal_prior=False, **kwargs
    )
    np.testing.assert_allclose(B, reference @ inverse_root, atol=2e-7, rtol=2e-7)
    delta = B - P
    objective = np.sum(mass[:, None] * (I - forward(U, B, g)) ** 2)
    objective += 0.4 * np.sum((delta @ A) * delta)
    np.testing.assert_allclose(
        diagnostics["rank_objective_history"][-1], objective, rtol=1e-10, atol=1e-8
    )
    assert np.all(np.diff(diagnostics["rank_objective_history"]) <= 1e-8)
    assert np.linalg.matrix_rank(B if target == "matrix" else delta, tol=1e-8) <= 2
    np.testing.assert_allclose(
        svd._metric_action(
            U, P, *svd._metric_spectrum(P, power, floor, alpha, values, tensor), -0.5
        ),
        U @ inverse_root,
        atol=1e-8,
        rtol=1e-9,
    )
    x, y = rng.normal(size=d), rng.normal(size=(J, p))
    transformed = U @ inverse_root
    np.testing.assert_allclose(
        np.sum((transformed @ x) * y),
        x @ np.einsum("jpd,jp->d", transformed, y),
        atol=1e-10,
    )
    if target == "correction":
        np.testing.assert_allclose(
            diagnostics["correction_frobenius_norm"], np.linalg.norm(delta), atol=1e-10
        )


def test_metric_identity_keeps_exact_baseline():
    rng = np.random.default_rng(7)
    P = np.linalg.qr(rng.normal(size=(8, 3)))[0].T
    args = (
        P,
        rng.normal(size=(12, 4, 8)),
        rng.normal(size=(12, 4)),
        rng.normal(size=(12, 3)),
        np.ones(12),
    )
    a, da = svd.solve_fixed_coefficients(*args, rank=2, lambda_penalty=0.2)
    b, db = svd.solve_fixed_coefficients(
        *args, rank=2, lambda_penalty=0.2, metric_power=0, metric_floor=0.1
    )
    np.testing.assert_array_equal(a, b)
    assert da["rank_objective_history"] == db["rank_objective_history"]
    assert db["metric_transform_bytes"] == 0


@pytest.mark.parametrize(
    "extra",
    [
        {"metric_alpha": 0},
        {"metric_alpha": float("nan")},
        {"metric_power": 0.7},
        {"metric_floor": -0.1},
        {"metric_eigenvalues": [1.0, -1.0]},
        {"metric_eigenvalues": [1.0, float("nan")]},
        {"metric_eigenvalues": [2.0, 1.0]},
        {"metric_tensor": "invalid"},
        {"metric_tensor": "orthogonal", "metric_eigenvalues": [1.0, 0.0]},
    ],
)
def test_metric_rejects_invalid_or_singular(extra):
    kwargs = {"metric_power": 1, "metric_alpha": 0.1, "metric_eigenvalues": [1.0, 0.5]}
    kwargs.update(extra)
    with pytest.raises(ValueError):
        svd.solve_fixed_coefficients(
            np.eye(2, 4),
            np.ones((4, 3, 4)),
            np.ones((4, 3)),
            np.ones((4, 2)),
            np.ones(4),
            rank=1,
            lambda_penalty=0.2,
            **kwargs,
        )


@pytest.mark.parametrize("target", ["matrix", "correction"])
def test_public_fit_uses_current_localization_context(target):
    X, y, _ = _synthetic_data(90, 5, 2, 0.05, 17)
    config = ADP_Config(
        N_J=18, N_phi=6, N_loc=20, outer_steps=3, h_min=1e-6, multi_tensor="full"
    )
    model = ADP_multi_index(
        2,
        config,
        ADP_solver(
            svd.solve,
            rank=1,
            inner_maxiter=12,
            low_rank_target=target,
            metric_power=0.5,
            metric_floor=0.1,
        ),
    ).fit(X, y)
    np.testing.assert_allclose(model.basis_.T @ model.basis_, np.eye(2), atol=1e-10)
    for k, row in enumerate(model.trace_):
        diag = row["solver"]
        assert diag["metric_alpha"] == row["factor"]
        assert diag["metric_tensor"] == row["tensor"]
        expected = np.ones(2) if k == 0 else model.trace_[k - 1]["eigenvalues"]
        np.testing.assert_allclose(diag["metric_eigenvalues"], expected)
        assert diag["metric_condition"] <= 10 + 1e-10
        assert diag["metric_transform_bytes"] > 0


@pytest.mark.parametrize("ridge", [0.0, 1e-12, 1e6])
def test_metric_zero_correction_rank_and_extreme_ridge(ridge):
    P = np.eye(2, 5)
    rng = np.random.default_rng(5)
    U, I, g = (
        rng.normal(size=(5, 3, 5)),
        rng.normal(size=(5, 3)),
        rng.normal(size=(5, 2)),
    )
    B, diag = svd.solve_fixed_coefficients(
        P,
        U,
        I,
        g,
        np.ones(5),
        rank=0,
        low_rank_target="correction",
        lambda_penalty=ridge,
        metric_power=1,
        metric_floor=0.1,
        metric_alpha=0.001,
        metric_eigenvalues=np.array([1.0, 0.0]),
    )
    np.testing.assert_allclose(B, P, atol=1e-12)
    assert diag["correction_rank"] == 0
    np.testing.assert_allclose(
        diag["rank_objective_history"],
        [np.sum((I - forward(U, P, g)) ** 2)],
        atol=1e-10,
    )


def test_metric_rejects_complex_spectrum_and_nonorthogonal_public_prior():
    P, U, I = np.eye(2, 5), np.ones((5, 3, 5)), np.ones((5, 3))
    with pytest.raises(TypeError, match="real numeric"):
        svd.solve(
            P,
            U,
            I,
            rank=1,
            metric_power=1,
            metric_alpha=0.1,
            metric_eigenvalues=np.array([1.0, 0.5j]),
        )
    with pytest.raises(ValueError, match="orthonormal incoming"):
        svd.solve(
            2 * P,
            U,
            I,
            rank=1,
            metric_power=1,
            metric_alpha=0.1,
            metric_eigenvalues=np.array([1.0, 0.5]),
        )


def test_metric_whitening_keeps_one_contiguous_u_buffer():
    rng = np.random.default_rng(6)
    U = np.asfortranarray(rng.normal(size=(7, 4, 8)))
    _, diag = svd.solve_fixed_coefficients(
        np.eye(2, 8),
        U,
        rng.normal(size=(7, 4)),
        rng.normal(size=(7, 2)),
        np.ones(7),
        rank=1,
        lambda_penalty=0.2,
        metric_power=0.5,
        metric_alpha=0.2,
        metric_eigenvalues=np.array([1.0, 0.5]),
    )
    assert not diag["u_flat_copy"]
    assert diag["metric_transform_bytes"] == U.nbytes
