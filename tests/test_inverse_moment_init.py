from __future__ import annotations

import numpy as np
import pytest

from experiments.inverse_moment_init import (
    _slice_operator,
    initialize_basis_inverse_moments,
)


def _dense_moments(Z, y, slices=10):
    result = np.zeros((Z.shape[1], Z.shape[1]))
    for indices in np.array_split(np.argsort(y, kind="stable"), slices):
        group = Z[indices]
        mean = group.mean(axis=0)
        covariance = (group - mean).T @ (group - mean) / len(group)
        remainder = np.eye(Z.shape[1]) - covariance
        result += len(group) / len(Z) * (np.outer(mean, mean) + remainder @ remainder)
    return result


def test_inverse_operator_matches_dense_and_adjoint():
    rng = np.random.default_rng(7190)
    Z = rng.normal(size=(205, 8))
    y = Z[:, 0] ** 2 + Z[:, 1]
    operator = _slice_operator(Z, y, 10)
    expected = _dense_moments(Z, y)
    x, v = rng.normal(size=(2, 8))
    np.testing.assert_allclose(operator @ x, expected @ x, atol=2e-14)
    np.testing.assert_allclose(x @ operator.rmatvec(v), v @ (operator @ x), atol=2e-14)
    assert np.linalg.eigvalsh(expected)[0] >= -1e-12


def test_inverse_initializer_matches_svd_whitened_dense_reference():
    """Независимый SVD reference проверяет pivoted QR и back-transform."""
    rng = np.random.default_rng(7191)
    X = rng.normal(size=(500, 8)) @ rng.normal(size=(8, 8))
    y = X[:, 0] ** 2 + np.sin(X[:, 1])
    centered = X - X.mean(axis=0)
    left, singular, right = np.linalg.svd(centered, full_matrices=False)
    M = _dense_moments(np.sqrt(len(X)) * left, y)
    _, vectors = np.linalg.eigh(M)
    raw = right.T @ (vectors[:, -2:] / singular[:, None])
    expected, _ = np.linalg.qr(raw)
    actual, diagnostics = initialize_basis_inverse_moments(X, y, 2)
    np.testing.assert_allclose(actual @ actual.T, expected @ expected.T, atol=2e-10)
    np.testing.assert_allclose(actual.T @ actual, np.eye(2), atol=2e-14)
    assert diagnostics["relative_eigen_residual"] < 1e-8


def test_inverse_initializer_affine_equivariance_and_determinism():
    rng = np.random.default_rng(7192)
    X = rng.normal(size=(600, 8))
    y = X[:, 0] ** 2 + X[:, 1]
    basis, _ = initialize_basis_inverse_moments(X, y, 2, seed=41)
    repeated, _ = initialize_basis_inverse_moments(X, y, 2, seed=41)
    np.testing.assert_array_equal(basis, repeated)
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    transform = rotation @ np.diag(np.linspace(0.4, 3.0, 8))
    other, _ = initialize_basis_inverse_moments(X @ transform + 1000, y, 2, seed=52)
    mapped, _ = np.linalg.qr(transform @ other)
    np.testing.assert_allclose(basis @ basis.T, mapped @ mapped.T, atol=2e-10)
    shifted, _ = initialize_basis_inverse_moments(X + 1e8, y + 1e5, 2)
    np.testing.assert_allclose(basis @ basis.T, shifted @ shifted.T, atol=1e-7)


def test_inverse_initializer_rejects_degenerate_inputs():
    rng = np.random.default_rng(7193)
    X = rng.normal(size=(80, 5))
    y = X[:, 0] ** 2
    duplicate = X.copy()
    duplicate[:, 2] = duplicate[:, 0]
    with pytest.raises(RuntimeError, match="full column rank"):
        initialize_basis_inverse_moments(duplicate, y, 1)
    with pytest.raises(ValueError, match="constant response"):
        initialize_basis_inverse_moments(X, np.ones(80), 1)
    with pytest.raises(ValueError, match="n>d"):
        initialize_basis_inverse_moments(X, y, 4)
    with pytest.raises(ValueError, match="slices"):
        initialize_basis_inverse_moments(X, y, 1, slices=50)
    with pytest.raises(TypeError, match="real inputs"):
        initialize_basis_inverse_moments(X.astype(complex), y, 1)
    invalid = X.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite inputs"):
        initialize_basis_inverse_moments(invalid, y, 1)


def test_recovery_gate_retains_failures_and_checks_each_pair():
    from copy import deepcopy

    from experiments.index_recovery import make_protocol, summarize

    protocol = make_protocol()
    rows = []
    for seed in protocol["selection_seeds"]:
        for variant, quality in (("local", 0.8), ("sir-save", 0.85)):
            rows.append(
                {
                    "case": "test",
                    "variant": variant,
                    "seed": seed,
                    "quality": quality,
                    "initial_quality": quality,
                    "convergence_pass": variant == "local",
                    "recovered": False,
                    "spent_seconds": 1.0,
                    "peak_process_rss_mib": 100.0,
                    "data_sha256": str(seed),
                }
            )
    result = summarize(protocol, rows, "selection")
    assert result["groups"]["test/sir-save"]["passed"]
    assert not result["groups"]["test/sir-save"]["convergence_nonregression_debug"]
    regression = deepcopy(rows)
    regression[1]["quality"] = 0.799
    group = summarize(protocol, regression, "selection")["groups"]["test/sir-save"]
    assert not group["passed"] and group["quality_regressions"] == 1
    failed = deepcopy(rows)
    failed[1]["quality"] = None
    group = summarize(protocol, failed, "selection")["groups"]["test/sir-save"]
    assert not group["passed"] and group["fits"] == 4 and group["failures"] == 1
    unmatched = deepcopy(rows)
    unmatched[1]["data_sha256"] = "different"
    with pytest.raises(ValueError, match="unpaired data"):
        summarize(protocol, unmatched, "selection")
