"""Independent Gaussian integrals, pair sums and geometric recovery checks."""

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss


def test_stein_moments_match_analytic_linear_quadratic_and_sine_integrals():
    # Wrong score/phase signs break the quadratic imaginary component.
    from experiments.gaussian_fourier import moments

    nodes, weights = hermgauss(30)
    grid = np.array(np.meshgrid(nodes, nodes, indexing="ij")).reshape(2, -1).T
    x = np.sqrt(2) * grid
    quadrature = np.outer(weights, weights).ravel() / np.pi
    frequencies = np.array([[0.0, 0.0], [0.4, -0.7], [-1.1, 0.3]])
    damping = np.exp(-np.sum(frequencies**2, axis=1) / 2)
    beta = np.array([0.8, -0.3])
    hessian = np.array([[0.4, 0.2], [0.2, -0.6]])
    direction = np.array([0.6, 0.8])
    responses = [
        x @ beta,
        0.5 * np.sum((x @ hessian) * x, axis=1),
        np.sin(x @ direction),
    ]
    expected = [
        beta[:, None] * damping,
        1j * (hessian @ frequencies.T) * damping,
        direction[:, None]
        * (
            np.exp(-np.sum((frequencies + direction) ** 2, axis=1) / 2)
            + np.exp(-np.sum((frequencies - direction) ** 2, axis=1) / 2)
        )
        / 2,
    ]
    for y, reference in zip(responses, expected, strict=True):
        # Encode integration weights in y: moments is linear in its response.
        result = moments(x, len(x) * quadrature * y, frequencies, batch_size=2)
        np.testing.assert_allclose(result, reference, atol=2e-12, rtol=2e-12)


def test_matrix_equals_direct_distinct_observation_pair_sum():
    # A plug-in Gram matrix or a missing frequency term fails this reference.
    from experiments.gaussian_fourier import candidate_matrix

    x = np.array([[1.0, 0.0], [-0.5, 2.0], [0.2, -0.7]])
    y = np.array([0.4, -1.2, 0.8])
    frequencies = np.array([[0.0, 0.0], [0.2, -0.8]])
    reference = np.zeros((2, 2))
    for t in frequencies:
        for i in range(len(x)):
            for j in range(len(x)):
                if i != j:
                    left = y[i] * (x[i] - 1j * t) * np.exp(1j * (x[i] @ t))
                    right = y[j] * (x[j] - 1j * t) * np.exp(1j * (x[j] @ t))
                    reference += np.real(np.outer(left, right.conj()))
    reference /= len(frequencies) * len(x) * (len(x) - 1)
    actual = candidate_matrix(x, y, frequencies, batch_size=1)
    np.testing.assert_allclose(actual, reference, atol=2e-15)


def test_population_spectral_objective_is_edr_squared_even_for_unequal_signal():
    # Forgetting to flatten eigenvalues weights angular errors unequally.
    from experiments.gaussian_fourier import objective, solve

    matrix = np.diag([9.0, 0.01, 0.0, 0.0])
    result = solve(matrix, 2)
    angles = np.array([0.3, 0.7])
    basis = np.vstack((np.diag(np.cos(angles)), np.diag(np.sin(angles))))
    value, gradient = objective(basis, result.basis)
    assert value == pytest.approx(np.mean(np.sin(angles) ** 2), abs=1e-14)
    assert objective(result.basis, result.basis)[0] == pytest.approx(0, abs=1e-14)
    direction = np.array([[0.1, -0.2], [0.3, 0.1], [-0.4, 0.5], [0.2, -0.3]])
    step = 1e-6
    plus = np.linalg.qr(basis + step * direction)[0]
    minus = np.linalg.qr(basis - step * direction)[0]
    difference = (
        objective(plus, result.basis)[0] - objective(minus, result.basis)[0]
    ) / (2 * step)
    assert difference == pytest.approx(np.sum(gradient * direction), abs=2e-9)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    assert objective(basis @ rotation, result.basis @ rotation)[0] == pytest.approx(
        value
    )
    assert result.diagnostics["sample_recovery_certified"] is False


def test_spectral_perturbation_bound_covers_known_rotated_target():
    from experiments.gaussian_fourier import perturbation_bound, solve

    population = np.diag([2.0, 1.0, 0.0])
    perturbation = np.array([[0.0, 0.0, 0.03], [0.0, 0.0, 0.08], [0.03, 0.08, 0.0]])
    result = solve(population + perturbation, 2)
    delta = np.linalg.norm(perturbation, ord=2)
    bound = perturbation_bound(result.eigenvalues[1], delta)
    assert np.linalg.norm(result.basis[2:]) / np.sqrt(2) <= bound < 0.1
    assert perturbation_bound(0.1, 0.2) == 1
    with pytest.raises(ValueError):
        perturbation_bound(0, 0.1)
    with pytest.raises(ValueError):
        perturbation_bound(1, -0.1)


@pytest.mark.parametrize(
    "matrix,m", [(np.zeros((3, 3)), 1), (np.eye(3), 1), (np.diag([2.0, -1.0, -2.0]), 2)]
)
def test_spectral_solve_rejects_unidentified_selected_subspace(matrix, m):
    from experiments.gaussian_fourier import solve

    with pytest.raises(ValueError, match="identified"):
        solve(matrix, m)


def test_fit_is_reproducible_equivariant_and_has_no_truth_input():
    from experiments.gaussian_fourier import candidate_matrix, fit, moments

    rng = np.random.default_rng(8)
    x = rng.normal(size=(400, 3))
    y = x[:, 0] ** 2 - 1 + 0.1 * rng.normal(size=len(x))
    result = fit(x, y, 1, n_frequencies=32, seed=19)
    repeated = fit(x, y, 1, n_frequencies=32, seed=19)
    np.testing.assert_array_equal(result.basis, repeated.basis)
    assert result.basis[0, 0] ** 2 > 0.9
    np.testing.assert_allclose(result.basis.T @ result.basis, np.eye(1), atol=1e-14)
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    frequencies = rng.normal(size=(7, 3))
    original = candidate_matrix(x, y, frequencies)
    rotated = candidate_matrix(x @ rotation, y, frequencies @ rotation)
    np.testing.assert_allclose(rotated, rotation.T @ original @ rotation, atol=1e-13)
    np.testing.assert_allclose(
        moments(x, y, frequencies, batch_size=1),
        moments(x, y, frequencies, batch_size=7),
        atol=1e-14,
    )
    with pytest.raises(TypeError):
        fit(x, y, 1, truth=np.eye(3)[:, :1])


def test_invalid_data_and_subspace_inputs_are_rejected():
    from experiments.gaussian_fourier import fit, moments, objective, solve

    x = np.ones((4, 3))
    for y in (np.ones(3), np.array([1, 1, np.nan, 1]), np.ones(4, dtype=complex)):
        with pytest.raises((TypeError, ValueError)):
            fit(x, y, 1)
    for kwargs in (
        {"index_dim": 0},
        {"index_dim": True},
        {"index_dim": 3},
        {"index_dim": 1, "n_frequencies": 0},
        {"index_dim": 1, "frequency_scale": np.inf},
    ):
        with pytest.raises((TypeError, ValueError)):
            fit(x, np.arange(4.0), **kwargs)
    with pytest.raises(ValueError):
        moments(x, np.arange(4.0), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        moments(x, np.arange(4.0), np.zeros((2, 3)), batch_size=0)
    with pytest.raises(ValueError):
        solve(np.array([[1.0, 2.0], [0.0, 1.0]]), 1)
    with pytest.raises(ValueError):
        objective(np.ones((3, 1)), np.eye(3)[:, :1])


def test_forced_zero_frequency_has_only_one_real_population_direction():
    from experiments.gaussian_fourier import fit

    rng = np.random.default_rng(3)
    x = rng.normal(size=(20, 4))
    with pytest.raises(ValueError, match="population rank"):
        fit(x, rng.normal(size=20), 2, n_frequencies=1)


def test_npz_cli_saves_a_usable_basis_and_explicit_uncertified_diagnostics(tmp_path):
    import json
    import subprocess
    import sys

    rng = np.random.default_rng(17)
    x = rng.normal(size=(500, 3))
    source, target = tmp_path / "data.npz", tmp_path / "fit.npz"
    np.savez(source, X=x, y=x[:, 0] ** 2 - 1)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.gaussian_fourier",
            str(source),
            "--index-dim",
            "1",
            "--output",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["sample_recovery_certified"] is False
    with np.load(target, allow_pickle=False) as saved:
        assert saved["basis"][0, 0] ** 2 > 0.9
        assert json.loads(str(saved["diagnostics"]))["predictor_law"] == "known N(0,I)"


def test_study_gate_rejects_slow_or_worse_pairs_and_missing_attempts():
    from experiments.gaussian_fourier_study import evaluate_gate

    rows = []
    for seed in (11, 12):
        for m in (2, 3, 5):
            for arm, edr in (("base", 0.2), ("fourier-2", 0.08)):
                rows.append(
                    {
                        "seed": seed,
                        "m": m,
                        "function": "original",
                        "arm": arm,
                        "status": "ok",
                        "edr": edr,
                        "fit_seconds": 1.0,
                        "process_seconds": 1.0,
                    }
                )
    result = evaluate_gate(rows, 12)
    assert result["pass"] is True
    assert result["primary_edr_ratio"] == pytest.approx(0.4)
    rows[-1]["fit_seconds"] = 3.01
    assert evaluate_gate(rows, 12)["checks"]["fit_time_le_3x"] is False
    rows[-1]["fit_seconds"] = 1
    rows[-1]["edr"] = 0.9
    assert evaluate_gate(rows, 12)["pass"] is False
    assert evaluate_gate(rows[:-1], 12)["pass"] is False


@pytest.mark.parametrize("m", [3, 5])
def test_exact_population_moments_recover_all_original_generator_directions(m):
    # This checks finite-feature rank on the hard generator, without sample noise.
    from experiments.gaussian_fourier import solve

    rng = np.random.default_rng(42)
    truth = np.linalg.qr(rng.normal(size=(8, m)))[0]
    frequencies = rng.normal(scale=2 / np.sqrt(8), size=(128, 8))
    u = frequencies @ truth
    damping = np.exp(-np.sum(frequencies**2, axis=1) / 2)
    components = np.zeros((m, len(frequencies)), dtype=complex)
    components[0] = damping * np.exp(-0.5) * np.cosh(u[:, 0])
    components[1] = 1j * u[:, 1] * damping
    nodes, weights = hermgauss(60)
    z = np.sqrt(2) * nodes
    for j in range(2, m):
        average = (np.cos(u[:, j, None] * z) / np.cosh(z) ** 2) @ (
            weights / np.sqrt(np.pi)
        )
        components[j] = (
            0.35
            * average
            * np.exp(-0.5 * (np.sum(frequencies**2, axis=1) - u[:, j] ** 2))
        )
    vectors = truth @ components
    population = (vectors @ vectors.conj().T).real / len(frequencies)
    result = solve(population, m)
    np.testing.assert_allclose(
        result.basis @ result.basis.T, truth @ truth.T, atol=2e-11
    )
