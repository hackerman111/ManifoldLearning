from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from experiments.r_sdr import design
from scipy.special import roots_hermitenorm


def test_seven_oscillation_frequencies_and_cli():
    cases = design.catalog("oscillation-seven")
    assert [c.frequency for c in cases] == [0.5, 1, 2, 3, 4, 6, 8]
    assert len({c.name for c in cases}) == 7
    assert all((c.n, c.d, c.m, c.noise) == (2000, 20, 3, 0.3) for c in cases)
    assert {c.data_key for c in cases} == {"family-oscillatory"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.r_sdr",
            "--profile",
            "oscillation-seven",
            "--runs",
            "15",
            "--seed",
            "0",
            "--methods",
            "ADP-Grassmann-combined,MAVE-package",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "7 cases x 15 repetitions x 2 methods = 210 fits" in result.stdout


def test_frequency_sweep_shares_x_basis_and_noise_and_reproduces_legacy_high():
    cases = design.catalog("oscillation-seven")
    for seed in (0, 14):
        x0, _, b0 = design.generate(cases[0], seed)
        residuals = []
        for case in cases:
            x, y, basis = design.generate(case, seed)
            _, signal, _ = design.generate(replace(case, noise=0), seed)
            np.testing.assert_array_equal(x, x0)
            np.testing.assert_array_equal(basis, b0)
            residuals.append(y - signal)
            assert signal.std() == pytest.approx(1)
        for residual in residuals[1:]:
            np.testing.assert_allclose(residual, residuals[0], atol=8e-16, rtol=0)
        old = design.Case("family-oscillatory", 2000, 20, 3, "oscillatory3")
        for actual, expected in zip(
            design.generate(cases[-1], seed), design.generate(old, seed), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("frequency", [0.5, 1, 2, 4, 8])
def test_population_gradient_moments_against_gaussian_quadrature(frequency):
    from experiments.r_sdr.oscillation import population_moments

    z, weights = roots_hermitenorm(512)
    weights /= np.sqrt(2 * np.pi)
    a, b, c = frequency * np.array([1, 1.25, 1.5])
    ca, cb, cc = (np.cos(t * z) for t in (a, b, c))
    sa, sb, sc = (np.sin(t * z) for t in (a, b, c))
    mean = np.array([a * (weights @ ca) * (weights @ cb), 0, c * (weights @ cc)])
    gram = np.diag(
        [
            a * a * (weights @ ca**2) * (weights @ cb**2),
            b * b * (weights @ sa**2) * (weights @ sb**2),
            c * c * (weights @ cc**2),
        ]
    )
    gram[0, 2] = gram[2, 0] = mean[0] * mean[2]
    variance = (weights @ sa**2) * (weights @ cb**2) + weights @ sc**2
    moments = population_moments(frequency)
    assert moments["signal_variance"] == pytest.approx(variance, abs=2e-13)
    np.testing.assert_allclose(
        moments["gradient_mean"], mean / np.sqrt(variance), atol=3e-12
    )
    np.testing.assert_allclose(moments["gradient_gram"], gram / variance, atol=3e-11)
    assert np.linalg.eigvalsh(moments["gradient_gram"])[0] > 0


def test_frequency_link_and_analytic_gradient():
    from experiments.r_sdr.oscillation import signal_gradient

    z = np.random.default_rng(21).normal(size=(100, 3))
    for a in (0.5, 3, 8):
        signal = design.link_signal("oscillatory-frequency3", z, frequency=a)
        np.testing.assert_allclose(
            signal,
            np.sin(a * z[:, 0]) * np.cos(1.25 * a * z[:, 1])
            + np.sin(1.5 * a * z[:, 2]),
        )
        step = 1e-6
        gradient = np.column_stack(
            [
                (
                    design.link_signal(
                        "oscillatory-frequency3", z + step * e, frequency=a
                    )
                    - design.link_signal(
                        "oscillatory-frequency3", z - step * e, frequency=a
                    )
                )
                / (2 * step)
                for e in np.eye(3)
            ]
        )
        np.testing.assert_allclose(
            signal_gradient(z, a), gradient, rtol=2e-7, atol=1e-8
        )
    for bad in (0, -1, np.inf, np.nan):
        with pytest.raises(ValueError, match="frequency"):
            design.link_signal("oscillatory-frequency3", z, frequency=bad)
