from __future__ import annotations

import math
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from experiments.r_sdr import design, protocol


def test_frozen_python_worker_imports_only_snapshot(tmp_path):
    protocol.snapshot(tmp_path)
    frozen = tmp_path / "source"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import experiments.r_sdr.worker as w; print(w.__file__)",
        ],
        cwd=frozen,
        env={**os.environ, "PYTHONPATH": str(frozen)},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert str(frozen / "benchmarks/experiments/r_sdr/worker.py") in result.stdout


def test_seven_families_catalog_and_cli():
    cases = design.catalog("seven-families")
    assert len(cases) == len({case.name for case in cases}) == 7
    assert [case.link for case in cases] == [
        "additive",
        "quadratic3",
        "product3",
        "oscillatory3",
        "saturating3",
        "localized3",
        "max3",
    ]
    assert all(
        (c.n, c.d, c.m, c.noise, c.rho, c.distribution)
        == (2000, 20, 3, 0.3, 0.0, "gaussian")
        for c in cases
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.r_sdr",
            "--profile",
            "seven-families",
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


@pytest.mark.parametrize(
    ("link", "reference"),
    [
        ("additive", lambda a, b, c: math.sin(a) + 0.5 * b * b + 0.35 * math.tanh(c)),
        (
            "quadratic3",
            lambda a, b, c: (
                a * a + 0.8 * b * b + 0.6 * c * c + 0.5 * a * b + 0.4 * b * c
            ),
        ),
        ("product3", lambda a, b, c: a * b * c),
        (
            "oscillatory3",
            lambda a, b, c: math.sin(8 * a) * math.cos(10 * b) + math.sin(12 * c),
        ),
        (
            "saturating3",
            lambda a, b, c: math.tanh(2 * a) + math.tanh(2 * b) * math.tanh(2 * c),
        ),
        ("localized3", lambda a, b, c: math.exp(-(a * a + 2 * b * b + 3 * c * c) / 2)),
        ("max3", lambda a, b, c: max(a, b, c)),
    ],
)
def test_link_formulas_and_full_gradient_span(link, reference):
    z = np.random.default_rng(51).normal(size=(1000, 3))
    np.testing.assert_allclose(
        design.link_signal(link, z),
        [reference(*row) for row in z],
        atol=1e-14,
    )
    step = 1e-6
    gradients = np.column_stack(
        [
            (
                design.link_signal(link, z + step * axis)
                - design.link_signal(link, z - step * axis)
            )
            / (2 * step)
            for axis in np.eye(3)
        ]
    )
    # Full derivative span is necessary for the declared three-dimensional target.
    assert np.linalg.eigvalsh(gradients.T @ gradients / len(z))[0] > 1e-3
    if link != "additive":
        with pytest.raises(ValueError, match="three"):
            design.link_signal(link, z[:, :2])


def test_seven_families_normalization_and_reproducibility():
    for case in design.catalog("seven-families"):
        clean_case = replace(case, n=100, noise=0.0)
        x, y, truth = design.generate(clean_case, 0)
        for actual, expected in zip(
            (x, y, truth),
            design.generate(clean_case, 0),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
        assert np.isfinite(y).all()
        assert y.mean() == pytest.approx(0.0, abs=1e-14)
        assert y.std() == pytest.approx(1.0, abs=1e-14)
        np.testing.assert_allclose(truth.T @ truth, np.eye(3), atol=1e-14)
        assert not np.array_equal(x, design.generate(clean_case, 1)[0])
