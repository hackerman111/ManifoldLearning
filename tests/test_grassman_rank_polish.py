"""Independent chart profile derivative and safe reference improvement."""

import numpy as np

from experiments.grassman_rank_k.inner import make_frozen
from experiments.grassman_rank_k.reference import direct_profile
from experiments.grassman_rank_k.reference_polish import chart_profile, polish


def test_chart_envelope_gradient_and_gauge():
    snap = make_frozen(2, 7, 31, J=12)
    base, U, I, mass = (snap[k] for k in ("truth", "U", "I", "mass"))
    rng = np.random.default_rng(42)
    z = rng.normal(size=base.shape) * 0.1
    delta = rng.normal(size=base.shape)
    f, gradient = chart_profile(z.ravel(), base, U, I, mass)
    eps = 1e-6
    fp = chart_profile((z + eps * delta).ravel(), base, U, I, mass)[0]
    fm = chart_profile((z - eps * delta).ravel(), base, U, I, mass)[0]
    np.testing.assert_allclose(
        (fp - fm) / (2 * eps), gradient @ delta.ravel(), rtol=1e-6
    )
    B = base + z - base @ (base.T @ z)
    np.testing.assert_allclose(f, direct_profile(np.linalg.qr(B)[0], U, I, mass)[0])
    assert np.linalg.svd(B, compute_uv=False)[-1] >= 1 - 1e-12


def test_reference_polish_certifies_small_problem():
    result = polish(make_frozen(2, 7, 31, J=12), max_steps=300)
    assert result["converged"]
    assert result["normalized_gradient"] <= 1e-9
    assert result["orthogonality_error"] < 1e-12
    assert all(
        r["objective"] <= r["initial_objective"] + 1e-10 for r in result["starts"]
    )
