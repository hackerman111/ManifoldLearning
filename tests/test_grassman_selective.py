"""Independent checks for localized QR/SVD profile dispatch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ADP.solver import grassman as baseline
from experiments.grassman_accuracy_2026_10_18 import selective


def direct(M: np.ndarray, I: np.ndarray, mass: np.ndarray, tau: float):
    J, _, m = M.shape
    coefficients, ranks, value = np.empty((J, m)), [], 0.0
    for j, matrix in enumerate(M):
        design = np.vstack((matrix, np.sqrt(tau) * np.eye(m)))
        target = np.r_[I[j], np.zeros(m)]
        coefficients[j], _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        ranks.append(rank)
        residual = I[j] - matrix @ coefficients[j]
        value += mass[j] * (
            residual @ residual + tau * (coefficients[j] @ coefficients[j])
        )
    return float(value), coefficients, np.asarray(ranks)


def assert_profiles(M: np.ndarray, I: np.ndarray, mass: np.ndarray, tau: float):
    got = selective.profile(M, I, mass, tau)
    reference = baseline._profile(M, I, mass, tau)
    value, coefficients, ranks = direct(M, I, mass, tau)
    assert got.value == pytest.approx(value, rel=2e-12, abs=2e-11)
    assert got.value == pytest.approx(reference.value, rel=2e-12, abs=2e-11)
    assert got.coefficients == pytest.approx(coefficients, rel=2e-12, abs=2e-11)
    assert got.coefficients == pytest.approx(
        reference.coefficients, rel=2e-12, abs=2e-11
    )
    assert got.ranks.tolist() == ranks.tolist() == reference.ranks.tolist()
    rhs = np.random.default_rng(77).normal(size=(len(M), M.shape[2], 3))
    assert got.normal_solve(rhs) == pytest.approx(
        reference.normal_solve(rhs), abs=2e-10
    )
    d, m = M.shape[2] + 2, M.shape[2]
    Y = np.eye(d, m)
    U = np.random.default_rng(89).normal(size=(len(M), M.shape[1], d))
    U[:, :, :m] = M
    state_g = baseline._gradient(Y, U, got, mass)
    state_ref = baseline._gradient(Y, U, reference, mass)
    assert state_g == pytest.approx(state_ref, abs=2e-10)
    assert got.smooth == reference.smooth


@pytest.mark.parametrize("tau", [0.0, 0.2])
def test_profile_gradient_matches_independent_tangent_difference(tau: float) -> None:
    rng = np.random.default_rng(93901)
    U, I = rng.normal(size=(6, 7, 5)), rng.normal(size=(6, 7))
    U[0] = 0.0
    mass, Y = np.geomspace(0.2, 2.0, len(I)), np.eye(5, 2)
    H = rng.normal(size=Y.shape)
    H -= Y @ (Y.T @ H)
    state = selective.profile(U @ Y, I, mass, tau)
    gradient = baseline._gradient(Y, U, state, mass)
    assert np.linalg.norm(gradient) > 1e-8
    step = 1e-6
    plus = np.linalg.qr(Y + step * H, mode="reduced")[0]
    minus = np.linalg.qr(Y - step * H, mode="reduced")[0]
    difference = (
        direct(U @ plus, I, mass, tau)[0] - direct(U @ minus, I, mass, tau)[0]
    ) / (2 * step)
    assert np.sum(gradient * H) == pytest.approx(difference, rel=2e-6, abs=2e-7)


@pytest.mark.parametrize(
    ("m", "p", "tau"),
    [(1, 4, 0.0), (1, 2, 0.2), (2, 5, 0.0), (2, 3, 0.1), (3, 2, 0.0), (3, 5, 0.3)],
)
def test_profile_matches_lstsq_and_production(m: int, p: int, tau: float) -> None:
    rng = np.random.default_rng(610 + m + p)
    M = rng.normal(size=(5, p, m))
    I = rng.normal(size=(5, p))
    mass = np.array([0.0, 0.2, 1.0, 2.0, 0.5])  # zero mass, nonzero target
    assert_profiles(M, I, mass, tau)


def test_mixed_exactzero_rankone_and_tiny_centers_refit_only_unsafe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(93800)
    M, I = rng.normal(size=(6, 7, 2)), rng.normal(size=(6, 7))
    M[1] = 0
    M[2, :, 1] = 0
    M[3, :, 1] *= 1e-15
    mass = np.array([1.0, 0.7, 2.0, 0.3, 1.4, 0.8])
    calls = []
    profile_calls = []
    original = baseline._profile_svd
    original_profile = baseline._profile

    def recording(Ms, Is, masses, tau):
        calls.append(len(Ms))
        return original(Ms, Is, masses, tau)

    def recording_profile(Ms, Is, masses, tau):
        profile_calls.append(len(Ms))
        return original_profile(Ms, Is, masses, tau)

    monkeypatch.setattr(baseline, "_profile_svd", recording)
    monkeypatch.setattr(baseline, "_profile", recording_profile)
    state = selective.profile(M, I, mass, 0.0)
    assert calls == [3]
    assert not state.smooth
    assert profile_calls == []
    monkeypatch.setattr(baseline, "_profile_svd", original)
    monkeypatch.setattr(baseline, "_profile", original_profile)
    assert_profiles(M, I, mass, 0.0)
    assert state.value == pytest.approx(baseline._profile(M, I, mass, 0.0).value)


def test_all_safe_profile_reuses_guard_qr_factors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(93801)
    M, I = rng.normal(size=(5, 7, 2)), rng.normal(size=(5, 7))
    mass = np.geomspace(0.2, 2.0, len(I))
    calls = []
    original = baseline._profile

    def recording(Ms, Is, masses, tau):
        calls.append(len(Ms))
        return original(Ms, Is, masses, tau)

    monkeypatch.setattr(baseline, "_profile", recording)
    selective.profile(M, I, mass, 0.1)
    assert calls == []
    monkeypatch.setattr(baseline, "_profile", original)
    assert_profiles(M, I, mass, 0.1)


def test_smooth_but_qrunsafe_batch_falls_back_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    M = np.zeros((2, 6, 2))
    M[0, 0, 0], M[0, 1, 1] = 1.0, 1.0
    M[1, 0, 0], M[1, 0, 1], M[1, 1, 1] = 1.0, 1.0, 1e-5
    I, mass = np.ones((2, 6)), np.ones(2)
    calls, original = [], baseline._profile_svd

    def recording(Ms, Is, masses, tau):
        calls.append(len(Ms))
        return original(Ms, Is, masses, tau)

    monkeypatch.setattr(baseline, "_profile_svd", recording)
    state = selective.profile(M, I, mass, 0.0)
    assert state.smooth
    assert calls == [1, 2]
    assert state.value == pytest.approx(baseline._profile(M, I, mass, 0.0).value)


def test_sample_frozen_inputs_match_direct_profiles() -> None:
    root = Path(__file__).parents[1] / "benchmarks/experiments/grassman_accuracy_2026_10_18/frozen"
    names = ("001.npz", "008.npz", "014.npz")
    if not all((root / name).is_file() for name in names):
        pytest.skip("optional frozen diagnostic inputs are unavailable")
    for name in names:
        with np.load(root / name) as saved:
            M = saved["U"] @ saved["index_init"].T
            assert_profiles(M, saved["I"], saved["mass"], 0.0)


def test_wrapper_step_budgets_and_input_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(615)
    U = rng.normal(size=(8, 6, 5))
    I, index = rng.normal(size=(8, 6)), np.eye(2, 5)
    seen = []
    original = selective._core.solve

    def capture(*args, **kwargs):
        seen.append(kwargs["max_steps"])
        return original(*args, **kwargs)

    monkeypatch.setattr(selective._core, "solve", capture)
    selective.solve(index, U, I, max_steps=10)
    assert seen[-1] == 5
    U[0] = 0
    selective.solve(index, U, I, max_steps=10)
    assert seen[-1] == 10
    with pytest.raises(TypeError, match="max_steps"):
        selective.solve(index, U, I, max_steps=True)
    with pytest.raises(ValueError, match="mass"):
        selective.solve(index, U, I, mass=np.zeros(len(I)))
    with pytest.raises(ValueError):
        selective.solve(np.ones((2, 6)), U, I, max_steps=1)
