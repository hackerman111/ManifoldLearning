from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from ADP.solver._grassman_schur import metric_direction
from ADP.solver.grassman_research import solve


def _metric_oracle(G, coefficients, mass, rank, floor):
    R = coefficients.T @ (mass[:, None] * coefficients)
    values, vectors = np.linalg.eigh(R)
    values = np.maximum(values, floor * values[-1])
    whitening = (vectors / np.sqrt(values)) @ vectors.T
    left, singular, right = np.linalg.svd(G @ whitening, full_matrices=False)
    scale = np.linalg.norm(singular[:rank])
    return -((left[:, :rank] * singular[:rank]) @ right[:rank]) @ whitening / scale


def _small_case(seed=9601):
    rng = np.random.default_rng(seed)
    d, m, J = 9, 3, 13
    G = rng.normal(size=(d, m))
    coefficients = rng.normal(size=(J, m)) @ np.diag([4.0, 1.0, 0.25])
    mass = np.geomspace(0.3, 2.1, J)
    return rng, G, coefficients, mass


def _profile(U, I, mass, index):
    Y = index.T
    value = 0.0
    for j in range(len(I)):
        design = U[j] @ Y
        coefficient = np.linalg.lstsq(design, I[j], rcond=None)[0]
        residual = I[j] - design @ coefficient
        value += mass[j] * float(residual @ residual)
    return value


def _full_gradient(U, I, mass, index):
    Y = index.T
    ambient = np.zeros_like(Y)
    for j in range(len(I)):
        design = U[j] @ Y
        coefficient = np.linalg.lstsq(design, I[j], rcond=None)[0]
        residual = I[j] - design @ coefficient
        ambient -= 2 * mass[j] * U[j].T @ np.outer(residual, coefficient)
    return ambient - Y @ (Y.T @ ambient)


def _polar_basis(Y, V, K):
    left, _, right = np.linalg.svd(Y + V @ K, full_matrices=False)
    return left @ right


def test_metric_direction_matches_independent_floored_r_svd_and_descent():
    _, G, coefficients, mass = _small_case()
    floor, rank = 0.07, 2
    actual = metric_direction(G, coefficients, mass, rank, floor)
    expected = _metric_oracle(G, coefficients, mass, rank, floor)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    R = coefficients.T @ (mass[:, None] * coefficients)
    values, vectors = np.linalg.eigh(R)
    R_floor = (vectors * np.maximum(values, floor * values[-1])) @ vectors.T
    metric_norm = np.sqrt(np.trace(actual @ R_floor @ actual.T))
    assert metric_norm == pytest.approx(1.0, rel=2e-12)
    assert np.linalg.matrix_rank(actual, tol=1e-11) <= rank
    assert np.sum(G * actual) < 0


def test_floor_one_uses_frobenius_normalized_rank_two_truncation_without_cap():
    _, G, coefficients, mass = _small_case(seed=9602)
    G = np.diag([5.0, 2.0, 0.3])
    coefficients *= 500.0
    rank = 2
    left, singular, right = np.linalg.svd(G, full_matrices=False)
    truncated = (left[:, :rank] * singular[:rank]) @ right[:rank]
    lambda_max = np.linalg.eigvalsh(coefficients.T @ (mass[:, None] * coefficients))[-1]
    expected = -truncated / (np.sqrt(lambda_max) * np.linalg.norm(truncated))
    actual = metric_direction(G, coefficients, mass, rank, metric_floor=1.0)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    assert np.linalg.norm(actual, ord=2) < np.tan(0.5)


def test_metric_direction_is_right_gauge_covariant():
    rng, G, coefficients, mass = _small_case(seed=9603)
    Q, _ = np.linalg.qr(rng.normal(size=(G.shape[1], G.shape[1])))
    original = metric_direction(G, coefficients, mass, 2, 1e-3)
    rotated = metric_direction(G @ Q, coefficients @ Q, mass, 2, 1e-3)
    np.testing.assert_allclose(rotated, original @ Q, rtol=2e-11, atol=2e-11)


def test_metric_profile_gradient_matches_polar_finite_difference():
    rng = np.random.default_rng(9604)
    J, p, d, m, q = 12, 7, 9, 2, 2
    U = rng.normal(size=(J, p, d))
    I = rng.normal(size=(J, p))
    mass = np.geomspace(0.4, 1.7, J)
    Y = np.linalg.qr(rng.normal(size=(d, m)))[0]
    frame = np.linalg.qr(np.column_stack((Y, rng.normal(size=(d, q)))))[0]
    V = frame[:, m : m + q]
    direction = rng.normal(size=(q, m))
    gradient = _full_gradient(U, I, mass, Y.T)
    analytic = float(np.sum((V.T @ gradient) * direction))
    step = 1e-6
    plus = _polar_basis(Y, V, step * direction)
    minus = _polar_basis(Y, V, -step * direction)
    numeric = (_profile(U, I, mass, plus.T) - _profile(U, I, mass, minus.T)) / (
        2 * step
    )
    assert numeric == pytest.approx(analytic, rel=2e-6, abs=2e-7)


def test_spokoiny_population_opg_isotropic_by_gauss_legendre_quadrature():
    nodes, weights = np.polynomial.legendre.leggauss(8)
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    u, v = (x + y) / np.sqrt(2), (x - y) / np.sqrt(2)
    f_u = 4 * u**3 - 3 * u**2 * v**3 + v
    f_v = u - 3 * u**3 * v**2 - 4 * v**3
    w = np.outer(weights, weights) / 4
    opg = np.array(
        [
            [np.sum(w * f_u**2), np.sum(w * f_u * f_v)],
            [np.sum(w * f_u * f_v), np.sum(w * f_v**2)],
        ]
    )
    np.testing.assert_allclose(opg, np.eye(2) * (5557 / 1155), rtol=2e-14, atol=2e-14)


def test_matched_isotropic_worker_uses_spectral_norm_ratio_for_rank_two():
    from experiments.metric_spokoiny_2026_10_15.diagnostics import (
        metric_direction_matched,
    )

    _, G, coefficients, mass = _small_case(seed=9606)
    actual = metric_direction_matched(G, coefficients, mass, 2)
    left, singular, right = np.linalg.svd(G, full_matrices=False)
    truncated = -(left[:, :2] * singular[:2]) @ right[:2]
    original = metric_direction(G, coefficients, mass, 2)
    expected = truncated * (
        np.linalg.norm(original, ord=2) / np.linalg.norm(truncated, ord=2)
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    assert np.linalg.norm(actual, ord=2) == pytest.approx(
        np.linalg.norm(original, ord=2)
    )


def test_isotropic_worker_arms_smoke_and_restore_patch():
    from ADP.solver import grassman_research
    from experiments.metric_spokoiny_2026_10_15 import diagnostics, workers

    rng = np.random.default_rng(9607)
    J, p, d, m = 8, 6, 9, 2
    U = rng.normal(size=(J, p, d))
    I = rng.normal(size=(J, p))
    mass = np.geomspace(0.6, 1.8, J)
    B = np.linalg.qr(rng.normal(size=(d, m)))[0].T
    original = grassman_research.metric_direction
    outputs = {}
    for arm in ("isotropic2", "isotropic2_matched"):
        patch = diagnostics.metric_patch(arm)
        try:
            if arm == "isotropic2_matched":
                assert (
                    grassman_research.metric_direction
                    is diagnostics.metric_direction_matched
                )
                assert diagnostics._MATCH_REFERENCE is original
            else:
                assert grassman_research.metric_direction is original
            result = workers._solver(arm, 2).method(B, U, I, mass=mass)
            outputs[arm] = result
        finally:
            diagnostics.restore_metric(patch)
        assert grassman_research.metric_direction is original
    for result in outputs.values():
        assert np.all(np.isfinite(result.index))
        assert np.all(np.diff(result.diagnostics["profile_history"]) <= 2e-12)
        np.testing.assert_allclose(result.index @ result.index.T, np.eye(m), atol=2e-12)


def test_frozen_launcher_forwards_each_snapshot_path(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from experiments.metric_spokoiny_2026_10_15 import run as launcher

    captures = tmp_path / "captures"
    captures.mkdir()
    rows = []
    for seed in range(3):
        curves = [
            {"capture_kind": "independent_snapshot_input", "call_index": index}
            for index in (0, 80, 160)
        ]
        for index in (0, 80, 160):
            (captures / f"core_seed{seed}_call{index}.npz").touch()
        rows.append(
            {"status": "ok", "arm": "core", "seed": seed, "capture_curves": curves}
        )
    (tmp_path / "capture.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    jobs, _ = launcher._jobs("frozen", tmp_path)
    keys = {(job["seed"], job["phase"], job["budget"], job["arm"]) for job in jobs}
    assert len(jobs) == len(keys) == 90
    job = jobs[0]
    captured_commands = []

    def fake_run(command, **kwargs):
        captured_commands.append(command)
        return SimpleNamespace(returncode=0, stdout='{"status":"ok"}\n', stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    record = launcher._run_job(job, tmp_path, 60)
    command = captured_commands[0]
    snapshot_arg = command.index("--snapshot")
    assert command[snapshot_arg + 1] == str(job["snapshot"])
    assert record["status"] == "ok"


@pytest.mark.parametrize("rank", [1, 2])
def test_metric_solver_is_monotone_orthogonal_and_live_gradient_certified(rank):
    rng = np.random.default_rng(9605)
    J, p, d, m = 14, 8, 10, 2
    U = rng.normal(size=(J, p, d)) / np.sqrt(p)
    I = rng.normal(size=(J, p))
    mass = np.geomspace(0.5, 2.0, J)
    Y = np.linalg.qr(rng.normal(size=(d, m)))[0].T
    result = solve(Y, U, I, mass=mass, method="metric", rank=rank, max_steps=4)
    diagnostics = result.diagnostics
    history = cast(tuple[float, ...], diagnostics["profile_history"])
    assert np.all(np.diff(history) <= 2e-12)
    np.testing.assert_allclose(result.index @ result.index.T, np.eye(m), atol=2e-12)
    assert cast(float, diagnostics["orthogonality_error"]) < 2e-12
    expected_value = _profile(U, I, mass, result.index)
    assert cast(float, diagnostics["profile_objective"]) == pytest.approx(
        expected_value, rel=3e-11
    )
    gradient = _full_gradient(U, I, mass, result.index)
    norm = np.linalg.norm(gradient)
    scale = max(1.0, float(np.sum(mass[:, None] * I**2)))
    raw = cast(float, diagnostics["gradient_norm"])
    normalized = cast(float, diagnostics["riemannian_gradient"])
    assert raw == pytest.approx(norm, rel=3e-10, abs=3e-10)
    assert normalized == pytest.approx(norm / scale, rel=3e-10)
    assert cast(int, diagnostics["full_gradient_evaluations"]) >= 1
