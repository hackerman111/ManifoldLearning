from __future__ import annotations

import numpy as np
import pytest
from experiments.e_means_adp.candidate import (
    EmaSubspace,
    MiniBatchSolver,
    cosine_step,
)

from ADP import ADP_Config, ADP_multi_index, ADP_solver
from ADP.solver import grassman as g
from ADP.solver.LSMR import HPAOResult


def basis(rng, d=7, m=2):
    return np.linalg.qr(rng.normal(size=(d, m)))[0].T


def reference(B, U, I, mass, tau=0.0):
    coefficients = []
    value = 0.0
    for u, y, weight in zip(U, I, mass, strict=True):
        design = u @ B.T
        if tau:
            c = np.linalg.lstsq(
                np.vstack((design, np.sqrt(tau) * np.eye(len(B)))),
                np.concatenate((y, np.zeros(len(B)))),
                rcond=None,
            )[0]
        else:
            c = np.linalg.lstsq(design, y, rcond=None)[0]
        value += weight * (np.sum((y - design @ c) ** 2) + tau * (c @ c))
        coefficients.append(c)
    return np.asarray(coefficients), value


def test_ema_matches_dense_recurrence_and_gauge():
    rng = np.random.default_rng(17)
    B = basis(rng)
    ema, rotated = EmaSubspace(B), EmaSubspace(-B)
    dense = 1e-10 * B.T @ B
    count = 1e-10
    for t in range(8):
        B = basis(rng)
        R = np.linalg.qr(rng.normal(size=(2, 2)))[0]
        alpha = cosine_step(t, 8)
        ema.update(B, alpha)
        rotated.update(R @ B, alpha)
        dense = (1 - alpha) * dense + alpha * B.T @ B
        count = (1 - alpha) * count + alpha
        np.testing.assert_allclose(ema.factor @ ema.factor.T, dense, atol=4e-16)
        assert ema.mass == pytest.approx(count, rel=2e-15)
    proposed, gap = ema.proposal()
    other, other_gap = rotated.proposal()
    assert proposed is not None and other is not None and gap > 0
    np.testing.assert_allclose(proposed.T @ proposed, other.T @ other, atol=2e-14)
    values, vectors = np.linalg.eigh(dense / count)
    np.testing.assert_allclose(
        proposed.T @ proposed, vectors[:, -2:] @ vectors[:, -2:].T, atol=2e-14
    )
    assert gap == pytest.approx(values[-2] - values[-3])
    assert other_gap == pytest.approx(gap)


def test_tied_ema_is_rejected_and_endpoints_are_explicit():
    ema = EmaSubspace(np.array([[1.0, 0.0]]))
    ema.update(np.array([[1.0, 0.0]]), 1.0)
    ema.update(np.array([[0.0, 1.0]]), 0.5)
    candidate, gap = ema.proposal()
    assert candidate is None and abs(gap) < 1e-14
    assert cosine_step(0, 8) == 0.5
    assert cosine_step(7, 8) == 0.05
    assert cosine_step(0, 1) == 0.5


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1, np.nan, np.inf])
def test_ema_rejects_invalid_step(alpha):
    ema = EmaSubspace(np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError):
        ema.update(np.array([[1.0, 0.0]]), alpha)


@pytest.mark.parametrize("bad", [np.zeros((1, 2)), np.ones((2, 2)), [[np.nan, 0.0]]])
def test_ema_rejects_invalid_frame(bad):
    with pytest.raises(ValueError):
        EmaSubspace(np.asarray(bad))


@pytest.mark.parametrize("ema", [False, True])
@pytest.mark.parametrize("tau", [0.0, 0.2])
@pytest.mark.parametrize("degenerate", [False, True])
def test_solver_full_reference_and_full_certificate(ema, tau, degenerate):
    rng = np.random.default_rng(21)
    B = basis(rng)
    U, I = rng.normal(size=(9, 5, 7)), rng.normal(size=(9, 5))
    if degenerate:
        U[0] = 0
        U[1, :, 1:] = 0
    mass = np.geomspace(1e-3, 1e3, 9)
    result = MiniBatchSolver(123, ema=ema)(B, U, I, mass=mass, local_ridge=tau)
    coeff, loss = reference(result.index, U, I, mass, tau)
    _, start_loss = reference(B, U, I, mass, tau)
    np.testing.assert_allclose(result.coefficients, coeff, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(result.index @ result.index.T, np.eye(2), atol=2e-13)
    assert loss <= start_loss + 2e-10
    d = result.diagnostics
    assert d["profile_objective"] == pytest.approx(loss, rel=2e-12, abs=2e-10)
    assert d["certificate_scope"] == "full_frozen_profile"
    assert d["mini_batch_calls"] == 8
    assert d["converged"] == (
        d["stationarity_applicable"] and d["riemannian_gradient"] <= 1e-6
    )
    state = g._profile(g._project(U, result.index.T), I, mass, tau)
    gradient = g._gradient(result.index.T, U, state, mass)
    assert d["gradient_norm"] == pytest.approx(np.linalg.norm(gradient), rel=2e-12)


def test_solver_is_deterministic_and_gauge_invariant():
    rng = np.random.default_rng(26)
    B = basis(rng)
    U, I = rng.normal(size=(13, 5, 7)), rng.normal(size=(13, 5))
    R = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    a = MiniBatchSolver(101, ema=True)(B, U, I)
    b = MiniBatchSolver(101, ema=True)(B, U, I)
    c = MiniBatchSolver(101, ema=True)(R @ B, U, I)
    np.testing.assert_array_equal(a.index, b.index)
    np.testing.assert_allclose(a.index.T @ a.index, c.index.T @ c.index, atol=2e-8)


def test_tiny_and_zero_operators_keep_full_rows():
    B = np.array([[1.0, 0.0, 0.0]])
    U, I = np.zeros((1, 2, 3)), np.ones((1, 2))
    result = MiniBatchSolver(20, ema=True)(B, U, I)
    np.testing.assert_allclose(result.index.T @ result.index, B.T @ B)
    np.testing.assert_array_equal(result.coefficients, np.zeros((1, 1)))
    assert result.diagnostics["profile_objective"] == 2
    assert result.diagnostics["mini_batch_calls"] == 2
    assert not result.diagnostics["stationarity_applicable"]
    assert not result.diagnostics["converged"]


def test_finalization_guard_rejects_increasing_polish(monkeypatch):
    original = g.solve
    B = np.array([[1.0, 0.0]])
    U, I = np.eye(2)[None], np.array([[1.0, 0.0]])

    def damaged(index, U, I, **kwargs):
        if kwargs["max_steps"] == 4:
            return HPAOResult(np.array([[0.0, 1.0]]), np.zeros((1, 1)), {})
        return original(index, U, I, **kwargs)

    monkeypatch.setattr(g, "solve", damaged)
    result = MiniBatchSolver(20, ema=True)(B, U, I)
    np.testing.assert_allclose(result.index.T @ result.index, B.T @ B)
    assert result.diagnostics["finalization_rejected"]
    assert result.diagnostics["profile_objective"] == 0


def test_descent_does_not_imply_edr():
    U, I, mass = np.diag([1.0, 2.0])[None], np.ones((1, 2)), np.ones(1)
    truth = np.array([[1.0, 0.0]])
    proposed = np.array([[2.0, 1.0]]) / np.sqrt(5)
    assert reference(truth, U, I, mass)[1] == 1
    assert reference(proposed, U, I, mass)[1] < 1e-28
    assert np.sqrt(1 - (truth @ proposed.T).item() ** 2) == pytest.approx(
        1 / np.sqrt(5)
    )


def test_public_fit_keeps_initial_data_and_moments():
    rng = np.random.default_rng(28)
    X = rng.normal(size=(120, 6))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2 + 0.3 * X[:, 2]
    config = ADP_Config(seed=210, N_J=60, N_phi=5, N_loc=10, outer_steps=3)
    inputs = []

    def recording(solver):
        def call(B, U, I, **kwargs):
            if not inputs or inputs[-1] is None:
                inputs.append((B.copy(), U.copy(), I.copy(), kwargs["mass"].copy()))
            return solver(B, U, I, **kwargs)

        return call

    original = ADP_multi_index(3, config, ADP_solver(recording(g.solve))).fit(X, y)
    inputs.append(None)
    modified = ADP_multi_index(
        3, config, ADP_solver(recording(MiniBatchSolver(400, ema=True)))
    ).fit(X, y)
    for a, b in zip(inputs[0], inputs[2], strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(
        original.result_.beta_init, modified.result_.beta_init
    )
    assert all(len(c) == config.N_J for c in [modified.coefficients_])
    assert all(
        row["solver"]["certificate_scope"] == "full_frozen_profile"
        for row in modified.trace_
    )


def test_campaign_gates_keep_failures_and_missing_pairs_visible():
    from experiments.e_means_adp.study import evaluate

    protocol = {
        "cases": [{"name": "a"}, {"name": "b"}],
        "seeds": [1, 2, 3],
        "methods": ["original20", "mini", "mini_ema"],
    }
    rows = [
        {
            "case": name,
            "seed": seed,
            "method": method,
            "status": "ok",
            "edr": 0.5,
            "max_sin_angle": 0.7,
            "orthogonality_error": 1e-15,
            "fit_seconds": 2 if method == "original20" else 1,
            "peak_rss_mib": 100,
            "all_inner_certified": False,
            "n_inner_certified": 0,
            "n_inner_total": 2,
            "full_guard_max_increase": 0,
        }
        for name in ("a", "b")
        for seed in (1, 2, 3)
        for method in protocol["methods"]
    ]
    passing = evaluate(rows, protocol)
    assert passing["candidates"]["mini"]["pass"]
    rows[1]["status"] = "timeout"
    rows.pop()
    failing = evaluate(rows, protocol)
    assert not failing["candidates"]["mini"]["pass"]
    assert not failing["candidates"]["mini_ema"]["pass"]
    assert failing["candidates"]["mini"]["failures"] == 1
    assert failing["candidates"]["mini_ema"]["missing"] == 1


@pytest.mark.parametrize("epsilon", [0.0, 1e-18, 1e-16, 1e-9])
def test_near_rank_cutoff_has_no_stationarity_certificate(epsilon):
    B = np.eye(4)[:2]
    U = np.tile(np.array([[1, 0, 0, 0], [0, epsilon, 0, 0], [0, 0, 0, 0]]), (5, 1, 1))
    result = MiniBatchSolver(45, ema=True)(B, U, np.ones((5, 3)))
    assert np.isfinite(result.index).all() and np.isfinite(result.coefficients).all()
    assert result.diagnostics["local_rank_loss"] == (5 if epsilon <= 1e-16 else 0)
    assert not result.diagnostics["stationarity_applicable"]
    assert not result.diagnostics["converged"]
    bad = U.copy()
    bad[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        MiniBatchSolver(45)(B, bad, np.ones((5, 3)))


def test_successive_outer_calls_replay_from_fresh_instance():
    rng = np.random.default_rng(46)
    B = basis(rng)
    inputs = [(rng.normal(size=(11, 5, 7)), rng.normal(size=(11, 5))) for _ in range(2)]
    first, replay = MiniBatchSolver(47, ema=True), MiniBatchSolver(47, ema=True)
    for U, I in inputs:
        old = (B.copy(), U.copy(), I.copy())
        a, b = first(B, U, I), replay(B, U, I)
        np.testing.assert_array_equal(a.index, b.index)
        assert a.diagnostics == b.diagnostics
        for live, saved in zip((B, U, I), old, strict=True):
            np.testing.assert_array_equal(live, saved)
        B = a.index
    assert first.calls == replay.calls == 2


def test_minibatch_false_certificates_never_replace_full_refit(monkeypatch):
    original = g.solve
    B = np.array([[1.0, 0.0]])
    U, I = np.eye(2)[None], np.array([[1.0, 0.0]])

    def bad_batch(index, U, I, **kwargs):
        if kwargs["max_steps"] == 1:
            return HPAOResult(
                np.array([[0.0, 1.0]]),
                np.array([[123.0]]),
                {
                    "accepted_steps": 1,
                    "profile_evaluations": 1,
                    "rank_guard_fallbacks": 0,
                    "converged": True,
                },
            )
        return original(index, U, I, **kwargs)

    monkeypatch.setattr(g, "solve", bad_batch)
    result = MiniBatchSolver(48, ema=True)(B, U, I)
    np.testing.assert_allclose(result.index.T @ result.index, B.T @ B)
    np.testing.assert_array_equal(result.coefficients, np.array([[1.0]]))
    assert result.diagnostics["selected_start"] == "input"
    assert result.diagnostics["profile_objective"] == 0


def test_frozen_workers_and_independent_audit_detect_tampering(tmp_path):
    from dataclasses import asdict

    from experiments.e_means_adp.audit import audit_stage
    from experiments.e_means_adp.study import freeze, stage, write_json
    from experiments.r_sdr.design import Case

    root = tmp_path / "campaign"
    protocol = freeze(root)
    protocol["cases"] = [asdict(Case("harness-smoke", 120, 6, 3))]
    protocol["selection_seeds"] = [1131350]
    protocol["methods"] = ["original20", "mini_ema"]
    write_json(root / "PROTOCOL.json", protocol)
    stage(root, protocol, "selection", [1131350], protocol["methods"])
    audited = audit_stage(root, "selection")
    assert audited["pass"] and audited["valid_fits"] == 2
    manifest = root / "selection/manifest.json"
    runs = root / "selection/runs.jsonl"
    old_manifest, old_runs = manifest.read_text(), runs.read_text()
    write_json(manifest, {"cases": [], "seeds": [], "methods": []})
    runs.write_text("")
    assert not audit_stage(root, "selection")["pass"]
    manifest.write_text(old_manifest)
    runs.write_text(old_runs)
    gates = root / "selection/gates.json"
    old_gates = gates.read_text()
    import json

    altered = json.loads(old_gates)
    altered["candidates"]["mini_ema"]["pass"] = not altered["candidates"]["mini_ema"][
        "pass"
    ]
    write_json(gates, altered)
    assert not audit_stage(root, "selection")["pass"]
    gates.write_text(old_gates)
    path = root / "selection/harness-smoke-1131350/mini_ema/basis.npz"
    saved = dict(np.load(path))
    saved["basis"] = np.roll(saved["basis"], 1, axis=0)
    np.savez(path, **saved)
    assert not audit_stage(root, "selection")["pass"]


def test_all_centers_profile_runs_actual_n_and_keeps_bounded_control(tmp_path):
    import json
    from dataclasses import asdict

    from experiments.e_means_adp.audit import audit_stage
    from experiments.e_means_adp.study import freeze, stage, write_json
    from experiments.r_sdr.design import Case

    root = tmp_path / "all-centers"
    protocol = freeze(root, all_centers=True)
    assert protocol["center_policy"] == "all"
    assert protocol["selection_seeds"] == list(range(1134400, 1134410))
    assert protocol["validation_seeds"] == list(range(1135400, 1135410))
    assert "original20_bounded" in protocol["methods"]
    assert protocol["controls"] == ["original20", "full6", "original20_bounded"]
    protocol["cases"] = [asdict(Case("all-centers-smoke", 240, 6, 3))]
    protocol["selection_seeds"] = [1134350]
    protocol["methods"] = ["original20", "mini_ema", "original20_bounded"]
    write_json(root / "PROTOCOL.json", protocol)
    stage(root, protocol, "selection", protocol["selection_seeds"], protocol["methods"])
    audited = audit_stage(root, "selection")
    assert audited["pass"] and audited["valid_fits"] == 3
    block = root / "selection/all-centers-smoke-1134350"
    for method in protocol["methods"]:
        job_path = block / method / "job.json"
        job = json.loads(job_path.read_text())
        assert job["config"]["N_J"] == (200 if method.endswith("bounded") else 240)
    path = block / "original20/job.json"
    job = json.loads(path.read_text())
    job["config"]["N_J"] = 200
    write_json(path, job)
    audited = audit_stage(root, "selection")
    assert not audited["pass"]
    assert any("job_config:" in failure for failure in audited["failures"])
    job["config"]["N_J"] = 240
    write_json(path, job)
    trace_path = block / "original20/trace.json"
    trace = json.loads(trace_path.read_text())
    trace["N_J"] = 200
    write_json(trace_path, trace)
    audited = audit_stage(root, "selection")
    assert not audited["pass"]
    assert any("actual_N_J:" in failure for failure in audited["failures"])
