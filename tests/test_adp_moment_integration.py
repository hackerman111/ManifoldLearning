"""Reference and integration contracts for the opt-in ADP experiment."""

import numpy as np
import pytest

from experiments.adp_moment_integration import fit, frozen_profile, propose
from experiments.weak_moments import fit as moment_fit


def problem():
    rng = np.random.default_rng(3020011)
    index = np.linalg.qr(rng.normal(size=(7, 2)))[0].T
    u, target = rng.normal(size=(6, 5, 7)), rng.normal(size=(6, 5))
    mass = np.geomspace(1e-4, 3, 6)
    return index, u, target, mass


def test_frozen_profile_matches_independent_lstsq_including_rank_loss():
    index, u, target, mass = problem()
    u[0] = 0
    u[1, :, :] = u[1, :1, :]
    result = frozen_profile(index, u, target, mass)
    reference = sum(
        a
        * np.linalg.norm(
            y - (v @ index.T) @ np.linalg.lstsq(v @ index.T, y, rcond=None)[0]
        )
        ** 2
        for a, v, y in zip(mass, u, target, strict=True)
    )
    assert result.value == pytest.approx(reference, rel=2e-13)


def test_proposal_strict_acceptance_orthogonality_and_rejection():
    index = np.eye(5)[:2]
    u = np.broadcast_to(np.eye(5), (3, 5, 5)).copy()
    target = np.tile([1.0, 0.0, 2.0, 0.0, 0.0], (3, 1))
    h = np.diag([0.0, 0.0, 2.0, 0.0, 0.0])
    proposed, diag = propose(index, u, target, np.ones(3), h)
    assert diag["accepted"]
    assert diag["after"] < diag["before"]
    np.testing.assert_allclose(proposed @ proposed.T, np.eye(2), atol=1e-14)
    rejected, no = propose(
        index, u, np.tile([1.0, 1.0, 0.0, 0.0, 0.0], (3, 1)), np.ones(3), h
    )
    assert not no["accepted"]
    np.testing.assert_array_equal(index, rejected)
    for degenerate in (np.zeros((5, 5)), np.diag([1.0, 1.0, 0.0, 0.0, 0.0])):
        same, no = propose(index, u, target, np.ones(3), degenerate)
        assert not no["accepted"]
        np.testing.assert_array_equal(index, same)
    with pytest.raises(ValueError):
        propose(index, u, target, np.ones(3), np.full((5, 5), np.nan))


def test_live_outer_loop_pairing_and_initializer_restoration():
    from ADP.engine.common import index_fit

    rng = np.random.default_rng(3020012)
    x = rng.normal(size=(90, 5))
    y = 2 * x[:, 0] + 0.2 * (x[:, 1] ** 2 - 1) + 0.1 * rng.normal(size=len(x))
    original = index_fit._initial_index
    options = {"N_lin": 15, "N_J": 9, "N_loc": 10, "N_phi": 6, "outer_steps": 2}
    a = fit(x, y, 3020012, "adp", config_overrides=options)
    b = fit(x, y, 3020012, "budget", config_overrides=options)
    c = fit(x, y, 3020012, "proposal", config_overrides=options)
    r = fit(x, y, 3020012, "residual_start", config_overrides=options)
    assert original is index_fit._initial_index
    for z in (b, c):
        assert z["diagnostics"]["initial_hash"] == a["diagnostics"]["initial_hash"]
        assert (
            z["diagnostics"]["stages"][0]["weights"]
            == a["diagnostics"]["stages"][0]["weights"]
        )
        assert (
            z["diagnostics"]["stages"][0]["directions"]
            == a["diagnostics"]["stages"][0]["directions"]
        )
    expected = moment_fit(x, y)["basis"]
    np.testing.assert_allclose(
        r["initial_basis"] @ r["initial_basis"].T, expected @ expected.T, atol=1e-13
    )
    for row in (a, b, c, r):
        np.testing.assert_allclose(row["basis"].T @ row["basis"], np.eye(2), atol=1e-12)
    with pytest.raises(ValueError):
        fit(x, y, 1, "unknown", config_overrides=options)
    with pytest.raises(ValueError):
        fit(x, y, 1, "residual_start", config_overrides={**options, "N_loc": 91})
    assert original is index_fit._initial_index


def test_protocol_serializes_slotted_config_and_forbids_overwrite(
    tmp_path, monkeypatch
):
    import json

    from experiments import integration_study as study

    monkeypatch.setattr(study, "OUT", tmp_path)
    monkeypatch.setattr(study, "SEEDS", {"selection": [3020000]})
    study.prepare()
    protocol = json.loads((tmp_path / "PROTOCOL.json").read_text())
    assert protocol["configuration"]["N_loc"] == 20
    assert protocol["configuration"]["outer_steps"] == 8
    assert "maximum" in protocol["configuration"]["kernel"]
    with pytest.raises(RuntimeError, match="already frozen"):
        study.prepare()
