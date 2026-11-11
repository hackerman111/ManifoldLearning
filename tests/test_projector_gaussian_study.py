"""Freeze boundaries and the predeclared, independent benchmark cases."""

import numpy as np
import pytest


def test_schedule_is_complete_unique_and_uses_fresh_disjoint_seeds():
    from experiments.projector_gaussian_study import SEEDS, schedule

    expected = {"selection": 600, "validation": 900, "stress": 300}
    for phase, count in expected.items():
        jobs = schedule(phase)
        assert len(jobs) == len({j["id"] for j in jobs}) == count
    assert not set(SEEDS["selection"]) & set(SEEDS["validation"])
    assert not set(SEEDS["stress"]) & (
        set(SEEDS["selection"]) | set(SEEDS["validation"])
    )


def test_heldout_is_unavailable_before_independent_pass(tmp_path):
    from experiments.projector_gaussian_study import require_previous_pass

    with pytest.raises(RuntimeError, match="audit"):
        require_previous_pass("validation", tmp_path)
    with pytest.raises(RuntimeError, match="audit"):
        require_previous_pass("stress", tmp_path)


def test_generators_are_deterministic_and_all_truth_directions_are_present():
    from experiments.projector_gaussian_study import cases, generate

    for spec in cases("stress"):
        tiny = {**spec, "n": 100}
        first = generate(tiny, 2840099)
        second = generate(tiny, 2840099)
        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_equal(a, b)
            assert np.isfinite(a).all()
        x, y, truth = first
        assert x.shape == (100, spec["d"])
        assert y.shape == (100,)
        np.testing.assert_allclose(truth.T @ truth, np.eye(spec["m"]), atol=1e-12)


def test_timeout_keeps_input_identity_for_independent_audit(tmp_path, monkeypatch):
    import json

    from experiments import projector_gaussian_study as study

    phase = tmp_path / "selection"
    (phase / "fits").mkdir(parents=True)
    (phase / "logs").mkdir()
    job = {"id": "example-two_sided", "case_id": "example", "arm": "two_sided"}
    item = {"seed": 2841000, "case": {"m": 5}, "array_sha256": "saved-input"}
    (phase / "manifest.json").write_text(json.dumps({"inputs": {"example": item}}))
    monkeypatch.setattr(study, "OUT", tmp_path)
    monkeypatch.setattr(
        study,
        "run_process",
        lambda *a, **kw: {
            "status": "timeout",
            "returncode": -9,
            "peak_rss_mib": 100.0,
        },
    )
    row = study.run_one("selection", job, {})
    assert row["status"] == "timeout"
    assert row["seed"] == item["seed"]
    assert row["data_sha256"] == item["array_sha256"]
