"""Keep independent hypothesis gates and protect held-out data from source drift."""

import numpy as np
import pytest

from experiments import light_metric_study as study


def test_one_candidate_numerical_failure_does_not_reject_another(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "OUT", tmp_path)
    monkeypatch.setattr(study, "SEEDS", {"selection": (1,)})
    directory = tmp_path / "selection"
    (directory / "fits").mkdir(parents=True)
    (directory / "inputs").mkdir()
    (directory / "sources.zip").write_bytes(b"test archive")
    arms = ("identity", "right", "feature")
    jobs = [(*case, 1, arm) for case in study.CASES for arm in arms]
    manifest = {
        "jobs": jobs,
        "arms": arms,
        "fingerprint": "fixture",
        "sources": {},
        "inputs": {},
        "archive_sha256": study.sha(directory / "sources.zip"),
    }
    study.write(directory / "manifest.json", manifest)
    for link, m in study.CASES:
        truth = np.eye(6)[:, :m]
        np.savez(directory / "inputs" / f"{link}-m{m}-1.npz", truth=truth)
        for arm in arms:
            basis = truth.copy()
            angle = 0.3 if arm == "identity" else 0.2
            basis[:, -1] = (
                np.cos(angle) * truth[:, -1] + np.sin(angle) * np.eye(6)[:, -1]
            )
            row = {
                "link": link,
                "m": m,
                "seed": 1,
                "arm": arm,
                "fingerprint": "fixture",
                "status": "ok",
                "basis": basis,
                "edr": float(np.sin(angle) / np.sqrt(m)),
                "fit_seconds": 1.0,
                "process_seconds": 1.0,
                "process_peak_rss_mib": 100,
                "stages": [
                    {
                        "converged": True,
                        "profile_history": [1, 0.5],
                        "normal_residual_max": 1e-4 if arm == "right" else 1e-12,
                    }
                ],
                **{
                    name: "paired"
                    for name in (
                        "data_sha256",
                        "initial_sha256",
                        "first_weights_sha256",
                        "first_directions_sha256",
                    )
                },
            }
            study.write(directory / "fits" / f"{link}-{m}-{arm}.json", row)
    result = study.analyze("selection")
    assert result["candidates"]["feature"]["gates"]["numerical"]
    assert not result["candidates"]["right"]["gates"]["numerical"]


@pytest.mark.parametrize("failure", ["source drift", "stale audit"])
def test_source_or_audit_drift_rejected_before_validation(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(study, "OUT", tmp_path)
    selected = tmp_path / "selection"
    selected.mkdir()
    source = tmp_path / "source.py"
    source.write_text("before")
    (selected / "sources.zip").write_bytes(b"archive")
    manifest = {
        "sources": {str(source): study.sha(source)},
        "archive_sha256": study.sha(selected / "sources.zip"),
        "inputs": {},
    }
    study.write(selected / "manifest.json", manifest)
    study.write(selected / "summary.json", {"passed_arms": ["right"]})
    study.write(
        selected / "independent_audit.json",
        {"integrity_ok": True, "agreed_passed_arms": ["right"]},
    )
    if failure == "source drift":
        source.write_text("after")

    def expensive_freeze(phase, arms):
        (tmp_path / phase / "inputs").mkdir(parents=True)
        raise AssertionError("held-out data consumed before source verification")

    monkeypatch.setattr(study, "freeze", expensive_freeze)
    with pytest.raises(RuntimeError, match=failure):
        study.run("validation")
    assert not (tmp_path / "validation").exists()
