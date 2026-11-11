"""Numerical identities and pairing guards for the diagnostic experiment."""

import numpy as np
import pytest

from experiments import init_rank_diagnosis as study


def test_population_contrast_is_weak_but_identifiable():
    matrix = study.population_opg(5)
    contrast = np.array([0, 0, 1, -1, 0]) / np.sqrt(2)
    eigenvalue = float(contrast @ matrix @ contrast)
    assert 0 < eigenvalue < 0.02
    np.testing.assert_allclose(matrix @ contrast, eigenvalue * contrast, atol=1e-12)
    assert np.linalg.eigvalsh(matrix)[0] > 0


def test_geometry_detects_missing_contrast_despite_good_coordinate_capture():
    truth = np.eye(7)[:, :5]
    basis = np.column_stack(
        (truth[:, :2], truth[:, 2:].sum(axis=1) / np.sqrt(3), np.eye(7)[:, 5:])
    )
    result = study.geometry(basis, truth)
    assert result["edr"] == pytest.approx(np.sqrt(2 / 5))
    assert result["contrast_leakage"] == pytest.approx(1)
    assert result["common_leakage"] == pytest.approx(0, abs=1e-14)


def test_paired_summary_rejects_missing_and_duplicate_pairs():
    rows = [
        {"m": 2, "seed": 1, "arm": "current", "status": "ok", "edr": 0.2},
        {"m": 2, "seed": 1, "arm": "rank1", "status": "ok", "edr": 0.1},
        {"m": 2, "seed": 2, "arm": "rank1", "status": "failed"},
    ]
    result = study.paired(rows, 2, "rank1")
    assert result["n_pairs"] == 1
    assert result["n_attempts"] == 2
    assert result["mean_delta"] == pytest.approx(-0.1)
    with pytest.raises(ValueError, match="duplicate"):
        study.paired(rows + rows[:1], 2, "rank1")
