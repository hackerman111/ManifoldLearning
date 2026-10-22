from __future__ import annotations

from experiments.grassman_gpu_2026_10_22.benchmark import acceptance, heavy_settings
from experiments.r_sdr.design import Case


def test_heavy_settings_respects_minimum_centers():
    from ADP import ADP_Config
    from ADP.engine.common.index_fit import validate_sizes

    case = Case("heavy", 10000, 50, 5)
    settings = heavy_settings(case, {})
    assert settings["N_J"] == 500
    validate_sizes(ADP_Config(**settings), "multi", case.n, case.d, case.m)


def test_acceptance_requires_complete_paired_stage():
    rows = [
        {
            "seed": seed,
            "method": method,
            "status": "ok",
            "projector_distance": 0.05,
            "fit_seconds": 2.0,
        }
        for seed in range(10)
        for method in ("GPU", "MEANMAVE")
    ]
    assert acceptance(rows, "GPU", expected_seeds=list(range(10)))["passed"]
    assert not acceptance(rows[:-1], "GPU", expected_seeds=list(range(10)))["passed"]
    rows[-1]["status"] = "timeout"
    assert not acceptance(rows, "GPU", expected_seeds=list(range(10)))["passed"]


def test_acceptance_rejects_quality_or_time_regression():
    rows = [
        {
            "seed": seed,
            "method": method,
            "status": "ok",
            "projector_distance": 0.1 if method == "GPU" else 0.05,
            "fit_seconds": 1.0 if method == "GPU" else 2.0,
        }
        for seed in range(10)
        for method in ("GPU", "MEANMAVE")
    ]
    assert not acceptance(rows, "GPU", expected_seeds=list(range(10)))["passed"]
    for row in rows:
        row["projector_distance"] = 0.05
        if row["method"] == "GPU":
            row["fit_seconds"] = 3.0
    assert not acceptance(rows, "GPU", expected_seeds=list(range(10)))["passed"]
