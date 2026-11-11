"""Unit checks for the frozen anisotropic-geometry benchmark analysis."""

import itertools

import numpy as np
import pytest

from experiments import anisotropic_fixed_windows as fixed
from experiments import anisotropic_geometry_study as study


def paired_rows(delta=-0.01):
    rows = []
    for seed in range(1286000, 1286010):
        for link, m in itertools.product(("additive", "quadratic", "sine"), (3, 5)):
            for arm, edr in ((study.BASELINE, 0.15), ("left", 0.15 + delta)):
                rows.append(
                    {
                        "link": link,
                        "m": m,
                        "seed": seed,
                        "arm": arm,
                        "status": "ok",
                        "edr": edr,
                        "method_seed": seed + 1,
                        "fit_seconds": 1.0 if arm == study.BASELINE else 1.2,
                        "process_seconds": 1.1 if arm == study.BASELINE else 1.21,
                        "process_peak_rss_mib": 100.0
                        if arm == study.BASELINE
                        else 105.0,
                        "data_sha256": f"data-{link}-{m}-{seed}",
                        "initial_sha256": f"init-{seed}",
                        "first_weights_sha256": f"weights-{seed}",
                        "first_directions_sha256": f"directions-{seed}",
                        "first_moments_sha256": f"moments-{seed}",
                        "first_tau": 0.01,
                        "selected_iteration": 2,
                        "stages": [{"iterations": 2, "converged": True}],
                    }
                )
    return rows


def test_cluster_bootstrap_resamples_six_cells_as_seed_clusters():
    pairs = [
        {"seed": seed, "delta_edr": -0.01}
        for seed in range(1286000, 1286010)
        for _ in study.CASES
    ]
    result = study.cluster_bootstrap_upper(pairs)
    assert result["quantile"] == 0.99
    assert result["upper"] == pytest.approx(-0.01)
    assert len(result["cluster_means"]) == 10


def test_cluster_bootstrap_rejects_incomplete_seed_cluster():
    pairs = [{"seed": 1286000, "delta_edr": -0.01} for _ in range(len(study.CASES) - 1)]
    result = study.cluster_bootstrap_upper(pairs)
    assert result["upper"] is None
    assert result["interval95"] is None


def test_candidate_gates_use_paired_full_seed_schedule():
    result = study.summarize_candidate(paired_rows(), "left", range(1286000, 1286010))
    assert result["complete_pairs"] == 60
    assert result["mean_delta_edr"] == pytest.approx(-0.01)
    assert result["familywise_upper"] == pytest.approx(-0.01)
    assert result["median_fit_ratio"] == pytest.approx(1.2)
    assert result["median_rss_ratio"] == pytest.approx(1.05)
    assert all(result["gates"].values())
    assert result["passed"]


def test_missing_fit_fails_completeness_and_all_promotion_gates():
    rows = paired_rows()
    rows.pop()
    result = study.summarize_candidate(rows, "left", range(1286000, 1286010))
    assert result["complete_pairs"] == 59
    assert not result["gates"]["complete"]
    assert not result["gates"]["familywise_cluster_upper"]
    assert not result["passed"]


def test_fixed_window_cases_have_requested_spd_and_rank_mixture():
    full = fixed.synthetic(20261026, 40, "spd")
    mixed = fixed.synthetic(20261026, 80, "mixed")
    assert full["covariance"].shape == (40, 25, 25)
    assert set(full["ranks"]) == {25}
    assert mixed["covariance"].shape == (80, 25, 25)
    assert set(mixed["ranks"]) == {0, 19, 25}
    assert mixed["ranks"][-1] == 0
    assert mixed["mass"].sum() == pytest.approx(1.0)
    assert np.linalg.norm(mixed["cross"][-1]) == 0
    assert np.linalg.norm(mixed["ideal_cross"][-1]) == 0


def test_ideal_zero_ridge_profile_and_tangent_curvature_keep_zero_window():
    from experiments import anisotropic_metrics as geometry
    from experiments import full_ridge_batched as ridge

    case = fixed.synthetic(20261026, 40, "mixed")
    problem = ridge.build(case["stats"], case["initial"])
    problem.tau = 0.0
    problem.reference_problem = None
    state = fixed.ideal_state(case["initial"], problem)
    operator = geometry.TangentGeometry(case["initial"], problem, state)
    assert state.coefficients[-1].tolist() == [0.0] * 5
    assert state.normal_residual <= 1e-10
    assert np.linalg.eigvalsh(operator.matrix("schur")).min() >= -1e-9
    assert np.linalg.eigvalsh(operator.matrix("varpro")).min() >= -1e-9


def test_ideal_zero_residual_has_equal_sc_vp_and_half_hessian():
    from types import SimpleNamespace

    from experiments import anisotropic_metrics as geometry
    from experiments import full_ridge_batched as ridge

    case = fixed.synthetic(20261026, 40, "mixed")
    stats = SimpleNamespace(
        U=case["covariance"], I=case["ideal_cross"], mass=case["mass"]
    )
    problem = ridge.build(stats, case["truth"])
    problem.tau = 0.0
    problem.reference_problem = None
    state = fixed.ideal_state(case["truth"], problem)
    operator = geometry.TangentGeometry(case["truth"], problem, state)
    schur, varpro, hessian = (
        operator.matrix(kind) for kind in ("schur", "varpro", "hessian")
    )
    scale = max(np.linalg.norm(schur), 1e-300)
    assert state.normal_residual <= 1e-10
    assert np.linalg.norm(schur - varpro) / scale <= 1e-8
    assert np.linalg.norm(schur - hessian) / scale <= 1e-8
