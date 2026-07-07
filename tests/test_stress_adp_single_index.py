import json

import matplotlib
import numpy as np
import pytest

from adp.evaluation import stress


def test_stress_profiles_cover_five_scales_and_required_axes():
    profiles = stress.stress_profiles()

    assert list(profiles) == ["smoke", "quick", "medium", "large", "extreme"]
    assert profiles["smoke"].max_n < profiles["quick"].max_n < profiles["medium"].max_n
    assert profiles["medium"].max_n < profiles["large"].max_n < profiles["extreme"].max_n
    assert profiles["smoke"].max_d < profiles["quick"].max_d < profiles["medium"].max_d
    assert profiles["medium"].max_d < profiles["large"].max_d < profiles["extreme"].max_d

    for profile in profiles.values():
        assert profile.seeds
        assert profile.links
        assert profile.noise_levels
        assert profile.sigma_x_values
        assert profile.corr_values
        assert profile.q_values
        assert 0.0 < min(profile.q_values) <= max(profile.q_values) <= 1.0


def test_build_cases_tracks_data_localization_and_optimizer_parameters():
    case = stress.build_cases(["quick"], base_seed=123, max_cases=1)[0]
    record = case.to_manifest_record()

    expected = {
        "profile",
        "seed",
        "data_seed",
        "fit_seed",
        "n",
        "d",
        "sigma_eps",
        "sigma_x",
        "corr",
        "q",
        "q_definition",
        "beta_support_size",
        "link",
        "kernel",
        "n_centers",
        "theta_centers",
        "n_directions",
        "min_neighbors",
        "lambda_penalty",
        "outer_steps",
        "inner_steps",
        "bandwidth_decay",
        "anisotropy_min",
        "renew_directions",
        "algorithm_variant",
        "directions_distribution_step0",
        "directions_distribution_stepk",
        "localizing_tensor_form",
    }
    assert expected.issubset(record)
    assert record["algorithm_variant"] == "single-index-random-projection"
    assert record["q_definition"].startswith("fraction of active coordinates")
    assert record["theta_centers"] == pytest.approx(record["n_centers"] / record["n"])
    assert 1 <= record["beta_support_size"] <= record["d"]


def test_run_case_smoke_records_internal_statistics_and_performance(tmp_path):
    case = stress.build_cases(["smoke"], base_seed=7, max_cases=1)[0]

    record = stress.run_case(case, show_progress=False)

    required = {
        "failed",
        "error",
        "cosine_abs",
        "angle_deg",
        "y_mean",
        "y_std",
        "y_outlier_frac_3sigma",
        "h0",
        "h_final",
        "rho_final",
        "N_min",
        "N_mean",
        "N_frac_below_min_neighbors",
        "imav_shape",
        "U_shape",
        "directions_shape",
        "U_beta_abs_mean",
        "U_beta_denominator_min",
        "residual",
        "beta_norm",
        "beta_delta_last",
        "inner_iterations_total",
        "cg_iterations_total",
        "cg_info_failures",
        "statistics_time_sec",
        "solve_time_sec",
        "fit_time_sec",
        "peak_memory_kib",
        "estimated_U_storage_kib",
        "estimated_weights_matrix_kib",
        "has_dxd_normal_matrix",
        "has_local_d_plus_1_regression",
        "uses_matrix_free_beta_update",
    }
    assert required.issubset(record)
    assert not record["failed"], record["error"]
    assert record["beta_norm"] == pytest.approx(1.0)
    assert 0.0 <= record["cosine_abs"] <= 1.0
    assert record["uses_matrix_free_beta_update"]
    assert not record["has_dxd_normal_matrix"]
    assert not record["has_local_d_plus_1_regression"]
    assert tuple(json.loads(record["U_shape"]))[-1] == case.d
    assert np.isfinite(record["peak_memory_kib"])


def test_dry_run_writes_manifest_without_runtime_metrics(tmp_path):
    exit_code = stress.main(
        [
            "--profile",
            "smoke",
            "--dry-run",
            "--output",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "adp_single_index_stress_records.csv").exists()
    assert (tmp_path / "adp_single_index_stress_summary.csv").exists()
    manifest = json.loads((tmp_path / "adp_single_index_stress_manifest.json").read_text())
    assert manifest["records"] == 1


def test_write_outputs_saves_required_stress_plots(tmp_path):
    case = stress.build_cases(["smoke"], base_seed=5, max_cases=1)[0]
    record = case.to_manifest_record()
    record.update(
        {
            "failed": False,
            "case_passed_quality_gate": True,
            "cosine_abs": 0.93,
            "angle_deg": 21.5,
            "N_min": 4.0,
            "N_mean": 7.5,
            "N_frac_below_min_neighbors": 0.1,
            "U_beta_denominator_min": 0.02,
            "U_beta_abs_mean": 0.3,
            "residual": 1.2,
            "cg_iterations_total": 8,
            "statistics_time_sec": 0.12,
            "solve_time_sec": 0.05,
            "fit_time_sec": 0.2,
            "peak_memory_kib": 256.0,
            "complexity_proxy_nJ_n_d_P": case.n_centers * case.n * case.d * case.n_directions,
        }
    )

    paths = stress.write_outputs([record], tmp_path, breakdown_threshold=0.7)

    expected_plot_keys = {
        "quality_plot",
        "time_plot",
        "memory_plot",
        "localization_plot",
        "optimization_plot",
    }
    assert expected_plot_keys.issubset(paths)
    for key in expected_plot_keys:
        assert paths[key].exists()
        assert paths[key].suffix == ".png"

    manifest = json.loads((tmp_path / "adp_single_index_stress_manifest.json").read_text())
    assert expected_plot_keys.issubset(manifest["plots"])


def test_localization_plot_places_legend_outside_axes(tmp_path, monkeypatch):
    case = stress.build_cases(["smoke"], base_seed=6, max_cases=1)[0]
    record = case.to_manifest_record()
    record.update(
        {
            "failed": False,
            "case_passed_quality_gate": True,
            "cosine_abs": 0.91,
            "angle_deg": 24.0,
            "N_min": 3.5,
            "N_mean": 7.0,
            "N_frac_below_min_neighbors": 0.25,
            "U_beta_denominator_min": 0.03,
            "cg_iterations_total": 5,
            "fit_time_sec": 0.2,
            "peak_memory_kib": 128.0,
            "complexity_proxy_nJ_n_d_P": case.n_centers * case.n * case.d * case.n_directions,
        }
    )
    captured = {}
    original_save_figure = stress.save_figure

    def capture_save_figure(fig, path, *, dpi=150, close=False):
        if path.name == "stress_localization_mass.png":
            legend = fig.axes[0].get_legend()
            captured["legend_x0"] = legend.get_bbox_to_anchor()._bbox.x0
            captured["legend_loc"] = legend._loc
        return original_save_figure(fig, path, dpi=dpi, close=close)

    monkeypatch.setattr(stress, "save_figure", capture_save_figure)

    stress.write_outputs([record], tmp_path, breakdown_threshold=0.7)

    assert captured["legend_x0"] > 1.0
    assert captured["legend_loc"] == 2


def test_beta_update_plot_places_legend_outside_axes(tmp_path, monkeypatch):
    case = stress.build_cases(["smoke"], base_seed=7, max_cases=1)[0]
    record = case.to_manifest_record()
    record.update(
        {
            "failed": False,
            "case_passed_quality_gate": True,
            "cosine_abs": 0.92,
            "angle_deg": 23.0,
            "N_min": 4.0,
            "N_mean": 7.5,
            "N_frac_below_min_neighbors": 0.15,
            "U_beta_denominator_min": 0.04,
            "cg_iterations_total": 6,
            "fit_time_sec": 0.22,
            "peak_memory_kib": 140.0,
            "complexity_proxy_nJ_n_d_P": case.n_centers * case.n * case.d * case.n_directions,
        }
    )
    captured = {}
    original_save_figure = stress.save_figure

    def capture_save_figure(fig, path, *, dpi=150, close=False):
        if path.name == "stress_beta_update_stability.png":
            legend = fig.axes[0].get_legend()
            captured["legend_x0"] = legend.get_bbox_to_anchor()._bbox.x0
            captured["legend_loc"] = legend._loc
        return original_save_figure(fig, path, dpi=dpi, close=close)

    monkeypatch.setattr(stress, "save_figure", capture_save_figure)

    stress.write_outputs([record], tmp_path, breakdown_threshold=0.7)

    assert captured["legend_x0"] > 1.0
    assert captured["legend_loc"] == 2


def test_latex_mode_enables_usetex_and_latex_labels(tmp_path, monkeypatch):
    case = stress.build_cases(["smoke"], base_seed=8, max_cases=1)[0]
    record = case.to_manifest_record()
    record.update(
        {
            "failed": False,
            "case_passed_quality_gate": True,
            "cosine_abs": 0.94,
            "angle_deg": 20.0,
            "N_min": 4.0,
            "N_mean": 8.0,
            "N_frac_below_min_neighbors": 0.1,
            "U_beta_denominator_min": 0.05,
            "cg_iterations_total": 4,
            "fit_time_sec": 0.18,
            "peak_memory_kib": 120.0,
            "complexity_proxy_nJ_n_d_P": case.n_centers * case.n * case.d * case.n_directions,
        }
    )
    captured = {}

    def capture_save_figure(fig, path, *, dpi=150, close=False):
        if path.name == "stress_quality_by_dimension.png":
            captured["quality_ylabel"] = fig.axes[0].get_ylabel()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr(stress, "save_figure", capture_save_figure)

    stress.write_outputs([record], tmp_path, breakdown_threshold=0.7, use_latex=True)

    manifest = json.loads((tmp_path / "adp_single_index_stress_manifest.json").read_text())
    assert manifest["latex_plots"] is True
    assert "russian" in manifest["latex_preamble"]
    assert matplotlib.rcParams["text.usetex"] is True
    assert r"\hat{\beta}" in captured["quality_ylabel"]
    stress.configure_stress_matplotlib(use_latex=False)
