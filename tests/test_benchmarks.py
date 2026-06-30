import numpy as np

from adp.benchmarks import BenchmarkScenario, run_benchmark_suite, save_benchmark_report


def test_benchmark_suite_compares_adp_with_ready_edr_baselines(tmp_path):
    scenario = BenchmarkScenario(
        name="quick_linear",
        n=90,
        d=4,
        n_centers=14,
        n_directions=4,
        min_neighbors=5,
        outer_steps=1,
        inner_steps=3,
        noise=0.02,
        corr=0.2,
        link="linear",
        trials=1,
    )

    frame = run_benchmark_suite(
        [scenario],
        methods=("adp_new", "statsmodels_sir", "sklearn_pls"),
        random_state=10,
        show_progress=False,
    )

    assert {"adp_new", "statsmodels_sir", "sklearn_pls"} == set(frame["method"])
    assert {"scenario", "trial", "method", "cosine_abs", "angle_deg", "fit_time_sec"}.issubset(frame.columns)
    assert np.all(np.isfinite(frame["fit_time_sec"]))
    assert np.all((0.0 <= frame["cosine_abs"]) & (frame["cosine_abs"] <= 1.0))
    ready_baselines = frame[frame["method"].isin(["statsmodels_sir", "sklearn_pls"])]
    assert np.all(ready_baselines["cosine_abs"] > 0.7)

    saved = save_benchmark_report(frame, tmp_path, prefix="quick")

    assert saved["csv"].exists()
    assert saved["quality_plot"].exists()
    assert saved["time_plot"].exists()
    assert saved["csv"].read_text().startswith("scenario,trial,method")
