import numpy as np

from experiments.two_sided_meanmave import (
    ARMS,
    CELLS,
    SEED_VALUES,
    TimeOnlyProfiler,
    summarize,
)


def _synthetic_rows():
    rows = []
    for link, m in CELLS:
        for seed in SEED_VALUES:
            for arm in ARMS:
                edr = 0.2 if arm == "two_sided" else 0.1
                seconds = 1.0 if arm == "two_sided" else 10.0
                rows.append(
                    {
                        "link": link,
                        "m": m,
                        "seed": seed,
                        "arm": arm,
                        "status": "ok",
                        "edr": edr,
                        "recovered": edr <= 0.1,
                        "fit_seconds": seconds,
                        "input_sha256": f"{link}-{m}-{seed}",
                        "process": {"process_seconds": seconds + 0.5},
                        "projector_distance_to_saved_two_sided": 0.0,
                    }
                )
    return rows


def test_profiler_records_stages_without_enabling_tracemalloc():
    import tracemalloc

    tracing_before = tracemalloc.is_tracing()
    profiler = TimeOnlyProfiler()
    with profiler.stage("one"):
        sum(range(10))
    assert profiler.model_profile()["allocation_tracing"] is False
    assert profiler.model_profile()["stage_seconds"]["one"] >= 0
    assert tracemalloc.is_tracing() is tracing_before


def test_summary_uses_paired_orientation_and_seed_clusters():
    summary = summarize(_synthetic_rows())
    assert summary["planned_fits"] == 120
    assert summary["successful_fits"] == 120
    assert summary["complete_pairs"] == 60
    assert summary["complete_seed_clusters"] == 10
    assert summary["pairing_ok"] is True
    assert np.isclose(summary["aggregate_mean_delta_edr_two_sided_minus_meanmave"], 0.1)
    assert np.allclose(summary["aggregate_seed_cluster_bootstrap_95_ci"], [0.1, 0.1])
    assert summary["meanmave_more_accurate_if_delta_positive"] is True
    assert (
        summary["meanmave_over_two_sided_time_ratio_above_one_means_two_sided_faster"]
        is True
    )
    assert all(cell["recovered"]["MEANMAVE"] == 10 for cell in summary["cell_summary"])
    assert all(cell["recovered"]["two_sided"] == 0 for cell in summary["cell_summary"])


def test_cluster_bootstrap_seed_mean_includes_all_six_cells():
    rows = _synthetic_rows()
    summary = summarize(rows)
    values = np.asarray(list(summary["seed_cluster_mean_delta_edr"].values()))
    assert values.shape == (10,)
    assert np.allclose(values, 0.1)
