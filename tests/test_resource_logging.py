import time

import pytest
import numpy as np

from adp import ADP, ADPConfig
from adp.common.resource_monitor import ResourceMonitor, ResourceUsage


def test_resource_monitor_reports_ordered_rss_and_elapsed_time():
    with ResourceMonitor(sample_interval_sec=0.001) as monitor:
        time.sleep(0.004)

    usage = monitor.usage

    assert usage.elapsed_sec > 0.0
    assert usage.samples >= 2
    assert 0.0 < usage.rss_min_mib <= usage.rss_mean_mib <= usage.rss_max_mib
    assert usage.rss_peak_delta_mib >= 0.0
    assert usage.source in {"psutil", "procfs", "resource"}


def test_resource_usage_flattens_with_prefix():
    usage = ResourceUsage(
        elapsed_sec=1.25,
        rss_start_mib=100.0,
        rss_min_mib=99.0,
        rss_mean_mib=105.0,
        rss_max_mib=112.0,
        rss_peak_delta_mib=12.0,
        samples=8,
        source="psutil",
    )

    assert usage.to_dict("algorithm") == {
        "algorithm_time_sec": 1.25,
        "algorithm_rss_start_mib": 100.0,
        "algorithm_rss_min_mib": 99.0,
        "algorithm_rss_mean_mib": 105.0,
        "algorithm_rss_max_mib": 112.0,
        "algorithm_rss_peak_delta_mib": 12.0,
        "algorithm_memory_samples": 8,
        "algorithm_memory_source": "psutil",
    }


def test_resource_monitor_rejects_nonpositive_sampling_interval():
    with pytest.raises(ValueError, match="sample_interval_sec must be positive"):
        ResourceMonitor(sample_interval_sec=0.0)


def test_fit_exposes_algorithm_resource_usage():
    model = ADP.create(
        "new",
        ADPConfig(
            n_centers=8,
            n_directions=3,
            min_neighbors=4.0,
            outer_steps=1,
            inner_steps=2,
            show_progress=False,
            random_state=12,
        ),
    )
    data = model.generate_data(n=32, d=3, noise=0.01)

    result = model.fit(
        data.X,
        data.y,
        centers=data.centers,
        directions=data.directions,
    )
    usage = result.resource_usage

    assert usage["algorithm_time_sec"] > 0.0
    assert usage["algorithm_memory_samples"] >= 2
    assert usage["algorithm_rss_min_mib"] <= usage["algorithm_rss_mean_mib"]
    assert usage["algorithm_rss_mean_mib"] <= usage["algorithm_rss_max_mib"]
    assert model.last_resource_usage_ == usage
    assert model.summary()["resource_usage"] == usage


def test_failed_fit_retains_last_algorithm_resource_usage():
    model = ADP.create(
        "new",
        ADPConfig(show_progress=False, random_state=13),
    )

    with pytest.raises(ValueError, match="X и y имеют разные размеры по n"):
        model.fit(np.ones((5, 2)), np.ones(4))

    assert model.last_resource_usage_["algorithm_time_sec"] > 0.0
    assert model.last_resource_usage_["algorithm_memory_samples"] >= 2
