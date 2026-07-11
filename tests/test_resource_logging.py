import time

import pytest

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
