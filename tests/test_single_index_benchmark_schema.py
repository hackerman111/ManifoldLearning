import pytest

from adp.evaluation.single_index.schema import (
    ARTIFACT_COLUMNS,
    FAILURE_COLUMNS,
    INITIAL_PARAMETER_COLUMNS,
    ITERATION_COLUMNS,
    RUN_COLUMNS,
    SERIES_COLUMNS,
    SOLVER_ITERATION_COLUMNS,
)


@pytest.mark.parametrize(
    "columns",
    [
        SERIES_COLUMNS,
        RUN_COLUMNS,
        ITERATION_COLUMNS,
        SOLVER_ITERATION_COLUMNS,
        INITIAL_PARAMETER_COLUMNS,
        FAILURE_COLUMNS,
        ARTIFACT_COLUMNS,
    ],
)
def test_single_index_csv_schemas_have_stable_identity_columns(columns):
    assert columns[:2] == ("schema_version", "series_id")
    assert len(columns) == len(set(columns))


def test_run_schema_contains_quality_status_and_resource_contract():
    expected = {
        "run_id",
        "scenario_id",
        "method",
        "status",
        "failed",
        "error",
        "stage",
        "cosine_abs",
        "angle_deg",
        "result_persist_time_sec",
        "algorithm_time_sec",
        "algorithm_rss_min_mib",
        "algorithm_rss_mean_mib",
        "algorithm_rss_max_mib",
        "algorithm_rss_peak_delta_mib",
        "full_run_time_sec",
        "full_run_rss_min_mib",
        "full_run_rss_mean_mib",
        "full_run_rss_max_mib",
        "full_run_rss_peak_delta_mib",
        "dataset_source",
        "dataset_path",
        "dataset_size_bytes",
        "dataset_sha256",
        "dataset_rows",
        "dataset_features",
    }

    assert expected.issubset(RUN_COLUMNS)
    assert "data_dataset" in INITIAL_PARAMETER_COLUMNS


def test_run_dependent_schemas_include_run_id():
    for columns in (
        RUN_COLUMNS,
        ITERATION_COLUMNS,
        SOLVER_ITERATION_COLUMNS,
        INITIAL_PARAMETER_COLUMNS,
        FAILURE_COLUMNS,
    ):
        assert "run_id" in columns
