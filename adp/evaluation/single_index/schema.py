from __future__ import annotations


IDENTITY_COLUMNS = ("schema_version", "series_id")
RESOURCE_SUFFIXES = (
    "time_sec",
    "rss_start_mib",
    "rss_min_mib",
    "rss_mean_mib",
    "rss_max_mib",
    "rss_peak_delta_mib",
    "memory_samples",
    "memory_source",
)


def _resource_columns(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}_{suffix}" for suffix in RESOURCE_SUFFIXES)


SERIES_COLUMNS = IDENTITY_COLUMNS + (
    "config_fingerprint",
    "status",
    "profile",
    "started_at_utc",
    "finished_at_utc",
    "git_commit",
    "git_branch",
    "git_dirty",
    "python_version",
    "numpy_version",
    "platform",
    "requested_jobs",
    "completed_jobs",
    "failed_jobs",
    "unavailable_jobs",
    "process_jobs",
    "statistics_workers",
    "base_seed",
)

RUN_COLUMNS = IDENTITY_COLUMNS + (
    "run_id",
    "scenario_id",
    "family",
    "executor",
    "method",
    "repeat",
    "data_seed",
    "beta_seed",
    "centers_seed",
    "directions_seed",
    "init_seed",
    "status",
    "failed",
    "error",
    "stage",
    "stop_reason",
    "iteration_rows",
    "solver_iteration_rows",
    "result_persist_time_sec",
    "cosine",
    "cosine_abs",
    "angle_deg",
    "signed_l2",
    "objective",
    "dataset_source",
    "dataset_path",
    "dataset_size_bytes",
    "dataset_sha256",
    "dataset_rows",
    "dataset_features",
) + _resource_columns("algorithm") + _resource_columns("full_run")

ITERATION_COLUMNS = IDENTITY_COLUMNS + (
    "run_id",
    "scenario_id",
    "method",
    "outer_k",
    "h_k",
    "rho_k",
    "local_mass_mean",
    "local_mass_q05",
    "local_mass_min",
    "objective",
    "cosine_abs",
    "beta_delta",
    "statistics_time_sec",
    "solve_time_sec",
    "runtime_sec",
)

SOLVER_ITERATION_COLUMNS = IDENTITY_COLUMNS + (
    "run_id",
    "scenario_id",
    "method",
    "outer_k",
    "inner_k",
    "cg_k",
    "relative_objective",
    "relative_residual",
    "projective_delta",
    "cg_info",
)

INITIAL_PARAMETER_COLUMNS = IDENTITY_COLUMNS + (
    "run_id",
    "scenario_id",
    "family",
    "executor",
    "method",
    "repeat",
    "data_seed",
    "beta_seed",
    "centers_seed",
    "directions_seed",
    "init_seed",
    "hypothesis",
    "data_dataset",
    "data_n",
    "data_d",
    "data_link",
    "data_noise",
    "data_corr",
    "data_sigma_x",
    "algorithm_n_centers",
    "algorithm_n_directions",
    "algorithm_min_neighbors",
    "algorithm_statistics_workers",
    "solver_outer_steps",
    "solver_inner_steps",
)

FAILURE_COLUMNS = IDENTITY_COLUMNS + (
    "run_id",
    "scenario_id",
    "method",
    "status",
    "category",
    "exception_type",
    "error",
    "stage",
    "last_outer_k",
    "last_inner_k",
)

ARTIFACT_COLUMNS = IDENTITY_COLUMNS + (
    "artifact_type",
    "name",
    "path",
    "size_bytes",
    "status",
    "error",
)
