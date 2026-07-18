from adp.evaluation.single_index.schema import (
    ARTIFACT_COLUMNS,
    INNER_ITERATION_COLUMNS,
    LOCAL_DIAGNOSTIC_COLUMNS,
    OUTER_ITERATION_COLUMNS,
    RUN_SUMMARY_COLUMNS,
    SERIES_COLUMNS,
    SOLVER_ITERATION_COLUMNS,
)


PUBLIC_TABLES = {
    "run_summary": RUN_SUMMARY_COLUMNS,
    "outer_iterations": OUTER_ITERATION_COLUMNS,
    "inner_iterations": INNER_ITERATION_COLUMNS,
    "local_diagnostics": LOCAL_DIAGNOSTIC_COLUMNS,
    "solver_iterations": SOLVER_ITERATION_COLUMNS,
    "series": SERIES_COLUMNS,
    "artifacts": ARTIFACT_COLUMNS,
}


def test_every_public_schema_has_stable_unique_identity_columns():
    assert RUN_SUMMARY_COLUMNS[:3] == (
        "schema_version",
        "series_id",
        "run_id",
    )
    for name in (
        "outer_iterations",
        "inner_iterations",
        "local_diagnostics",
        "solver_iterations",
    ):
        assert PUBLIC_TABLES[name][:3] == (
            "schema_version",
            "series_id",
            "run_id",
        )
    for columns in PUBLIC_TABLES.values():
        assert len(columns) == len(set(columns))


def test_run_summary_contains_reproduction_status_and_quality_contract():
    expected = {
        "experiment",
        "seed",
        "d",
        "n",
        "n_over_d",
        "n_centers",
        "center_fraction",
        "sigma_x",
        "rho_corr",
        "effective_rho_corr",
        "sigma_eps",
        "snr",
        "link",
        "x_distribution",
        "noise_distribution",
        "effective_noise_distribution",
        "seed_beta",
        "seed_features",
        "seed_noise",
        "seed_centers",
        "seed_directions",
        "seed_init",
        "h_initial",
        "h_final",
        "rho_final",
        "cosine_abs",
        "projector_error",
        "objective",
        "runtime_sec",
        "peak_memory_mb",
        "singular_local_count",
        "invalid_value_count",
        "stop_reason",
        "status",
        "error_type",
        "error_message",
        "error_traceback",
    }

    assert expected <= set(RUN_SUMMARY_COLUMNS)


def test_detail_schemas_cover_requested_diagnostics():
    assert {
        "beta_k",
        "cosine_abs",
        "projector_error",
        "local_mass_q05",
        "local_mass_median",
        "local_mass_q95",
        "ess_mean",
        "condition_median",
        "service_overhead_sec",
    } <= set(OUTER_ITERATION_COLUMNS)
    assert {
        "objective_before",
        "objective_after",
        "linear_residual_norm",
        "relative_linear_residual",
        "linear_solver_iterations",
        "linear_solver_status",
    } <= set(INNER_ITERATION_COLUMNS)
    assert {
        "ess",
        "nonzero_weights",
        "condition",
        "rank",
        "regularization",
        "is_singular",
    } <= set(LOCAL_DIAGNOSTIC_COLUMNS)
    assert "relative_residual" in SOLVER_ITERATION_COLUMNS
