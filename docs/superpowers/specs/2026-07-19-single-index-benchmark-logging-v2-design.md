# Single-index benchmark logging v2

## Goal

Make the single-index benchmark's persisted timing, process-memory, and ADP
configuration fields unambiguous and complete on both successful and failed
fits. The change applies only to `adp.evaluation.single_index`; the generic
experiment-log schema and the separate stress/legacy benchmarks remain
unchanged.

## Chosen approach

Introduce single-index schema version 2 and start new series with the corrected
headers. Existing version-1 directories remain untouched on disk, but v2 code
rejects them for `--resume` and `--reports-only` with the existing schema
mismatch error. No migration layer or deprecated aliases are added.

The executor owns wall-clock phase timing. `ResourceMonitor` remains the source
of algorithm elapsed time and current-process RSS because it brackets the ADP
algorithm itself and already produces the same data after an exception.

## Timing contract

`run_summary.csv` stores five distinctly scoped measurements:

- `data_generation_time_sec`: wall time spent in the attempted
  `generate_synthetic_data(job)` call, including when that call raises;
- `fit_wall_time_sec`: wall time of the actual `model.fit(...)` call, including
  the active one-thread `threadpool_limits` context but excluding data
  generation and normalized-row construction;
- `algorithm_time_sec`: elapsed time reported by the algorithm's
  `ResourceMonitor`; it is never replaced with job wall time;
- `telemetry_serialization_time_sec`: wall time spent converting the available
  result and telemetry into normalized benchmark rows;
- `job_wall_time_sec`: wall time from the start of `execute_job` through the
  completed normalized run row.

The resource monitor is the canonical source for `algorithm_time_sec` on both
success and numerical failure. If a phase never starts, its value is missing
rather than borrowed from a differently scoped timer. Small timing differences
between `fit_wall_time_sec` and `algorithm_time_sec` are expected because they
are measured at adjacent wrapper boundaries.

Outer-iteration component timings keep their existing non-overlapping contract:
their named components plus `service_overhead_sec` equal
`iteration_time_sec`, within floating-point tolerance.

## Memory contract

Replace the ambiguous `peak_memory_mb` column with the complete flattened
`ResourceUsage.to_dict("algorithm")` payload:

- `algorithm_rss_start_mib`;
- `algorithm_rss_min_mib`;
- `algorithm_rss_mean_mib`;
- `algorithm_rss_max_mib`;
- `algorithm_rss_peak_delta_mib`;
- `algorithm_memory_samples`;
- `algorithm_memory_source`.

These are sampled resident-set-size values for the current worker process, in
MiB. `algorithm_rss_max_mib` is the absolute sampled maximum;
`algorithm_rss_peak_delta_mib` is the nonnegative increase above the first
sample. Reports use the absolute maximum and label the unit `MiB`/`МиБ`.

When `model.fit` raises a handled numerical exception, `_fit_adp` transfers
`model.last_resource_usage_` to the executor even if no partial `ADPResult`
exists. Thus a failed fit retains its real algorithm time and RSS samples. A
failure before `model.fit` has missing algorithm-resource fields.

## Parameter contract

Keep the existing requested/effective generator parameters and independent
sub-seeds. Additionally store every field of the actual `ADPConfig` passed to
`ADP.create`, prefixed with `adp_`. The schema derives this stable prefixed field
list from the `ADPConfig` dataclass, and `_run_row` serializes the same concrete
config object used by the fit.

This deliberately retains useful benchmark-level fields such as `n_centers`
while adding explicit algorithm-level counterparts such as
`adp_n_centers`, `adp_outer_steps`, `adp_inner_steps`, `adp_kernel`,
`adp_dtype`, `adp_random_state`, `adp_record_telemetry`, and all remaining ADP
knobs. A failed data-generation job still records the intended ADP config.

## Schema and reporting

Define a single-index-local `SCHEMA_VERSION = 2` and use it in storage and
report validation. Do not change `adp.common.experiment_log.SCHEMA_VERSION`, so
unrelated experiment families do not become incompatible.

Remove `runtime_sec` and `peak_memory_mb` from the single-index schema. Runtime
plots read `algorithm_time_sec`; memory plots read
`algorithm_rss_max_mib`. Fixture CSVs, required-column checks, labels, and plot
specifications change together.

## Failure handling

Handled numerical failures continue to preserve partial outer, inner, local,
and solver telemetry when an `ADPResult` exists. Independently, resource usage
is preserved from the model. Status classification and exception metadata do
not change.

Unexpected exception classes still propagate instead of being converted to a
benchmark row.

## Test strategy

Implementation follows RED-GREEN-REFACTOR:

1. Add schema tests for version 2, explicit timing/RSS columns, removal of the
   ambiguous aliases, and the complete prefixed `ADPConfig` field set.
2. Add executor regression tests proving that success uses resource-monitor
   algorithm time, `fit_wall_time_sec` excludes generated-data latency, and a
   failed fit without `partial_result` retains algorithm time and RSS.
3. Update report fixtures and assertions to use algorithm time and absolute RSS
   maximum in MiB.
4. Run focused resource/schema/executor/storage/report/runner tests, the full
   test suite, `git diff --check`, and one real smoke benchmark.
