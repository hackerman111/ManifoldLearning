# ADP Local Solver Comparison Design

## Goal

Add an alternative ADP local-coefficient solver with the intercept fixed at zero and make the existing single-index benchmark run a paired comparison against the current 2x2 local regression in one invocation.

The comparison must preserve the current 2x2 behavior by default and must make solver identity explicit in persisted artifacts so results from the two algorithms cannot be mixed accidentally.

## User interface

The existing command remains unchanged and runs only the current solver:

```bash
python run_benchmarks.py single-index --profile smoke --experiments 2
```

The paired comparison is enabled explicitly:

```bash
python run_benchmarks.py single-index \
  --profile full \
  --experiments 2,5 \
  --local-solvers zero_intercept,least_squares \
  --output benchmark_outputs/local_solver_comparison
```

Accepted local solver names are:

- `least_squares`: the current two-parameter local regression with a free intercept;
- `zero_intercept`: the ADP local regression constrained to `c_j = 0`.

Unknown names and an empty solver selection are rejected before any output directory is created. Duplicate names are removed while preserving the user's first-occurrence order.

## Solver mathematics

The existing `least_squares` implementation is unchanged. For every center it solves the ridge-regularized two-column local regression with design columns `S_j` and `U_j beta`, returning the two coefficients as intercept and slope.

The new `zero_intercept` solver returns an all-zero intercept vector and computes one slope per center:

```text
projected_j = U_j beta
slope_j = <imav_j, projected_j> / <projected_j, projected_j>
```

The denominator is floored using the active array dtype's smallest positive finite scale so zero or underflowing projected directions do not create NaN or infinity. The returned arrays must preserve the numerical dtype used by the local statistics.

The implementation belongs to the random-projection ADP variant because it depends on `imav` and `U`. A small built-in stage adapter exposes it through the existing `local_solver` registry. The current `least_squares` stage remains the default registry entry.

## Paired benchmark model

`local_solver` is an algorithm axis, not a data-generation parameter. It is therefore stored directly on `SingleIndexJob`, while `SingleIndexSeriesConfig.local_solvers` controls expansion and defaults to `("least_squares",)`.

Job expansion order is deterministic: experiment order, scenario-grid order, configured local-solver order, then seed order. Every selected scenario and seed produces one job per selected solver.

The run identity includes `local_solver`, so the paired jobs have distinct `run_id` values. The seed fingerprint deliberately excludes `local_solver`. Consequently both jobs in a pair receive identical seeds for beta, features, noise, centers, directions, initialization, outliers, and misspecification. Each worker still regenerates its own arrays, preserving independent timing and memory measurements while producing identical values.

`execute_job` constructs the model with both selected stages:

```text
statistics_builder = job.parameters.statistics_builder
local_solver = job.local_solver
```

All remaining ADP configuration and stages are identical between the pair.

## Persistence and resume safety

The public CSV schema is raised from version 4 to version 5.

`local_solver` is persisted in:

- `run_summary.csv`;
- `outer_iterations.csv`;
- `inner_iterations.csv`;
- `local_diagnostics.csv`;
- `solver_iterations.csv`.

`series.csv` stores the selected solver list in `local_solvers`. The series configuration fingerprint includes this list. Resume therefore rejects a command whose solver selection differs from the original series, and schema-version validation prevents a version-4 series from being resumed as version 5.

The benchmark progress line includes the local solver name so background logs make pair progress visible.

## Efficiency measurements and reports

Existing per-run measurements remain authoritative:

- `fit_wall_time_sec` for end-to-end fit time;
- `algorithm_time_sec` and algorithm RSS fields for core algorithm cost;
- `stage_local_solver_time_sec` and `stage_local_solver_calls` in outer-iteration telemetry;
- `cosine_abs`, `projector_error`, status, and stop reason for statistical and convergence behavior.

The run summary additionally stores total `local_solver_time_sec` and `local_solver_calls` from `result.stage_timings` and `result.stage_calls`, avoiding manual aggregation of the outer table.

When a series contains one solver, report paths and behavior remain unchanged. When it contains multiple solvers, standard plots are rendered separately under `by_local_solver/<solver>/` so incompatible algorithms are never silently pooled. A machine-readable `local_solver_comparison.csv` is produced from `run_summary.csv`, grouped by experiment, complete scenario parameters, and solver. It contains run count, convergence rate, mean and median absolute cosine, mean and median fit time, mean algorithm time, mean local-solver time, and mean peak RSS delta. It is registered in `artifacts.csv`.

No claim of a paired speedup is computed by subtracting wall times from concurrently executed jobs. The saved long-form rows share the same experiment, scenario-parameter, and seed keys, allowing downstream paired analysis without pretending that process scheduling noise is algorithmic work.

## Failure behavior

A failure in one solver produces the normal numerical-failure row for that solver and does not erase or relabel its paired run. The other solver remains independently executable and committable.

Invalid or non-finite local coefficients continue through the existing stage-output validation and are reported by the existing numerical-failure classification. No fallback from one local solver to the other is allowed because that would invalidate the comparison.

## Testing

Implementation follows test-driven development and covers:

1. the `zero_intercept` solver returns exact zero intercepts and the expected scalar-regression slopes, including a zero projected-direction case;
2. the built-in registry resolves both solver names while preserving `least_squares` as the default;
3. CLI parsing preserves the old default and accepts the paired flag;
4. job expansion doubles paired runs, gives each solver a distinct run ID, and gives both members identical seed bundles;
5. executor model construction selects the requested local solver and logs it in every output row;
6. schema-version, series metadata, fingerprint, resume, and storage tests include the new fields;
7. report tests prove that multi-solver standard plots are separated and that `local_solver_comparison.csv` does not pool solver identities;
8. a smoke benchmark runs both solvers for the same experiment and seed and persists two complete rows with finite timing and quality metrics.

## Non-goals

- Changing the default ADP algorithm from the current 2x2 solver.
- Changing the convergence tolerance or stop rule.
- Reinterpreting old version-4 benchmark artifacts.
- Sharing one in-memory generated dataset between the two fits.
- Optimizing either solver beyond what is required for a faithful comparison.
