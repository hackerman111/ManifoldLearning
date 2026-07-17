# Single-index ADP benchmark redesign

## Goal

Replace the executable C/S/T/R/M single-index benchmark catalog with the eight
independent experiment families specified in `new_benchmark.md`. The benchmark
must be reproducible, resumable, process-parallel across independent runs, and
strictly single-core inside each ADP fit. It must emit the requested normalized
CSV diagnostics and all requested PNG reports without retaining large working
matrices.

The standard full profile contains exactly 27,400 ADP runs. The top-level
`J/n` range in `new_benchmark.md` is not multiplied into that profile: the
standard matrix uses `J = n`. A center fraction remains available only as an
explicit CLI override.

## Chosen approach

Evolve the existing `adp.evaluation.single_index` package and the
`python run_benchmarks.py single-index` command in place. Reuse its deterministic
run IDs, process-pool orchestration, crash-safe worker shards, resume behavior,
progress reporting, resource monitoring, and CSV-based reporting. Replace the
old executable scenario registry and old report schema rather than introducing
a second benchmark package.

This avoids two competing single-index workflows and preserves the operational
parts that already solve process coordination and interrupted-run recovery. The
legacy `benchmark` and `stress` subcommands remain separate and unchanged.

## Non-goals

- Do not multiply `J/n in {0.25, 0.5, 1}` into the standard 27,400-run matrix.
- Do not preserve the old C/S/T/R/M scenario IDs as executable single-index
  profiles.
- Do not change the numerical update rules of ADP merely to populate benchmark
  diagnostics.
- Do not store full distance matrices, weight matrices, local `U` tensors,
  difference tensors, or every inner-iteration beta vector.
- Do not run the complete 27,400-run profile as part of automated tests.
- Do not alter the separate legacy `benchmark` or `stress` CLI contracts.

## CLI contract

The entrypoint remains:

```bash
python run_benchmarks.py single-index [OPTIONS]
```

The main full run is:

```bash
python run_benchmarks.py single-index \
  --profile full \
  --experiments all \
  --jobs auto \
  --output benchmark_outputs/single_index
```

The supported controls are:

- `--profile smoke|full`: `smoke` is the safe default and uses a small,
  representative subset; `full` uses every standard configuration and seeds
  0 through 99.
- `--experiments`: comma-separated selectors from `1`, `2`, `3`, `4`, `5`,
  `6`, `7.1`, `7.2`, `8.1`, `8.2`, and `8.3`, or `all`. The document's
  accidental `9.1`/`9.2`/`9.3` subsection labels are normalized to
  `8.1`/`8.2`/`8.3` because they belong to experiment 8.
- `--jobs auto|N`: `auto` is the default and resolves to the available logical
  CPU count; a positive integer sets an explicit process count.
- `--seeds 0:99`: an inclusive integer range or comma-separated seed list.
  Full-profile defaults are exactly 0 through 99.
- `--diagnostic-seeds 0,1,2`: seeds for detailed local and solver traces.
- `--center-fraction FLOAT`: explicit override for `J/n`; the default is 1.
  This override changes `J` but never expands the number of jobs.
- `--output PATH`: root for a newly created benchmark series directory.
- `--resume SERIES_DIR`: continue a compatible interrupted series.
- `--retry-failed`: replace failed commit markers and rerun those jobs.
- `--dry-run`: expand and validate jobs, print per-experiment counts and the
  total, and run no ADP fits.
- `--reports-only --resume SERIES_DIR`: rebuild summaries and figures only from
  the persisted CSV files.
- `--max-runs N`: deterministic post-expansion limit used for development and
  smoke verification. It is not part of a full scientific run.

The old `--statistics-workers` control is removed from this command. A
single-index benchmark fit always uses one statistics worker.

## Standard experiment matrix

Every configuration uses `n = ceil(d * n_over_d)`, dense `beta_true`, and 100
seeds unless a smoke profile or explicit seed selection is requested. Parameters
from different experiments are not cross-multiplied.

Common defaults are `sigma_x = 1`, `rho_corr = 0`, `sigma_eps = 0.5`,
`link = quadratic`, Gaussian features, Gaussian homoskedastic noise, `J = n`,
`delta = 0`, and `outlier_fraction = 0`.

| Selector | Experiment | Standard job count |
| --- | --- | ---: |
| `1` | correctness: 2 dimensions x 2 ratios x 2 links x 100 seeds | 800 |
| `2` | dimension and sample ratio: 4 dimensions x 5 ratios x 100 seeds | 2,000 |
| `3` | noise: 2 dimensions x 3 ratios x 7 noise levels x 100 seeds | 4,200 |
| `4` | AR(1) correlation: 2 dimensions x 3 ratios x 6 correlations x 100 seeds | 3,600 |
| `5` | feature scale: 2 dimensions x 3 ratios x 5 scales x 100 seeds | 3,000 |
| `6` | links: 2 dimensions x 3 ratios x 6 links x 100 seeds | 3,600 |
| `7.1` | feature distributions: 2 dimensions x 2 ratios x 3 distributions x 100 seeds | 1,200 |
| `7.2` | noise distributions: 2 dimensions x 2 ratios x 3 distributions x 100 seeds | 1,200 |
| `8.1` | heteroskedasticity: 2 dimensions x 2 ratios x 2 modes x 100 seeds | 800 |
| `8.2` | outliers: 2 dimensions x 2 ratios x 5 configurations x 100 seeds | 2,000 |
| `8.3` | misspecification: 2 dimensions x 2 ratios x 4 deltas x 100 seeds | 1,600 |
| | **Total** | **27,400** |

Experiment 1 overrides `sigma_eps` to zero. Every remaining fixed or varying
value follows `new_benchmark.md` literally.

## Deterministic job and seed model

A job is one experiment selector, one parameter configuration, and one user
seed. Its stable `run_id` includes the canonical configuration fingerprint and
seed. Changing process count, completion order, or resume order cannot change a
job's random values or identity.

Each user seed is deterministically split into independent substreams for:

- `beta_true`;
- feature generation;
- ordinary noise;
- center selection;
- ADP random directions;
- ADP initialization;
- outlier selection and outlier errors;
- the misspecification direction `gamma` and its signal.

The `run_summary.csv` commit-marker row stores all sub-seeds and all effective
parameter values. Resume validates the full series fingerprint before accepting
existing commit markers.

## Synthetic data generation

### Direction and sample size

Draw `v_j` independently from `N(0, 1)` and set
`beta_true = v / ||v||_2`. The direction is dense. Set
`n = ceil(d * n_over_d)` and `J = min(n, ceil(center_fraction * n))`.

### Feature distributions

For Gaussian features, construct the exact AR(1) covariance
`Sigma[i,j] = rho_corr ** abs(i-j)`, draw from `N(0, Sigma)`, and multiply by
`sigma_x`. The implementation uses a deterministic matrix factorization and
does not reuse the old shared-factor equicorrelation generator.

Uniform coordinates are drawn independently from `[-sqrt(3), sqrt(3)]`.
Student `t_5` coordinates are multiplied by `sqrt(3/5)`. Both therefore have
unit marginal variance before multiplication by `sigma_x`. Correlation varies
only where the experiment matrix requests Gaussian AR(1) features.

### Link and SNR normalization

The supported links are:

- `linear`: `z`;
- `quadratic`: `z + 0.5 z^2`;
- `square`: `z^2`;
- `sin`: `sin(1.5 z)`;
- `tanh`: `tanh(2 z)`;
- `oscillating`: `z sin(sqrt(5) z)`.

For each generated sample, the raw link vector is centered and divided by its
population standard deviation over that sample. The applied mean and scale are
recorded. This gives the generated signal variance one and makes
`SNR = infinity` when `sigma_eps = 0`, otherwise
`SNR = 1 / sigma_eps^2`. Degenerate or non-finite link variance is a generation
error rather than a silently modified response.

### Noise and contamination

Gaussian noise has unit variance before multiplication by `sigma_eps`.
Standardized `t_5` and `t_3` noise use factors `sqrt(3/5)` and `sqrt(1/3)`.

Heteroskedastic noise is

```text
sigma_eps * sqrt((0.25 + Z_i^2) / 1.25) * xi_i,
xi_i ~ N(0, 1), Z_i = beta_true^T X_i.
```

For response outliers, the ordinary error at selected observations is replaced
by an independent Gaussian error with standard deviation
`outlier_scale * sigma_eps`. The `(0, 1)` configuration selects no outliers.

For model misspecification, draw a Gaussian vector, project it orthogonally to
`beta_true`, normalize it to obtain `gamma`, and regenerate if the projected
norm is numerically zero. Define `g(z) = z + 0.5 z^2`, center and normalize its
sample values to variance one, and use

```text
Y = f(beta_true^T X) + delta * g(gamma^T X) + epsilon.
```

The `gamma` vector and normalization scalars are recorded by seed, but large
generated arrays are not persisted.

## Parallelism and one-core enforcement

The parent expands deterministic jobs and distributes them through
`ProcessPoolExecutor`. One future corresponds to one ADP fit and its atomic
worker-shard publication. The parent performs no ADP work; it advances progress,
merges committed shards, finalizes series metadata, and builds reports.

Single-core execution inside a worker is enforced at three layers:

1. Before process creation and in the worker initializer, set
   `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and
   `NUMEXPR_NUM_THREADS` to `1`.
2. Wrap the fit in `threadpoolctl.threadpool_limits(limits=1)` so already loaded
   BLAS runtimes are also restricted.
3. Construct `ADPConfig(statistics_workers=1, show_progress=False)` and expose no
   benchmark CLI override for nested statistics parallelism.

The job-level `tqdm` bar and flushed newline progress records remain usable in a
terminal and in redirected background logs. Progress advances only after a
job's final commit marker is atomically published.

## Telemetry architecture

The benchmark requires metrics that the current result object does not expose.
Add benchmark-neutral, optional ADP telemetry rather than recomputing the
algorithm from outside or changing its numerical steps.

Telemetry has five normalized levels:

- one run summary;
- one row per outer iteration;
- one row per alternating-solver inner iteration;
- one row per linear-solver iteration for selected diagnostic seeds;
- one row per local center and outer iteration for selected diagnostic seeds.

The ADP result exposes scalar outer/inner telemetry and transient vectors needed
to compute truth-based metrics in the benchmark executor. Detailed center and
linear-residual traces are enabled only for diagnostic seeds. Aggregate local
statistics are always computed because they are required in
`outer_iterations.csv`.

### Weight and neighborhood telemetry

While a center block's weights are already available, accumulate per center:

- `sum_w` and `sum_w2`;
- nonzero count;
- minimum and maximum weight.

Define `ESS = sum_w^2 / sum_w2`, with zero when `sum_w2` is zero. Only these
vectors survive the block; full weights do not. Outer telemetry stores mean,
minimum, median, 5th percentile, and 95th percentile as requested.

### Local-system telemetry

For center `j`, let `s = S[j,:]`, `u = U[j,:,:] @ beta`, and
`r = imav[j,:]`. The diagnostic normal system associated with local
coefficients `(c_j, a_j)` is

```text
A_j = [[s dot s, s dot u],
       [s dot u, u dot u]],
b_j = [s dot r, u dot r].
```

Record its determinant, symmetric eigenvalues, numerical rank, condition number,
and singular flag. Record the actual `c_j` and `a_j` returned by the configured
local solver, the actual local regularization value, and
`||r - c_j s - a_j u||_2`. Computing this diagnostic system does not cause ADP
to solve a different system.

A center is singular when its numerical rank is below two or its smallest
eigenvalue is not greater than
`2 * eps(dtype) * max(lambda_max, 1)`. A zero matrix has infinite condition
number. These rules are shared between detailed rows, outer aggregates, and
`singular_local_count`.

### Inner and linear-solver telemetry

For every alternating step, record objective before and after, relative
objective change, projective beta step, pre-normalization beta norm, beta truth
quality, gradient norm, absolute and relative linear residual, linear-solver
iteration count and status, mean changes in local intercepts and slopes, and
step runtime.

The built-in CG solver counts callback invocations and evaluates the final
residual for every run. Diagnostic seeds additionally retain the relative
residual after each CG callback so `solver_residual_vs_iteration.png` is backed
by real points rather than an interpolated curve. Custom solver stages may
report unsupported fields as missing, but the built-in benchmark configuration
must populate them.

### Timing telemetry

Measure non-overlapping spans for distance construction, kernel weights, local
statistics reduction, inner optimization, bandwidth/anisotropy update, and the
complete outer iteration. Define service overhead as the nonnegative remainder
between full iteration time and the named spans. Telemetry serialization time is
measured outside algorithm runtime.

## Persisted tables

Every series directory contains these required files with stable headers:

### `run_summary.csv`

One final commit-marker row per job. It contains `run_id`, experiment selector,
seed and all varied generation fields; `d`, `n`, `n_over_d`, `J`, and
`center_fraction`; initial and final bandwidth; final anisotropy; outer and
total inner iteration counts; absolute cosine and projector error; final loss;
algorithm runtime and peak process memory; singular-local and invalid-value
counts; stop reason; status; and error metadata. It also contains all effective
distribution, noise, outlier, misspecification, link-normalization, and sub-seed
fields needed to recreate the job.

### `outer_iterations.csv`

One row per outer iteration with the requested beta, norm, quality, projector
error, beta step and adjacent angle, objective values, relative objective
decrease, inner count, local mass quantiles, ESS aggregates, local-condition
aggregates, singular count, zero-weight fraction, and non-overlapping timings.
`beta_k` is encoded as a `|`-separated numeric vector using 17 significant
decimal digits for float64 and 9 for float32. Only outer estimates are stored;
inner beta vectors remain transient.

### `inner_iterations.csv`

One row per alternating-solver iteration with the requested objective, relative
change, beta step, pre-normalization norm, truth quality, gradient norm, linear
residuals, linear-solver count and status, mean local-coefficient changes, and
runtime.

### `local_diagnostics.csv`

One row per selected diagnostic center and outer iteration with local mass, ESS,
support, weight extrema, actual local coefficients, the local-system spectral
diagnostics, residual, regularization, and singular flag. Full rows are written
only for diagnostic seeds and for all telemetry collected before a failed run.

### Auxiliary CSV files

- `solver_iterations.csv` holds one relative-residual row per recorded CG step.
  This additional normalized table is necessary to support the required solver
  residual plot.
- `series.csv` holds environment, Git, profile, fingerprint, job counts, timing,
  and final series status.
- `artifacts.csv` records every CSV and PNG path, size, creation status, and
  plotting error.

No JSON manifest is required. Hidden `.shards/` directories contain atomic
per-run fragments and commit markers. Combined CSV files are derived from the
committed fragments.

## Status and failure semantics

Each job ends in exactly one public status:

- `success`: the result and required diagnostics are finite and every built-in
  linear solve completes normally;
- `nonconverged`: a finite direction is returned, but an alternating or linear
  solve reaches its configured iteration limit;
- `numerical_failure`: execution raises, the final direction has zero norm, or
  required numerical values contain NaN or infinity.

Reaching the configured outer schedule is not by itself a failure: outer steps
are an algorithm schedule, not a convergence solver. The more specific
`stop_reason` records tolerance, scheduled completion, linear iteration limit,
or numerical exception.

Partial inner, outer, local, and solver diagnostics are retained when available.
The run row is written last and is the only commit marker, so resume never treats
a partially persisted job as complete. Failed jobs are skipped unless
`--retry-failed` is explicit; retry atomically replaces their previous marker and
fragments.

## Reporting contract

Reports are built only from persisted CSV tables. `--reports-only` therefore
reproduces figures without rerunning ADP. Median curves and 5th/95th percentile
bands group seeds within one configuration; independent experiments are never
pooled. `success_rate` treats missing/numerically failed estimates as failures
and uses the document's `abs(cosine) >= 0.9` threshold. Experiment 1 additionally
reports the stricter `abs(cosine) >= 0.99` correctness criterion.

The report layer creates every applicable filename required by
`new_benchmark.md`, including:

- outer convergence: `quality_vs_outer_iteration.png`,
  `bandwidth_vs_outer_iteration.png`, `rho_vs_outer_iteration.png`,
  `beta_step_vs_outer_iteration.png`, and
  `objective_vs_outer_iteration.png`;
- inner optimization: `objective_vs_inner_iteration.png`,
  `beta_step_vs_inner_iteration.png`, and
  `solver_residual_vs_iteration.png`;
- local stability: `local_mass_by_outer_iteration.png`,
  `effective_neighbors_by_outer_iteration.png`,
  `local_condition_by_outer_iteration.png`, `mass_vs_condition.png`, and
  `local_slopes_by_outer_iteration.png`;
- dimension and ratio: `quality_heatmap_d_nd_ratio.png`,
  `success_rate_heatmap.png`, `runtime_vs_dimension.png`,
  `memory_vs_dimension.png`, and `iterations_heatmap_d_nd_ratio.png`;
- noise: `quality_vs_sigma_eps.png`, `success_rate_vs_sigma_eps.png`,
  `runtime_vs_sigma_eps.png`, `outer_iterations_vs_sigma_eps.png`, and
  `final_objective_vs_sigma_eps.png`;
- correlation: `quality_vs_correlation.png`,
  `success_rate_vs_correlation.png`, `local_condition_vs_correlation.png`,
  `solver_iterations_vs_correlation.png`, and
  `runtime_vs_correlation.png`;
- feature scale: `quality_vs_sigma_x.png`, `h0_vs_sigma_x.png`,
  `final_bandwidth_vs_sigma_x.png`, `local_mass_vs_sigma_x.png`, and
  `runtime_vs_sigma_x.png`;
- links: `quality_by_link_function.png`,
  `success_rate_by_link_function.png`,
  `outer_iterations_by_link_function.png`,
  `objective_by_link_function.png`, and
  `local_slopes_by_link_function.png`;
- distributions: `quality_by_x_distribution.png`,
  `quality_by_noise_distribution.png`, `failure_rate_by_distribution.png`, and
  `runtime_by_distribution.png`;
- contamination and misspecification:
  `quality_by_heteroscedasticity.png`,
  `quality_vs_outlier_fraction.png`, `failure_rate_vs_outliers.png`,
  `quality_vs_model_misspecification.png`, and
  `objective_vs_model_misspecification.png`;
- overall timing: `runtime_breakdown.png`.

Plots live below `plots/`, with experiment subdirectories where the same generic
diagnostic name would otherwise collide. Inner, solver, and detailed local plots
use diagnostic seeds. Summary plots use all committed statuses and apply their
documented failure treatment.

Failure to render one plot is recorded in `artifacts.csv` and does not delete
data or prevent other plots from being attempted. A complete full-profile
series is considered report-complete only when every applicable required plot
has status `created`.

## File boundaries

Keep modules focused around the existing package:

- `types.py`: immutable experiment, job, seed, and series configuration types;
- `scenarios.py`: exact profile grids and selector parsing;
- `datasets.py`: deterministic synthetic generation and normalization;
- `telemetry.py`: diagnostic dataclasses and scalar/local aggregation helpers;
- `executors.py`: one job's data generation, ADP configuration, fit, status, and
  normalized row construction;
- `schema.py`: stable table headers;
- `storage.py`: atomic shards, commit markers, resume, merge, and series state;
- `runner.py`: job expansion, process-pool dispatch, thread limits, progress,
  finalization, and dry-run;
- `reports.py`: CSV-only aggregation and plot orchestration;
- `plots.py`: individual required plot renderers, split out so `reports.py` does
  not remain a large mixed-purpose module;
- `adp` engine/backend files: only the minimal generic telemetry fields and
  measurement hooks required by the benchmark.

The current public helpers `build_single_index_jobs` and
`run_single_index_benchmark` remain available but adopt the new experiment
semantics. Old C/S/T/R/M IDs are not a compatibility promise.

## Test strategy

Implementation follows RED-GREEN-REFACTOR. Tests cover:

1. Exact standard job counts for every selector and total 27,400, plus proof
   that experiment parameters are not accidentally cross-multiplied.
2. `ceil(d * n_over_d)`, dense unit beta, deterministic sub-seeds, order- and
   process-count-independent job IDs, and resume fingerprint validation.
3. Exact AR(1) factor construction, standardized uniform and Student features,
   standardized links and noises, outlier replacement, orthogonal `gamma`, and
   deterministic misspecification.
4. ESS, local-system spectrum/rank/condition, singular classification, objective
   changes, residuals, and timing-remainder formulas on small fixed arrays.
5. Stable schemas, atomic commit-last behavior, partial failure recovery,
   retry replacement, and bounded-memory shard merging.
6. Serial and multi-process dispatch, `jobs=auto`, progress accounting, and
   one-core enforcement through environment variables, `threadpoolctl`, and
   `statistics_workers=1`.
7. Every required plot renderer from small fixture CSV frames, including
   missing-data and per-plot failure isolation.
8. Real end-to-end CLI smoke with two worker processes and required CSV/PNG
   artifacts.

The acceptance commands are:

```bash
python run_benchmarks.py single-index --profile full --dry-run
python run_benchmarks.py single-index \
  --profile smoke \
  --jobs 2 \
  --output /tmp/adp_new_benchmark_smoke
python -m pytest -q
git diff --check
```

The dry run must report exactly 27,400 jobs. The smoke run must execute real ADP
fits, use more than one worker when at least two CPUs are available, keep each
fit single-core, and produce the required smoke-applicable tables and figures.

## Documentation

Update `README.md` with the full, smoke, experiment-selection, resume,
reports-only, dry-run, and explicit process-count commands. Document that
`--jobs` controls independent process parallelism while every ADP fit is pinned
to one core, and document the exact required and auxiliary CSV files.
