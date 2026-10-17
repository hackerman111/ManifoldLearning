# Adaptive-center full-fit benchmark protocol

Status: harness design frozen for implementation; benchmark execution awaits an
explicit start from the lead. This is a new report-inspired experiment, not a
reproduction of the unavailable raw runs cited by
`SVD/adaptive_centers_anisotropic_adp_edr_report.tex`.

## Question and scope

Measure whether the opt-in spectral floor, exact projected-neighbor search,
projected statistics, and LogDet/coverage center selection improve full-fit EDR
recovery or cost on paired synthetic multi-index problems. The benchmark uses
the public `ADP_multi_index` full-fit path and built-in default LSMR solver.
Baseline code and defaults remain the control. No result here establishes a
general recovery guarantee or global optimum.

## Cases and inputs

All cases use `n=500`, Gaussian features with AR(1) correlation `rho=0.5`,
`sigma_eps=0.2`, a two-column orthonormal truth basis, and scalar response.
Generate feature, truth, and noise streams independently from each run seed.
Use the same generated `X`, `Y`, truth basis, and model configuration for every
arm within a case/seed pair. Hash all three arrays and the effective config.

| Case | `d,m` | Link before noise |
|---|---:|---|
| additive30 | `30,2` | `z1^2 + sin(2.5 z2)` |
| multiplicative30 | `30,2` | `z1 sin(2.5 z2)` |
| aniso_curv40 | `40,2` | `z1^2 + sin(2.5 z2) + 0.3 z2^2` |
| additive50 | `50,2` | `z1^2 + sin(2.5 z2)` |

Set `z = X @ truth`. The data generator standardizes the noiseless link to unit
sample standard deviation before adding independent Gaussian noise. Keep the
same seed-to-stream mapping in development and heldout. Fresh development seeds
are `92800..92802`; heldout seeds are `92900..92904`. Do not use `816xx` or
`817xx` seeds.

## Common fit settings

Freeze `N_J=500`, `N_phi=10`, `N_loc=20`, `N_lin=2d`, `a=sqrt(2)`, `h_min=1`,
`index_init="local"`, `estimator="new"`, and `select_step="last"`; inherit all
other `ADP_Config` defaults. Record the full effective config. Run the full
schedule through `h_min`; do not shorten outer steps. Each fit has a 120-second
timeout. All arms use the default built-in LSMR adapter and CPU execution.

For center-selection arms with fixed budget use
`J_budget=max(4m,2d)` (60, 60, 80, 100 for the four cases respectively). Keep
the original candidate pool at `N_J=500`.

## Development arms

The API surface is `AdaptiveCentersConfig(lambda_floor=...,
projection_search=..., projected_statistics=..., selection=..., budget=...,
quadrature=..., coverage_weight=1., score_ridge=1e-6, logdet_ridge=1.)`.
`experimental_centers=None` denotes baseline. All unlisted experimental fields
use the API defaults frozen in the source manifest.

| Arm | Floor | Search | Projected statistics | Selection | Budget | Quadrature |
|---|---:|---|---|---|---|---|
| baseline | 0 | off | off | all | all | off |
| floor25 | 0.25 | off | off | all | all | off |
| floor50 | 0.50 | off | off | all | all | off |
| floor25_search | 0.25 | on | off | all | all | off |
| floor25_search_stats | 0.25 | on | on | all | all | off |
| random_equal_j | 0.25 | on | on | random | fixed | off |
| score_equal_j | 0.25 | on | on | score | fixed | off |
| logdet_equal_j | 0.25 | on | on | logdet | fixed | off |
| logdet_coverage_equal_j | 0.25 | on | on | logdet_coverage | fixed | off |
| logdet_coverage_quad_equal_j | 0.25 | on | on | logdet_coverage | fixed | on |

Run all ten arms on all four cases and three development seeds: 120 fits. No
adaptive-budget variant is included unless separately frozen by the lead before
execution. Random, score, LogDet, and coverage comparisons use the same fixed
`J_budget`, seed, and candidate pool. The coverage variant is explicitly an
experimental selection rule; interpret its empirical score only as a surrogate.

An optional integration smoke stage uses seed `92799`, one named case, and all
ten arms. It writes separate smoke outputs and is excluded from development and
heldout tables. It requires the same explicit start authorization as the full
development stage.

## Paired execution and measurements

Freeze a manifest before any numerical run: case/arm definitions, seeds,
effective configs, numerical and harness source SHA-256 hashes, Git commit and
dirty state, Python/NumPy/SciPy/BLAS versions, and threadpool information. Use
one fresh serial subprocess per fit, pin BLAS/OMP/MKL/NUMEXPR to one thread, and
verify actual threadpool limits inside the worker. Preserve every timeout,
exception, invalid result, and completed fit.

Record input hashes for `X`, `Y`, truth, initial fitted basis, and the effective
config; overall wall time; profiler stage times (`initialization`, `bandwidth`,
`directions`, `statistics`, `solver`, `update`); process peak RSS in KiB; EDR
error `sqrt(mean(sin(principal_angles)^2))`; final orthogonality; outer stop
reason/iterations; LSMR convergence, iteration, and certificate counts;
active/selected center counts; covered mass; selection fallback counts; and the
full nested `experimental_centers` trace diagnostic. `selected_pool_indices`
maps coefficient rows to selected candidate-pool rows; here `N_J=n`, so each
candidate-pool row is the same original `X` row. Baseline has no nested
experimental diagnostic and represents all 500 candidates. Keep any finer
selection, projected-moment, and full-operator durations if exposed by the
engine; the current profiler groups them under `statistics`.
Separate fit wall time from data/config generation and report both when useful.
Equal-J quality comparisons must include the fixed-budget selection cost in
end-to-end wall time.

## Selection and heldout gate

Retain all candidate arms and failures. The development audit verifies matching
`X`, `Y`, truth, and initial-basis hashes within each case/seed pair, then checks
the source and protocol hashes against the freeze manifest again after runs.
Pairwise analysis reports median EDR difference, median fit-time ratio, recovery
count (`EDR <= 0.1`), completion/failure rates, and the number of paired EDR
regressions above `0.05` per case and pooled descriptively over all 12
case/seed pairs.

Predeclared gates, evaluated on paired development data:

- A speed candidate must have median EDR difference from baseline `<= 0.01`,
  zero paired EDR regressions above `0.05`, no worse completion/failure rate,
  and median fit-time ratio `<= 0.90`.
- The `floor25` P0 quality candidate must have median paired EDR difference
  `<= -0.01`, median fit-time ratio `<= 1.10`, and its recovery count is
  reported beside baseline.

The lead reviews the paired evidence and freezes `results/selection.json`
before any heldout run. Heldout seeds remain untouched until that selection is
accepted. If no arm is selected, stop without heldout execution. The heldout
comparison arms/count must be frozen by the lead after development review; this
protocol does not authorize starting them automatically.

Do not infer a universal EDR gain from these four synthetic cases. Report
paired uncertainty and failures, distinguish completed fits from converged
inner solves, and retain cases where lower objective coincides with worse EDR.
