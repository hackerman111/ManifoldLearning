# ADP versus R SDR comparison

User guide and commands: [experiments/r_sdr/README.md](../../experiments/r_sdr/README.md).
Entry point: `python -m experiments.r_sdr`. This is an isolated experimental
runner; production ADP and historical `benchmarks/adp_mave_*` remain unchanged.

## Source routes

- `experiments/r_sdr/__main__.py`: CLI, >=10 repetitions, filters and analysis.
- `design.py`: case catalogs, independent deterministic B/X/noise streams,
  original-coordinate subspace metrics without dense projectors.
- `methods.R`, `worker.R`: classical SIR, ordinary ambient-kernel MAVE,
  projected-kernel refined MAVE, joint-response density MAVE and package MEANMAVE.
- `worker.py`: public ADP fit with opt-in `ADP-Grassmann` dispatch, fit timer
  and inner diagnostics. ADP outer convergence is unknown; inner stationarity
  is a separate field.
- `protocol.py`: environment, immutable source/protocol fingerprint and snapshots.
- `runner.py`: paired data, isolated workers, process/RSS limits, raw outcomes,
  resumable journal. Workers execute the saved source tree.
- `report.py`: per-case statistics, failure denominators, same-seed comparisons,
  bootstrap intervals, CSV/Markdown and optional plots.

## Scientific boundaries

All methods receive the same X/Y and known target rank for each case/seed.
Profiles: smoke 2, standard 22, full 50, stress 6, and `multi-rank` 3 cases;
default 10 repetitions. The focused profile uses additive Gaussian
`n=500,d=20,m=3,4,5` cases and leaves standard/full catalogs unchanged.
Full gives 3000 fits across six methods; stress gives 360. The full catalog
extends to n=10000/d=1000; stress reaches n=20000/d=2000, including n<=d.
Classical SIR explicitly fails on singular covariance; no implicit ridge/PCA.

Ordinary MAVE uses weights in the full predictor space, refined MAVE uses
projected weights. dMAVE is an explicit untrimmed reference variant with all
observed response targets and a fixed response bandwidth, not a reproduction
of every paper default. Joint least squares uses streaming QR and exact
response-space compression, not averaged independently fitted subspaces.
Package MEANMAVE is a separate compiled implementation. R-MAVE means refined
MAVE here. See the guide for initialization, bandwidths and iteration limits.

Fit and whole-process wall times are separate. RSS is sampled from Linux /proc
and includes imports/runtime; the memory cap is polled, not a hard OS limit.
Failures remain in recovery denominators; paired time/quality comparisons use
jointly valid same-seed fits and state their available counts. Resume retains
recorded failures and rejects changed source/config/environment.

## Evidence

`tests/test_r_sdr.py` and `tests/r_sdr_reference.R`: dense reference checks for
subspace metrics, SIR, streaming/joint QR, response compression, ordinary versus
refined weights; real timeout/memory limits, R library discovery, interruption
and resume, manifest drift, missing/failure denominators and convergence labels.
`benchmark_outputs/r_sdr/validation-final/`: schema2 integration smoke, 120/120
valid fits (2 cases x 10 seeds x 6 methods); no large-catalog performance claim.
Earlier `validation-smoke` is schema1 and must not be pooled with schema2.
The three-method rank comparison is in
`benchmark_outputs/r_sdr/multi-rank-three-methods-2026-10-17/` (90/90 fits,
10 paired seeds per rank, zero threshold recoveries). Its separate ADP stage and
LSMR profile is in `diagnostics/adp-bottleneck.md`; it identifies solver/LSMR,
not statistics, as the dominant cost for these ranks. The interrupted
two-method preview is a separate incomplete folder and must not be pooled.

`ADP-Grassmann` is an explicit experiment method, not the default. With
`local_ridge=0`, `grassman.solve` minimizes the same unpenalized finite-sketch
profile and returns the `HPAOResult` contract; its documented warm-start budget
is five steps per scale. The isolated comparison against ordinary MAVE and
package MEANMAVE is in
`benchmark_outputs/r_sdr/multi-rank-grassman-2026-10-17/` (90/90 successful,
10 paired seeds per rank, zero threshold recoveries), with Grassmann as the
paired-report baseline. Its diagnostic
`diagnostics/grassman-inner-summary.md` records that all Grassmann calls reached
the five-step cap, 149/150 calls used rank-one fallback for local rank loss,
and all outer schedules stopped at `h_min`; no Grassmann inner convergence or
outer convergence is established.

Accuracy research is routed through PLAN.md, agent-notes/STATE.md and
experiments/grassman_accuracy_2026_10_18/REPORT.md. Explicit opt-in methods
include ADP-Grassmann-conjugate and ADP-Grassmann-relative; neither is in
design.METHODS. worker.solver_for lazy-loads prototypes; relative now calls public
grassman.solve with relative_gn=True. Runner records their
JSON metadata, implementation/certificate semantics and frozen solver config.
Relative GN uses a10-step budget/smooth cap5/lambda_prox=.05, eligible intrinsic
model only; RELATIVE_GN_PROTOCOL.md fixes independent105800/106800 stages.
ADP-Grassmann now explicitly pins the original5 control (relativeFalse,
quadratic0,constantFalse), even though public solve defaults changed at the
user's request. Relative pins10/smooth5,relativeTrue,quadratic0,constantFalse.
The new public smooth_max_steps option is valid; lambda_prox still must not
be duplicated because it comes from ADP_Config.
Original frozen dev/heldout sources call the prototype; current public port
reproduces all120 saved full fits (bases and every inner diagnostic) exactly.
Runner snapshots source/config/environment; a resumed run must match them.

Combined integration: COMBINED_{MATH,PROTOCOL,REPORT}.md. New explicit methods
ADP-Grassmann-combined10 (relative/quadratic3/zeros,10/smooth5) and
ADP-Grassmann-combined (same components20/no cap, current public default).
Development107800 and validation108800 complete150+150 fits, identical
sources/settings,30 matched blocks/stage and no errors. Validation final/original
EDRgeo.925311,time.860129; final/relative10 EDR1.003418,time1.275697. Do not
claim demonstrated extra recovery from the components/budget. Public-default
and explicit control replay120/120 exactly matches bases/all diagnostics/stops.
