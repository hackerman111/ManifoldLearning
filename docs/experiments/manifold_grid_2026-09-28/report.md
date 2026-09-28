# Manifold grid: boundary repair and bounded learning check

Date: 2026-09-28. The full 5,500-fit profile was not run. The evidence below
covers all 22 unique grid cells, with five development seeds on 15 cells and
one seed on each remaining high-dimensional cell. One `(800,10,2,.35,.1)`
case was run in two series and counted once in the combined figures.

## Protocol change

The two interrupted strict outputs contain 61/61 instances of
`function mass target is infeasible even with alpha=0` at the first
`(n,d,m,c,sigma)=(200,3,1,.35,.1)` cell. The model had already made four
projector updates in that seed. At the next smaller bandwidth, target mass
80 is impossible even with maximal anisotropy; this is a feasibility
boundary, not a linear-solver failure.

The grid now defaults to the existing explicit `scale_boundary=stop` variant.
It finds the last feasible bandwidth, makes that update, and records the
boundary reason. This is an **ESTIMATOR/protocol choice** relative to
`Manifold exp.md`, which fixes `raise`. The original strict behavior remains
available with `--scale-boundary raise`; previous outputs were not rewritten.
The production `ADP_Manifold` default, solver, masses, ranks and other fixed
grid parameters are unchanged. The already documented `N_lin` feasibility
adaptation remains in place.

Paired seed 81000 on the first cell: `raise` fails after four updates;
`stop` returns after five updates and two scales with mean function mass
80.0000000002 and last-step linear residual `9.1e-15`. Its center
RMS/max principal sine is `0.330/0.642`, so this valid fit does **not**
meet the recovery criterion.

Result schema 2 separates center error, same-center chart-estimation error,
raw nearest-chart query error, and oracle nearest-chart discretization error.
`recovered` uses center and chart-estimation RMS/max `<=0.2` only where the
target is assessable: `m=1` or flat `c=0`. For curved `m>=2`, the generator
chart `row(Dz)` is not identified by one scalar response, so
`recovered=null` while distances remain descriptive. The mathematical
reason and metric distinction are in the
[generalization repair note](../manifold_generalization_2026-09-28/repair.md).

## Reproduction and provenance

All runs used seed start 81000, one BLAS thread, `float64`, commit
`087ee88338f4870db0afcf649fc833fb5ad64139` with a dirty worktree,
Python 3.13.12, NumPy 2.5.2, SciPy 1.18.1 and OpenBLAS 0.3.34.
Each manifest records the data formula, exact model configuration, source
hashes and environment; each JSONL file preserves every failed fit and
trace. The three complete artifacts are:

| Scope | Command suffix | Runs | Artifact |
| --- | --- | ---: | --- |
| Six non-HD series | `--runs 5 --series dimension,sample-size,noise,intrinsic-dimension,curvature,noiseless-sanity` | 75 | [five-seed result](../../../benchmark_outputs/experiments/20260928-manifold-grid-repair-5seed/summary.json) |
| HD fixed n | `--runs 1 --series hd-fixed-n` | 5 | [HD fixed-n result](../../../benchmark_outputs/experiments/20260928-manifold-grid-repair-hd-fixed-n-1seed/summary.json) |
| HD fixed n/d | `--runs 1 --series hd-fixed-n-over-d` | 3 | [HD ratio result](../../../benchmark_outputs/experiments/20260928-manifold-grid-repair-hd-ratio-1seed/summary.json) |

Run from the repository root with
`UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.manifold_grid`,
followed by the listed suffix and a fresh `--output-dir` if a fixed path is
wanted. Omitting `--runs` restores 250 seeds per cell in `--profile full`.

## Measured result

| Sample | Unique attempts | Fit completed | Rank failures | Assessable attempts | Recovered |
| --- | ---: | ---: | ---: | ---: | ---: |
| Five-seed non-HD | 75 | 57 | 18 | 40 | 2 |
| All three artifacts, duplicate removed | 82 | 61 | 21 | 40 | 2 |

All 61 completed fits have `stop_reason=function_mass_boundary`; none
reached `h_min`. The maximum recorded linear relative residual among them is
`9.999e-7`, within the configured `cg_tol=1e-6`. The 21 errors comprise
11 rank-deficient local slopes, nine insufficient local EDR-rank cases,
and one rank-deficient local-linear pilot. They remain in the denominator;
the repaired boundary policy does not turn them into successful fits.

On the 35 curved `m=1` attempts, 32 fits completed and **0 recovered**.
The flat `(400,10,2,0,.1)` control recovered **2/5** attempts: three fits
completed and two failed rank checks. It shows learning can recover a
simple identifiable subspace on some seeds, but it is not reliable under
this fixed configuration. For the curved `m>=2` cells, fit completion and
descriptive generator-chart distances are available, but a recovery rate
would not have the claimed statistical meaning.

The high-dimensional cells were sampled on only one seed. At fixed `n=800`,
`d=50` failed local-linear rank (`39 < 50`) and `d=200` failed local-slope
rank (`1 < 2`); `d=10,20,100` completed. At fixed `n/d=80`, the `d=20`
case failed local-slope rank, while `d=50,100` completed. These outcomes
are diagnostic points, not estimated failure rates.

The non-HD run summed to 27.47 s over 75 fits, with median completed-fit
time 0.485 s. The largest one-seed high-dimensional fit took 15.24 s.
Maximum process `ru_maxrss` across the three processes was 242 MiB; this is
a cumulative process high-water mark, not per-fit peak memory.

**Conclusion:** the repaired grid executes past the former mass exception,
and the estimator can sometimes learn the flat target. Under the specified
local-support settings it did not reliably recover the identifiable target
in this bounded sample. Boundary completion establishes a usable fit, not
outer convergence or geometric recovery. A full 250-seed run and held-out
validation would be needed for population-level rates or parameter choices.
