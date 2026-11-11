# Adaptive-center development benchmark

## Result

The frozen 120-fit development run completed with 120/120 successful fits. The
pairing audit passed all 12 case/seed blocks: all ten arms shared identical
`X`, `Y`, truth, and initialization hashes, and selected coefficient mappings
were valid. The post-run source, protocol, and source-snapshot hashes match the
freeze manifest. The separate ten-arm smoke run also passed its harness audit
and is excluded from these results.

The runs reached their configured schedule endpoints (six or seven outer
iterations, depending on case), but **0/120 fits had all inner solver
certificates converged**. Completion at `h_min` must not be described as
convergence or successful recovery. Baseline recovery at EDR `<=0.1` was 1/12;
floor25 and floor25_search were also 1/12. The center-selection arms each had
0/12 recoveries.

## Paired development results

Differences are candidate minus control EDR; ratios are candidate/control fit
wall time. Pooled results are descriptive over 12 case/seed pairs (three seed
blocks, with related cases sharing generated inputs); they are not twelve
independent observations. The frozen gate definitions are in `PROTOCOL.md`.

| Candidate vs control | Median EDR difference | Median fit-time ratio | Pairs with EDR regression >0.05 | Recovery counts | Gate |
|---|---:|---:|---:|---:|---|
| floor25 vs baseline | 0.0000 | 0.994 | 1 | 1/12 vs 1/12 | P0 quality: fail |
| floor50 vs baseline | +0.0015 | 0.989 | 0 | 2/12 vs 1/12 | no predeclared gate |
| floor25_search vs baseline | ~0.0000 | 1.635 | 1 | 1/12 vs 1/12 | speed: fail |
| floor25_search vs floor25 | ~0.0000 | 1.682 | 0 | 1/12 vs 1/12 | slower; no speed gain |
| floor25_search_stats vs floor25_search | 0.0000 | 1.179 | 0 | 1/12 vs 1/12 | slower |
| random equal-J vs baseline | +0.0839 | 1.842 | 9 | 0/12 vs 1/12 | fail |
| score equal-J vs baseline | +0.0587 | 2.187 | 7 | 0/12 vs 1/12 | fail |
| LogDet equal-J vs baseline | +0.0464 | 2.481 | 6 | 0/12 vs 1/12 | fail |
| LogDet + coverage equal-J vs baseline | +0.0397 | 2.487 | 5 | 0/12 vs 1/12 | fail |
| LogDet + coverage + quadrature equal-J vs baseline | +0.0242 | 2.814 | 4 | 0/12 vs 1/12 | fail |
| LogDet + coverage equal-J vs random equal-J | -0.0377 | 1.338 | 2 | 0/12 vs 0/12 | better surrogate quality, slower |

P0 floor25 misses its quality gate: median EDR difference is not at most
`-0.01`, and one pair regresses by more than `0.05`. Its additive30 case
improves strongly, but that does not carry to the other cases. Floor50 improves
the additive30 median and has one additional recovered seed overall, while its
pooled EDR change is slightly worse; this was a development-only arm and was
not the predeclared P0 candidate.

The case-specific P0 medians make the variation clear:

| Case | Baseline median EDR | floor25 median EDR | Paired median difference | floor25/baseline fit-time ratio |
|---|---:|---:|---:|---:|
| additive30 | 0.4190 | 0.2486 | -0.1274 | 0.980 |
| multiplicative30 | 0.7668 | 0.7893 | +0.0388 | 0.994 |
| aniso_curv40 | 0.6014 | 0.6014 | -0.0002 | 0.993 |
| additive50 | 0.6188 | 0.6232 | +0.0033 | 1.006 |

Median peak RSS across the four case summaries was 83.0 MiB for baseline,
83.3 MiB for floor25 (1.004x baseline), and 98.8 MiB for floor25_search
(1.187x baseline). The fixed-budget
equal-J controls used J=60/80/100 for these cases; their median peak RSS ranged
from 98.7 MiB (random) to 99.3 MiB (LogDet), compared with 83.0 MiB for
baseline. LogDet plus coverage was 99.0 MiB (1.191x baseline). Thus reducing
active centers did not reduce process peak RSS in this implementation/workload.
Per-case RSS and center counts are in the summary CSV.

P1 exact projected-neighbor search does not improve full-fit wall time. It is
about 1.68x slower than floor25 overall, despite similar final EDR on most
pairs. Against floor25, one pair improves by about `0.0104` EDR; the small
differences show that full-fit equivalence is not exact at the stopping
criterion. P2 projected statistics adds about 18% fit wall time on top of P1,
with no observed median EDR change. These full-fit runs use only `n=500`; the
separate localization benchmark is intended to test the large-`n` cost crossover.

All four center-selection schemes reduce the active center count to the fixed
budget, but each worsens pooled EDR and increases end-to-end fit time relative
to baseline. Quadrature reduces the median quality gap relative to LogDet plus
coverage, but remains slower and does not pass the gate. These surrogate
selection outcomes are evidence for these cases only, not a recovery guarantee.

## Reproducibility and artifacts

Development seeds are `92800..92802`; held-out seeds `92900..92904` were not
run. Inputs use separate data, truth, and noise random streams. Workers ran
serially in fresh processes with one verified BLAS/OMP thread. The frozen fit
settings and all configurations are in `results/freeze_manifest.json`;
`results/source_snapshot.zip` preserves the numerical and harness sources.

- `results/fullfit_dev.jsonl`: 120 complete worker records and diagnostics.
- `results/fullfit_dev_summary.csv`: per-case/per-arm EDR, time, RSS,
  convergence, center-count, coverage, and profiler summaries.
- `results/fullfit_dev_pairs.csv`: per-case and pooled paired comparisons and
  predeclared gate values.
- `results/pairing_audit.json`: shared-input, coefficient-mapping, and frozen
  source checks.
- `results/verification_dev.json`: final freeze verification.
- `results/smoke_additive30_92799.jsonl` and companion audit files: excluded
  harness smoke.

The paired summaries use medians and report counts; no IID confidence intervals
are claimed. The report-inspired generators are new synthetic cases and do not
reproduce unavailable raw data from the referenced TeX report.

An additional deterministic trajectory audit replayed floor25 and
floor25_search for multiplicative30 seed 92802 and reproduced the raw initial
and final hashes. At all 12 identical states, support was identical and alpha
was bitwise equal; weights, `I`, `U`, mass, `S`, mean, and effective sample
size agreed to float64 tolerances. A diagnostic `eta` ratio was unstable on a
singleton support (1 versus 0) because of a tiny residual and zero denominator;
the solver does not consume that diagnostic. The first post-solver basis
already differed by about 0.001135, followed by amplification across later
steps. This establishes numerical sensitivity in that trajectory but does not
identify a solver root cause. See
[`verification/trajectory_audit.json`](verification/trajectory_audit.json)
for the bounded replay artifact.

The full-fit numerical source hashes were verified against the original frozen
manifest immediately after the run. A later compatibility-only `seed=None`
entropy patch was made after all benchmark timings; it is not part of the
measured source snapshot. Its explicit overlay and current hashes are tracked
in [`verification/compatibility_overlay.json`](verification/compatibility_overlay.json).

The separate P1 localization measurements and their exactness limits are in
[`supplement/REPORT.md`](supplement/REPORT.md); they measure geometry/weight
cost and do not establish full-fit recovery or end-to-end speed at other sizes.
The standalone per-case EDR/time plot is available as
[`development_edr_time.svg`](results/development_edr_time.svg) and
[`development_edr_time.png`](results/development_edr_time.png).
