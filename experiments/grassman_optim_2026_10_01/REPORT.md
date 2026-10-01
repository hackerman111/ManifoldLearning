# Grassmann optimized solver verification

The `grassman_optim.py` implementation preserves the current solver's objective
and public settings while changing local profile, gradient, Jacobian, and
operator-action work. The implementation details and mathematical safeguards
are recorded in [IMPLEMENTATION.md](IMPLEMENTATION.md). Full raw rows, protocol
metadata, cProfile output, and frozen inputs are retained under this directory.

## Reproducibility

The full-fit comparison uses the existing heavy Spokoiny point from
`benchmarks/grassman_benchmark.py`: `n=800, d=50, m=2, J=800, p=10`, float64,
162 outer steps, `core_gn`, five inner steps, `lambda_prox=0.05`, and seeds
0–2. Each solver/seed ran in a fresh process with `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`; order alternated by seed. The
effective `ADP_Config` was serialized into every full-fit row. Paired X, y, and
truth hashes match exactly. Process peak RSS is Linux `ru_maxrss` in KiB and
includes imports and fit inputs.

Final source hashes were original
`070d0bb91962f23610b3e0a647e8f10fb417bac75e049305c3ef38223966c1a2` and
optimized `9e1d91142d5c951f9efb718d8149b2eff3b0d7bed6246803ddc9f44d4d5b6139`.
Both hashes stayed unchanged before and after every final run. The repository
was dirty at commit `b7fe68aa7abb10092e1a349ed0b134bdae469765`. Python was
3.14.7, NumPy 2.5.2, SciPy 1.18.1, Linux x86_64; NumPy's BLAS/LAPACK build is
recorded in the protocol JSON. Final benchmark source hash at launch was
`a32e81f64513488298edc3608df0acbb871484ec73dbcb0641eb781dda68f858`; later
formatting and removal of one overwritten temporary did not change measured
solver code or rows. Current harness hash is recorded in `VERIFICATION.md`.

## Paired full fits

The final comparison is
[`fullfit_final_hash/fullfit_protocol.json`](fullfit_final_hash/fullfit_protocol.json)
and its append-only rows are
[`fullfit_final_hash/fullfit_runs.jsonl`](fullfit_final_hash/fullfit_runs.jsonl).
The optimized/original runtime ratios are 0.942, 0.929, and 0.987, with median
0.942 (about 5.8% faster). Per-seed values are:

| Seed | Original / optimized time (s) | Optimized/original | Original / optimized RSS (KiB) | Final basis projector difference (Frobenius) |
|---:|---:|---:|---:|---:|
| 0 | 30.285 / 28.526 | 0.942 | 114268 / 114360 | 8.92e-15 |
| 1 | 31.488 / 29.247 | 0.929 | 117308 / 111436 | 1.09e-15 |
| 2 | 32.144 / 31.740 | 0.987 | 107352 / 114392 | 6.90e-10 |

Each fit used 162 outer iterations and stopped at `h_min`. Every inner solver
call stopped at its five-step cap; none reported convergence. Truth projector
distances were unchanged to the shown precision: seed 0 `.05417079516475`, seed
1 `.04728184441625`, and seed 2 `.0449379443` (optimized differs by
`5.93e-11`). Every final call used five iterations and `max_steps`. Workspace
fallback count was zero throughout.

All aggregate inner iterations were 810 for each solver on every seed. Seed 0
had 620 GN solves and 190 rank-guard fallbacks for both solvers; profile
evaluations were 3390 original and 3427 optimized. Seed 1 had 810 GN solves,
810 profile evaluations, and no rank fallbacks for both. Seed 2 had 205 GN
solves and 605 rank-guard fallbacks for both; profile evaluations were 8916
original and 8794 optimized. These differences are retained rather than
rounded away.

Per-call profile objective histories were not identical on all seeds. Maximum
absolute original/optimized call differences were `7.41e-5` on seed 0,
`2.77e-13` on seed 1, and `3.436` on seed 2; median absolute differences were
`1.39e-8`, `3.20e-14`, and `2.42e-6`. The seed-2 maximum occurs at call 106
(original `32.929535`, optimized `29.493777`). Both calls took five rank-one
fallback steps. This large per-call difference is an outer-path effect, not a
frozen solver mismatch: the call-106 captured bases differ in projector norm by
only `3.60e-8`, while `U`, `I`, and mass differ by relative Frobenius/norms of
8.60%, 12.99%, and 2.19%. Solving either captured input with both solvers
reproduces the same output within `1.7e-8` objective and `2.4e-8` final
projector Frobenius difference. At these inputs every local system has rank 2,
but the profile is marked nonsmooth by its conditioning guard; both methods
use five rank-one fallback steps. Full diagnostic details and both
captured inputs are in [`call106_diagnostic/summary.json`](call106_diagnostic/summary.json).
The evidence separates close same-input solver outputs from the changed inputs
at call 106. It does not identify a more specific cause for those input changes
or make a bitwise trajectory claim.

## Frozen solver calls and kernels

The final frozen series uses the saved first-outer Spokoiny snapshots, one
warm-up and 21 timed calls per solver/seed. Both implementations return the
same final profile to approximately `3e-14` and projector differences between
`1.4e-15` and `1.9e-15`. Original/optimized median call times in milliseconds:

| Seed | Original | Optimized | Optimized/original |
|---:|---:|---:|---:|
| 0 | 29.15 | 11.56 | 0.397 |
| 1 | 25.69 | 11.45 | 0.446 |
| 2 | 31.11 | 11.87 | 0.382 |

The per-solver/per-seed isolated-process results and corrected row-basis
projector aggregation are in
[`frozen_final_hash/frozen_protocol.json`](frozen_final_hash/frozen_protocol.json)
and `frozen_final_hash/frozen_runs.jsonl`.

The original solver's frozen seed-0 cProfile took 39 ms with instrumentation.
Its largest costs were `_profile` (10 ms across seven calls), `_gradient`
(8 ms across six), 18 SVD calls (8 ms cumulative), `_core_jacobian` (3 ms),
five least-squares solves (2 ms), and polar retractions (2 ms). The optimized
frozen cProfile took 20 ms: `_profile` 5 ms, `_project` 3 ms across seven
actions, `_core_jacobian` 3 ms, `_gradient` 2 ms, five least-squares solves
2 ms, and polar retractions 2 ms. Instrumented cProfile times are diagnostic;
the warmed repeated timings above are the performance comparison.

For the single-thread `(J,p,d,k)=(800,10,50,2)` projection primitive, timings
include making the returned `(J,p,k)` result C-contiguous. Contiguous `U@Y`
and `U@V` transpose-GEMM-plus-copy took `.316/.319 ms`, versus batched
matmul `1.096/1.019 ms`, flat GEMM `1.008/1.075 ms`, and stacked GEMV
`.304/.306 ms`. For feature-stride-2 input with the boundary conversion
included, transpose-GEMM-plus-copy took `.413/.416 ms`. All alternatives
matched within `5e-14`; output stride/layout data are in
`baseline_profile/projection_microbench_copy_included.json`. The earlier raw
transpose benchmark excluded the required output copy and produced
non-C-contiguous views; it is retained, superseded by the copy-inclusive
measurement, and is not used for the claim.

In the original seed-0 full-fit cProfile, the solver ran 162 times and took
6.724 s cumulative (16.3% of 41.144 s cProfile internal time); statistics
construction took 25.912 s cumulative. This separate instrumented run explains
why the roughly 2.5x frozen-call gain becomes a smaller full-fit gain. It is not
a full-fit timing result.

## Numerical checks and retained failures

[`tests/test_grassman_optim.py`](../../tests/test_grassman_optim.py) adds
independent augmented-`lstsq` checks, multiple-RHS triangular normal solves,
ill-conditioning/absolute-scale fallback cases, `m=1/2/4`, `p<m`, ridge,
workspace fallback, non-contiguous input, finite-difference Jacobian, and a
forced multi-chunk gradient check. The original Grassmann tests also pass with
their solver reference redirected to the optimized module. Verification
commands and exact counts are in [VERIFICATION.md](VERIFICATION.md).

Initial harness setup failures are preserved under `frozen_final/` and
`frozen_final2/`; they were argument/output-path errors before numerical calls
and did not enter any timing aggregate. The successful corrected series are
under `frozen_final_hash/` and `fullfit_final_hash/`. The six-run
`fullfit_final/` directory is preserved as a preliminary series from the
pre-cleanup optimized source hash `11e9941d…ba16f79`; the final paired series
uses hash `9e1d9114…4d5b6139`. The lead's code-level rationale is documented in
[IMPLEMENTATION.md](IMPLEMENTATION.md), and the independent audit is in
[LEAD_AUDIT.md](LEAD_AUDIT.md).
