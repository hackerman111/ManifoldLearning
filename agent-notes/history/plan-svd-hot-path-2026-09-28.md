task_id: svd-hot-path-2026-09-28
status: done
change_class: EXACT kernels plus NUMERICAL iterative/direct v solves

# Optimize the fixed-statistics truncated-SVD solver

Goal: remove redundant passes and large temporaries from `ADP/solver/SVD.py`,
then profile the remaining solver path and fix only measured bottlenecks while
preserving the fixed-g objective, rank constraint, public API and float64
reference behavior.

Non-goals: change the rank-r estimator, add block ALS or the isotropic
approximation, introduce float32/GPU support, or add a dual solve. A direct
primal solve is admitted only for the measured small-d regime and must retain
an explicit stationarity certificate plus iterative fallback.

Known evidence: the live v-step uses batched adjoints with a `(J,d)` temporary,
an augmented ridge operator, a strict fixed Krylov tolerance and an explicit
normal-residual check after every inner solve. The rank update recomputes
`U @ candidate_V` and the full objective. Existing heavy full-fit evidence has
many capped inner loops. The user's `(J,p,d,m)=(300,4,400,8)` microbenchmark
found flat GEMV and relaxed/warm LSMR promising; those timings are external
evidence and must be reproduced locally before a repository claim.

Hypotheses:
- H1 (EXACT): contiguous `Uf`, shifted `damp`, cached `W=U@V`, factor objective,
  reused `Uv`, Cholesky ridge solves, and factor reuse remove dominant U passes
  without materially changing `B` or objective history.
- H2 (NUMERICAL): a bounded adaptive Krylov tolerance plus shifted warm start
  reduces LSMR iterations while retaining the explicit final stationarity
  certificate and final objective on representative full fits.
- H3 (NUMERICAL): for positive ridge and small d, forming the normal matrix and
  solving it by Cholesky can beat repeated GEMV while retaining the same
  subproblem and an explicit certificate. Keep the dimension gate conservative
  and fall back to LSMR on factorization/certificate failure.

Mathematical invariants: minimize the stated unhalved SSE plus
`lambda_penalty * ||B-P||_F^2`; keep positive-mass weighting, rank at most r,
orthonormal A/V factors, non-increasing accepted objective, explicit v-step
stationarity certification, finite/degenerate checks, float64, and bounded
`O(Jpd + md + Jpr)` storage. Ill-conditioned QR factor transforms must fall
back to a direct `U @ candidate_V` evaluation.

Exact read set: `ADP/solver/SVD.py`, `tests/test_svd_solver.py`, `SVD.tex`
sections 1-8, direct SVD callers, `benchmarks/svd_vs_hybrid.py`, and the
SVD/multi-index routing notes. Broader repository reconnaissance is read-only.

Work units:
- [x] P0: record a reproducible baseline with U-pass counts, wall time, peak
  traced allocation, objective, B and LSMR iterations on small reference,
  user-shaped synthetic and one representative full-fit case.
- [x] P1: implement and verify exact layout/damp/cache/small-system changes
  against dense references and the P0 numerical trajectory.
- [x] P2: benchmark adaptive tolerance/warm start separately; keep it only if
  final stationarity, objective and full-fit quality remain within declared
  tolerances, and record changed iteration counts.
- [x] P3: implement the measured positive-ridge small-d Cholesky path behind a
  bounded dimension option, certify/fallback, then run focused solver/public-fit
  checks and the three-seed representative benchmark.
- [x] P4: update the affected route, decisions, state and final evidence;
  format/lint touched code and check the diff. Pyright did not finish within
  several minutes and was stopped; this limit is recorded rather than inferred.

Verification: dense v-step and objective references including ridge=0,
collinear U and unequal masses; LinearOperator adjoint identity; old-versus-new
fixed-seed B/objective comparison for exact changes; warm-start/failure/cap
coverage; measured time, peak Python allocation, U-pass and LSMR iteration
counts; representative full-fit projector/objective diagnostics.

Stop conditions: reject a transformation if it increases the certified
objective, weakens rank/finite handling, or changes the exact-path result beyond
roundoff without being isolated as NUMERICAL. Stop profiling when remaining
cost is BLAS work required by LSMR and no small exact change has a measured
full-path benefit. Do not promote direct solve, approximate initializer, block
ALS, float32 or GPU work from microbenchmarks alone.

P0 evidence: `experiments/svd_hot_path_2026-09-28/baseline.json` records the
user-shaped `(300,4,400,8)`, rank-3, lambda=0.7 case with one BLAS thread and
five repetitions. Median fixed-g solve was 1.0372 s; traced peak was 2,201,589
bytes; all three components hit 20 inner steps; 60 LSMR solves used 4,453
iterations and an estimated 9,225 U passes. cProfile attributed 1.194/1.212 s
to `_v_step`, with the batched adjoint slower than forward. Existing frozen
heavy full-fit results remain the representative pre-change fit baseline.

P1/P2 checkpoint: the exact/refactored user-shaped solve took 0.6961 s; strict
warm start took 0.6052 s versus 1.0372 s baseline. On the heavy three-seed fit,
strict warm start reproduced seeds 0 and 1 to the recorded precision and gave
projector distance 0.10289 versus 0.09507 on the numerically sensitive seed 2;
median time was 40.85 s versus 47.56 s. Adaptive tolerance cut the median to
29.58 s but degraded seed-2 projector distance to 0.12634, so it remains opt-in.
Fresh one-thread kernel profiling found direct/LSMR ratios 0.58 at `(8000,50)`
and 1.10 at `(1200,400)`; this opens only the bounded small-d P3 branch.

P3/P4 evidence: with positive ridge and `d<=128`, the direct Cholesky path
solves the same v subproblem with a 16 MiB weighted-design work buffer, explicit
final normal-residual certificate and LSMR fallback. On the complete heavy
seeds 0/1/2, times were 24.92/25.10/25.20 s against the 30 s target and the
47.56 s old median. Projector distances were 0.11831/0.08987/0.09291 versus
0.11831/0.08987/0.09507 originally. All direct certificates passed; zero
fallbacks. Peak RSS was 102424/102828/103312 KiB versus old 96988/99204/98172
KiB. The user-shaped d=400 path stays on strict LSMR; its final five-repeat
median was 0.6263 s, with objective 283.9943609284 versus old 283.9943606823.
The final small algebraic cleanup followed the full-fit measurements; focused
dense/public tests (5 passed) and Ruff checks passed afterward. Pyright had
no output after several minutes and was interrupted; no type-check result is
claimed.
