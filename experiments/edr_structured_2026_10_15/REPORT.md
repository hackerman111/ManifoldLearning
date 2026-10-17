# Structured EDR benchmark results

## Protocol and decision

Settings, deterministic inputs, seeds, arms, and the 10% paired-speedup rule are in [`protocol.json`](protocol.json). Runs used serial subprocesses, one BLAS thread, a 120 s per-arm cap, `perf_counter`, and process peak RSS. The full-solve clock begins after imports at the first solve call, without an untimed solver warmup; operator and Schur-kernel paths have untimed warmups. Versions, thread pools, dirty state, protocol/harness hashes, and errors are retained in each run directory's `run_metadata.json`.

The primary finite suite had 9/9 baseline and 45/45 candidate rows. All hit the 30-step budget, so there are **no same-certificate primary speed comparisons**. The near-truth supplemental suite gave six-arm shared certificates on all seeds, but every candidate was slower than `core_gn`; nothing met the frozen 10% promotion rule. The single weak-signal supplemental case is reported below and did not trigger further tuning. Positive-`tau` stress was not run.

The `live_gradient_scaled` field in earlier rows used `max(1,2*abs(objective))`; that is not the solver certificate normalization. Use stop diagnostics or the raw gradient divided by `max(1, sum_j mass_j ||I_j||²)`. New weak-signal rows store this as `live_gradient_certificate`. No speed statement below relies on the older field.

## Primary frozen suite

Baseline `core_gn` used `(J,p,d,m)=(100,8,40,3)`, three cases and seeds 0–2. All 9 reached 30 accepted steps and stopped at `max_steps`. Median time/objective/gradient/projector error by case:

| case | ms | objective | raw gradient Frobenius | projector error |
|---|---:|---:|---:|---:|
| isotropic | 22.10 | 3.534809 | 2.425068 | 0.771740 |
| mild anisotropic covariance proxy | 20.80 | 3.098044 | 5.804045 | 0.760761 |
| correlated | 21.40 | 1.527281 | 2.059701 | 0.537010 |

All candidate rows use hash-matched `init/U/I/mass/truth` inputs. Median times were 22.1 ms core, 45.7 metric, 36.4 Schur GN, 120.2 Newton preconditioned, 116.3 Newton unpreconditioned, and 123.1 recycled Newton. Median objective deltas versus core were -1.523, -0.063, -1.531, -2.087, and -0.564 in that candidate order; quality varies by case and seed. These are budgeted objective results, not convergence or speed evidence.

Newton's primary-suite median operator counts were 32 projects / 31 adjoints and 763 CG iterations with preconditioning; without preconditioning they were 32 / 31 and 789 CG iterations. `recycle_steps=2` reduced counts to 24 / 23 and 783 CG iterations (8 recycled steps), yet was slower. Preconditioning applies only to Newton; the `schur_gn_preconditioned` arm label is retained for the suite, but GN used its direct least-squares solve and the precondition flag had no effect. Rank-guard fallback counts were zero in these timed new-solver rows.

Median subprocess peak RSS for the primary arms was 84,144 KiB core, 84,492 KiB Schur GN, 84,116 KiB Newton preconditioned, 84,064 KiB Newton unpreconditioned, 84,136 KiB recycled Newton, and 83,968 KiB metric. CX dense/factorized solver peak RSS medians were 88,420/88,412 KiB at support 8 and 94,704/94,624 KiB at support 400. These are Linux `ru_maxrss` KiB readings, not portable byte counts.

The `weak_index` generator raises the first three covariance eigenvalues to 1.15 but draws the planted subspace independently. It is a mildly anisotropic covariance proxy, not an aligned weak-index test. Same-budget objective/projector-error deltas versus core, by seed 0/1/2:

| arm | objective delta | projector-error delta |
|---|---|---|
| Schur GN | [-0.063, -0.494, -0.071] | [-0.007, -0.220, -0.004] |
| Newton preconditioned | [-5.367, -0.494, -2.965] | [-0.754, -0.220, -0.695] |
| Newton unpreconditioned | [-5.272, -0.494, -2.966] | [-0.685, -0.220, -0.699] |
| recycled Newton | [-2.558, -0.494, -0.482] | [-0.189, -0.220, -0.100] |
| metric | [-5.367, -0.494, -2.405] | [-0.753, -0.221, -0.402] |

## Supplemental recovery and kernel evidence

The near-truth noiseless calibration used `(80,8,12,2)`, `tau=0`, and max 100 steps. All six arms certified on all three seeds. Median times were core 5.32 ms, metric 15.94 ms, Schur GN 9.70 ms, and Newton variants 25.18–26.32 ms. Candidate rows did not meet the promotion rule.

The one predeclared weak-signal calibration scales planted coefficient column 1 by 0.03, with the same shape, initialization, seeds, and budget. All 18 rows succeeded. Metric and all Newton variants certified on 3/3 seeds; Schur GN certified on 3/3; core certified on seeds 0/1 and reached `max_steps` on seed 2, with certificate gradient 1.006e-6. Median solver times were core 22.76 ms, metric 13.66 ms, Schur GN 44.86 ms, Newton preconditioned 26.03 ms, Newton unpreconditioned 24.36 ms, and recycled Newton 27.91 ms. On the two seeds where both metric and core met the same certificate, metric had a 1.6621x paired median speedup and certified on 3/3 seeds versus core's 2/3. Metric's median projector error was 9.00e-6 versus core's 9.22e-4. This is a positive metric pilot for this one supplemental case; two matched pairs are too narrow to promote a default or general claim. Projector errors by seed 0/1/2:

| arm | projector error |
|---|---|
| core GN | [9.22e-4, 7.10e-4, 1.16e-3] |
| Schur GN | [9.44e-4, 6.99e-4, 1.19e-3] |
| Newton preconditioned | [1.36e-5, 2.13e-5, 1.70e-5] |
| Newton unpreconditioned | [1.39e-5, 2.13e-5, 1.71e-5] |
| recycled Newton | [1.39e-5, 2.13e-5, 1.70e-5] |
| metric | [1.26e-5, 9.00e-6, 5.06e-6] |

Each row stores the planted truth projector. This calibration shows much lower planted-subspace error for Newton/metric despite small residuals; it is one frozen weak-signal instance and does not establish general recovery or a promotion result.

The inactive-block Schur kernel used `J=100,p=12,d=40,m=6,q=2,a=2`, three distinct timed trial points per seed, and one separate warmup point. Setup medians were 0.751–0.765 ms; per-trial evaluate+HVP medians were 0.175–0.188 ms versus 0.810–0.852 ms for the matching polar/full-profile reference (about 4.5x in this kernel-only measurement). Each timed arm did 3 evaluate and 3 HVP calls; maximum profile-value discrepancy was 6.4e-16. This is not end-to-end speed. The earlier `results/schur_kernel/` reused one K after warmup and benefited from cached factors; it remains preserved but is cache-biased and is not used for the result above.

## CX construction and solver parity

For `n=400,d=100,J=100,p=8`, sparse support 8 gave median factorized/dense builder times 2.83/3.11 ms, storage ratio 0.625, and project/adjoint 0.043/0.060 and 0.071/0.057 ms. Full support 400 gave builder 14.04/4.93 ms, storage ratio 6.505, and project/adjoint 0.667/0.061 and 0.460/0.056 ms. Statistics and operator actions matched to floating-point precision. CSR saves storage only for sparse windows; dense construction and actions win for full windows.

On identical dense/factorized solver inputs, sparse-support Schur GN medians were about 38.5/38.9 ms solver time; including construction gives about 41.6/41.7 ms. Full-support medians were about 37.8/71.0 ms solver time; including construction gives about 42.7/85.1 ms. Both backends stopped at `max_steps`. Maximum objective differences were 2.1e-8 (support 8) and 8.2e-7 (support 400); maximum gradient differences were 1.3e-6 and 1.3e-4 at respective gradient scales around 1e3 and 1.7e6. There is no converged solver speed claim.

## Actual-fit smoke, audit, and retained failures

The actual-pipeline smoke used `n=150,d=12,m=2,J=80,p=6`, three outer steps and seeds 0–2. Core and candidate fits stopped at the outer-step limit; inner solves stopped at `max_steps`. Candidate median fit times were 183 ms core, 230 metric, 268 Schur GN, 475 Newton preconditioned, 479 recycled Newton, and 430 Newton unpreconditioned. Projector errors were core [0.5565,0.5488,0.3455], metric [0.5928,0.5460,0.3439], Schur GN [0.5641,0.5478,0.3439], Newton preconditioned [0.4913,0.5481,0.3439], recycled [0.4911,0.5509,0.3472], and unpreconditioned [0.4912,0.5480,0.3439]. This is only a bounded smoke, not recovery evidence. Full-fit frozen snapshots had rank-guard counts 30 on seeds 1/2 because the captured profile was nonsmooth and did not construct SchurCore; primary timed smooth Schur rows had zero guard fallbacks. The later fallback-bookkeeping fix therefore did not affect any timed candidate path; no retiming was needed.

Paired candidate uniqueness/input hashes were audited: 45/45 primary rows cover 9 input groups, 18/18 near-truth rows cover 3 groups, 18/18 weak-signal rows cover 3 groups, and 12/12 CX parity rows cover 6 groups; every key `(stage,case,seed,arm,support)` is unique within its run. Corrected weak-signal run metadata records solver hashes `grassman.py=f30c1d…b3ab333`, `grassman_research.py=ce6a64…48128e`, and pre-split harness hash `7589d4…0546d97`; full hashes are in metadata. Final formatted harness hashes after a behavior-preserving split are `bench.py=3639783dbce7be4a39540ca8d141b6accb30a0f8c05fc466361a49557a895c4f`, `benchmark_workers.py=f73d9bb2ad519fe6c6df2f40e41598f129f5b25cc4ba70c4b13945500eaf1b9c`, and `schur_kernel.py=587a7c73957ace693efeb7b9a5619686673406a3b9f8bad0bde85ab23f588c52`. The weak-signal run's protocol hash is `4728c236…24d8e4`; current protocol `2ebb068b…bfd9786d` only clarifies first-solve timing and the unrun positive-tau stress. Earlier suites retain their then-current source hashes. The fallback patch changed the research solver hash after earlier timings; their affected guard branch had zero new-solver fallbacks.

Failed attempts remain in their original artifacts: the first 9 baseline arms rejected `lambda_prox=0`; `cx_operator_only` retained 6 direction-shape errors; `fullfit_frozen_baseline` retained 3 obsolete-stage argument errors. Corrected retries passed. There were no failures or timeouts in the final 45-row primary candidates, 18-row near-truth calibration, 18-row weak-signal calibration, 12-row CX parity, or 3-row corrected Schur kernel run. `protocol.json` was clarified after the final run to distinguish first-solve timing and record that positive `tau` is unrun; frozen settings were unchanged.
