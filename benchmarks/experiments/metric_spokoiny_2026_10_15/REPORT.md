# Metric profiled descent on the hard Spokoiny full fit

## Outcome

The frozen development gate failed and no held-out seeds were run. At five solver steps, the paired median metric2/core fit-time ratio was 0.982 (the gate requires at most 0.909 for the speed route), while seed 1 projector error regressed by 5.666e-4, exceeding the 1e-4 per-seed limit. The alternative quality route also failed: median projector reduction was 0.043%, far below 20%. All paired data, truth, initial-index, and first-statistic hashes matched. No variant is promoted. The five-step full fits stopped at the scheduled `h_min`; this is not optimizer convergence.

## Full-fit development rows

Cold worker subprocesses; fit wall includes `.fit` setup but excludes imports and problem generation. Solver wall is the sum of wrapped solver calls. RSS is Linux `ru_maxrss` in KiB. Principal-angle columns are the two sine values; projector error is their RMS. The final live-gradient value is from the last inner solve. Rank loss is the sum of local rank-loss counts over outer solver calls.

### Five-step budget

| seed | arm | fit s | solver s | peak RSS KiB | sin(theta1), sin(theta2) | projector error | final gradient | accepted inner steps | rank-loss sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | core | 24.684 | 3.042 | 117100 | 0.048752, 0.059094 | 0.054171 | 0.0188 | 810 | 38 |
| 0 | metric1 | 24.318 | 2.573 | 116416 | 0.047878, 0.059230 | 0.053854 | 0.0531 | 810 | 44 |
| 0 | metric2 | 24.231 | 2.608 | 112488 | 0.048494, 0.059264 | 0.054148 | 0.0198 | 810 | 35 |
| 1 | core | 23.584 | 2.388 | 116148 | 0.046383, 0.048163 | 0.047282 | 0.0282 | 810 | 0 |
| 1 | metric1 | 23.211 | 2.148 | 117332 | 0.046135, 0.047394 | 0.046769 | 0.0571 | 810 | 0 |
| 1 | metric2 | 23.327 | 2.300 | 116796 | 0.046608, 0.049057 | 0.047848 | 0.0178 | 810 | 0 |
| 2 | core | 27.533 | 6.121 | 112512 | 0.038279, 0.050730 | 0.044938 | 0.0335 | 810 | 121 |
| 2 | metric1 | 24.717 | 3.203 | 112200 | 0.039186, 0.050212 | 0.045037 | 0.0747 | 810 | 125 |
| 2 | metric2 | 24.953 | 3.361 | 114860 | 0.038388, 0.050437 | 0.044819 | 0.0402 | 810 | 122 |

### Twenty-step budget

| seed | arm | fit s | solver s | peak RSS KiB | sin(theta1), sin(theta2) | projector error | final gradient | accepted inner steps | rank-loss sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | core | 31.826 | 10.718 | 118660 | 0.049151, 0.059295 | 0.054459 | 0.000484 | 3240 | 38 |
| 0 | metric2 | 32.319 | 11.261 | 118192 | 0.049168, 0.059284 | 0.054461 | 0.000503 | 3240 | 37 |
| 1 | core | 28.404 | 7.795 | 113360 | 0.045916, 0.050405 | 0.048213 | 0.0014 | 3240 | 0 |
| 1 | metric2 | 32.477 | 11.827 | 119816 | 0.045886, 0.050496 | 0.048246 | 0.00137 | 3240 | 0 |
| 2 | core | 40.855 | 18.844 | 112460 | 0.038233, 0.051426 | 0.045312 | 0.002 | 3240 | 110 |
| 2 | metric2 | 36.941 | 15.783 | 114552 | 0.038235, 0.051417 | 0.045308 | 0.00521 | 3240 | 117 |

At five steps each fit used 162 outer iterations and accepted 810 inner steps; the final inner solve stopped at `max_steps`. At twenty steps the corresponding totals were 3,240 accepted inner steps. Profile fractions (median over seeds) show statistics at 58.5%/59.3%/59.7% of core/metric2/metric1 fit time, solver at 12.6%/11.0%/10.8%, and outer updates at 22.5%/22.6%/22.7%. These whole-fit shares limit any end-to-end benefit from changing only the solver.

At five steps metric1 gives a slightly lower projector error than core for seeds 0 and 1, but a slight regression for seed 2; its last inner gradient is not smaller consistently. It remains a diagnostic ablation, with no promotion. At twenty steps the paired median metric2/core full-fit time ratio is 1.01549 (per-seed ratios 1.01549, 1.14340, and 0.90419). Metric2 and core recover nearly identical projectors, with mixed paired timing and no robust speed advantage.

## Frozen-statistic controls

Each cell is a median over the three seeds at one captured phase. Budget 5/100, elapsed solver seconds, objective ratio relative to that snapshot’s initialization, independently recomputed full-gradient certificate, truth projector error, and count of the three rows that meet smoothness plus certificate <=1e-6. These are paired fixed-statistic comparisons, not adaptive full fits.

### Budget 5

**Early**

| arm | time s | objective ratio | gradient certificate | projector error | certified rows |
| --- | --- | --- | --- | --- | --- |
| core | 0.01029 | 0.825121 | 0.0838 | 0.504972 | 0/3 |
| metric2 | 0.00944 | 0.965283 | 0.161 | 0.623590 | 0/3 |
| metric1 | 0.00816 | 0.968121 | 0.168 | 0.621352 | 0/3 |
| isotropic2 | 0.00923 | 0.967300 | 0.162 | 0.624044 | 0/3 |
| isotropic2_matched | 0.00948 | 0.964461 | 0.16 | 0.623607 | 0/3 |

**Middle**

| arm | time s | objective ratio | gradient certificate | projector error | certified rows |
| --- | --- | --- | --- | --- | --- |
| core | 0.01942 | 0.947725 | 0.021 | 0.168764 | 0/3 |
| metric2 | 0.01662 | 0.968906 | 0.0751 | 0.169401 | 0/3 |
| metric1 | 0.01578 | 0.975532 | 0.0974 | 0.171101 | 0/3 |
| isotropic2 | 0.01644 | 0.972785 | 0.0792 | 0.170284 | 0/3 |
| isotropic2_matched | 0.01757 | 0.961256 | 0.0518 | 0.168692 | 0/3 |

**Late**

| arm | time s | objective ratio | gradient certificate | projector error | certified rows |
| --- | --- | --- | --- | --- | --- |
| core | 0.00991 | 0.958482 | 0.0265 | 0.045054 | 0/3 |
| metric2 | 0.00931 | 0.959456 | 0.0219 | 0.045309 | 0/3 |
| metric1 | 0.00799 | 0.963070 | 0.0673 | 0.046293 | 0/3 |
| isotropic2 | 0.00873 | 0.959773 | 0.0227 | 0.045216 | 0/3 |
| isotropic2_matched | 0.00964 | 0.959162 | 0.0265 | 0.045796 | 0/3 |

### Budget 100

**Early**

| arm | time s | objective ratio | gradient certificate | projector error | certified rows |
| --- | --- | --- | --- | --- | --- |
| core | 0.15231 | 0.760398 | 7.25e-05 | 0.398095 | 1/3 |
| metric2 | 0.20275 | 0.760398 | 0.00036 | 0.397086 | 1/3 |
| metric1 | 0.17525 | 0.812302 | 0.0509 | 0.433853 | 0/3 |
| isotropic2 | 0.19637 | 0.760398 | 0.00036 | 0.397085 | 1/3 |
| isotropic2_matched | 0.22391 | 0.760398 | 0.00038 | 0.397253 | 1/3 |

**Middle**

| arm | time s | objective ratio | gradient certificate | projector error | certified rows |
| --- | --- | --- | --- | --- | --- |
| core | 0.17499 | 0.945826 | 8.87e-07 | 0.168602 | 1/3 |
| metric2 | 0.26972 | 0.945826 | 8.1e-07 | 0.168601 | 1/3 |
| metric1 | 0.66893 | 0.945826 | 9.18e-07 | 0.168602 | 1/3 |
| isotropic2 | 0.55644 | 0.945826 | 4.13e-07 | 0.168601 | 1/3 |
| isotropic2_matched | 0.73551 | 0.945826 | 9.1e-07 | 0.168602 | 1/3 |

**Late**

| arm | time s | objective ratio | gradient certificate | projector error | certified rows |
| --- | --- | --- | --- | --- | --- |
| core | 0.09471 | 0.956608 | 8.84e-07 | 0.046508 | 3/3 |
| metric2 | 0.17443 | 0.956608 | 8.81e-07 | 0.046508 | 3/3 |
| metric1 | 0.34143 | 0.956608 | 9.03e-07 | 0.046508 | 3/3 |
| isotropic2 | 0.18832 | 0.956608 | 3.3e-07 | 0.046508 | 3/3 |
| isotropic2_matched | 0.24393 | 0.956608 | 6.27e-07 | 0.046508 | 3/3 |

There were five jointly certified core/metric2 pairs, all at budget 100. Their median metric2/core wall-time ratio was 2.040; projector errors agreed to the displayed precision. Metric2 sometimes used fewer accepted steps but remained slower here, so iteration count alone did not explain the solve-time difference. The remaining 13 budget-5/100 paired endpoints did not both meet the smooth certificate; no convergence-speed claims are made for them.

## Captured coefficient metric and mathematical scope

`R` eigenvalues below come only from the independently recomputed coefficient matrix at the nine actual core snapshots (seeds 0–2, calls 0/80/160). Two snapshots (seed 0 and seed 2, call 80) were nonsmooth with local minimum rank 1; the other seven were smooth with local rank 2. The observed eigenvalue span was approximately 2.93e3–2.55e4, with condition ratios approximately 1.06–4.80 (independent lead-side audit). This is empirical localized `R`, not the outer tensor eigenspectrum; per-call outer eigenvalues are saved under the distinct `outer_tensor_eigenvalues` label. The optional metric2 capture on seed 0 is diagnostic and its timings are excluded.

The population OPG for the exact Spokoiny generator is `(5557/1155) I_2`, so this benchmark has no intrinsically weak population index. Finite-sample, noisy, localized profiled `R` can still vary, but the sampled spectra do not support a broad weak-index premise. The independent proposal audit confirms requested rank q=2 on all nine inputs, an inactive default metric floor and inactive `tan(0.5)` caps for both metric and isotropic proposals on all nine. Proposal spectral norms were only about 0.00525–0.01475 against the 0.5463 cap. Rescaling both `mass` and `G` by 100 shrank the metric proposal by 10× on all nine inputs, showing that coefficient/mass scale is a material confound. The floor-one control changes step scale, so its differences cannot be attributed to metric geometry alone. Metric proposal cosine with negative gradient ranged about 0.764–0.9997; the lowest case was the nonsmooth rank-boundary call 80 for seed 0. For m=q=2, metric2 and isotropic2 span the same left plane when `G` has full column rank; their differences are within-plane orientation/scale. The matched isotropic control additionally matches the proposal’s spectral magnitude per call, while still allowing different Armijo trajectories. Across the matched-isotropic frozen control, there is no stable quality advantage attributable to metric orientation: five-step outcomes are mixed and budget-100 certified outcomes converge to essentially the same projector error. Metric2 is slower than core on all five jointly certified pairs. The matched control itself computes the metric reference plus an additional SVD/norm scaling, so its wall time includes diagnostic overhead and does not isolate the cost of geometry. This small frozen study does not establish a recovery benefit from metric orientation. `metric_floor=1` is not presumed to activate the angle cap.

An untested follow-up hypothesis is to compare a proposal normalization calibrated against coefficient/mass rescaling, then profile a line-search path that safely reuses each accepted-point profile and polar factorization. Any such change should preserve the same objective and be checked on frozen inputs before another full-fit claim.

## Launch repair and provenance

The first frozen launcher wrote 90 `process_error` rows because it omitted `--snapshot`; no numerical solver ran. Those rows remain in `results/frozen.jsonl` and are copied byte-for-byte as `results_repaired/original_launch_errors.jsonl`. After the one-line runner fix, the repaired frozen stage completed 90/90. Original five-step fits and captures were copied byte-for-byte into the repaired results; they were not rerun. See `results_repaired/repair_note.md` and `results/launch_original.py.txt` for the old/new runner hash and imported-artifact hashes.

Raw repaired rows, the imported launch-error rows, the JSON summary, paired CSV, and PNG are in `results_repaired/`. Reproduction commands and scoped verification are in `harness_verification.md`. Positive-tau runs and holdout validation were not performed.

## Untimed solver profiles

A separate cProfile run used the jointly certified seed-1 call-160 snapshot, with each solver isolated in a fresh one-BLAS-thread process. These profiler timings are diagnostic only and are not the benchmark wall-time evidence. Both final objectives are 11.21904338; core certificate is 6.17e-7 and metric2 is 8.81e-7. Projector errors are 0.04532938 and 0.04532948. In the instrumented calls, core made 50 `_profile` calls (0.021 s cumulative) and 48 polar calls, while metric2 made 249 `_profile` calls (0.096 s) and 247 polar calls; metric2 also made 38 operator project/adjoint calls and 72 metric-direction calls. This points to extra true-profile/retraction work as a likely reason for slower frozen wall time despite fewer accepted steps in some comparisons.

The top-25 cumulative/self-time tables, `.pstats`, and final certificates are in `results_repaired/profiles/seed1_call160_{core,metric2}.{txt,pstats,json}`. Reproduce with two fresh processes (choose distinct labels to retain both artifacts):

```text
rtk proxy env UV_CACHE_DIR=/tmp/adp-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync python experiments/metric_spokoiny_2026_10_15/results_repaired/profiles/profile_solver.py --arm core --snapshot experiments/metric_spokoiny_2026_10_15/results_repaired/captures/core_seed1_call160.npz --output experiments/metric_spokoiny_2026_10_15/results_repaired/profiles --label rerun1
rtk proxy env UV_CACHE_DIR=/tmp/adp-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync python experiments/metric_spokoiny_2026_10_15/results_repaired/profiles/profile_solver.py --arm metric2 --snapshot experiments/metric_spokoiny_2026_10_15/results_repaired/captures/core_seed1_call160.npz --output experiments/metric_spokoiny_2026_10_15/results_repaired/profiles --label rerun1
```
