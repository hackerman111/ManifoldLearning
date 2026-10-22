# Combined Grassmann solver and larger budget

Status: development150/150 and independent validation150/150 complete.
Combined20 settings now public defaults; final replay/checks in progress.
Request2026-10-22: integrate earlier joint error/time improvements below the
old adoption threshold, then raise computational budget. Protocol:
COMBINED_PROTOCOL.md; derivation and compatibility: COMBINED_MATH.md.

## Components selected from existing evidence

| Idea | Earlier paired EDR ratio | Median time ratio | Integration |
|---|---:|---:|---|
| Relative intrinsic GN10 | .923729 on independent106800 | .653572 | Already public option; retained |
| Direct true-profile rank-one Armijo10 | .95550 | .95545 | Fallback for unavailable GN |
| Up to3 quadratic angle proposals10 | .96196 | .82290 | Lowest actual Armijo-accepted proposal |
| Exact zero-U center reduction5 | .999729, CI includes1 | .87490 | Restore full loss/coefs/gradient scale |

These historical effects use different development datasets and are not
additive. Zero-center error gain is statistically unresolved; its exact
objective reduction and observed time saving justify inclusion under the
new request. Existing QR/SVD, bounded gradient, transpose GEMM and batched
core Jacobian optimizations were already in production and remain present.

CG, warmup, extra scalar searches, full-tangent/flat/rank-k changes and
per-center QR reuse did not jointly improve full-fit quality/time in their
recorded comparisons; isolated profile speed or lower frozen loss is not
evidence of that joint gain. They remain explicit historical variants.

## Exploratory budget selection

Only reused90800 data;24 full fits, d5/20/50,n500,m2. Saved raw settings,
input/source hashes and original sources: pilot_combined.json and
pilot_combined_sources/. Later syntactic formatting/type edits do not replace
the saved pilot code. Single-seed times are exploratory measurements.

| Variant | EDR d5 / d20 / d50 | Seconds d5 / d20 / d50 |
|---|---|---|
| Original5 | .030890 / .108376 / .230428 | .3844 / .6327 / .6854 |
| Relative10, smooth5 | .032950 / .102057 / .199034 | .1892 / .3775 / .6821 |
| Combined10, smooth5 | .032944 / .101968 / .189981 | .1929 / .3469 / .6349 |
| Combined12, no smooth cap | .032926 / .101986 / .194888 | .2052 / .3897 / .6895 |
| Combined20, no smooth cap | .032923 / .101947 / .193999 | .2186 / .4820 / .8519 |
| Combined30, no smooth cap | .032923 / .101947 / .214726 | .2203 / .5660 / 1.0607 |

Freeze20 before development107800. More steps do not monotonically improve
outer EDR. Final comparison will separate integration from added budget.

## Frozen full-fit evidence

Development: benchmark_outputs/r_sdr/grassman-combined-dev-2026-10-22/,
150/150 successful fits,30 complete matched input blocks, no failures.
Fingerprint09104de07ff12b3c72ebb5e7e33db001df2742b39f242f2511517f434c316796.
combined_analysis.json retains all seven explicit paired comparisons.

| Development comparison | EDR geometric ratio [bootstrap95] | Time median / p90 ratio |
|---|---|---|
| Relative10 / original5 | .917735 [.865303,.966869] | .678730 / .968179 |
| Combined10 / original5 | .927610 [.880049,.971082] | .640911 / .930836 |
| Combined20 / original5 | .918800 [.864972,.970248] | .842937 / 1.232421 |
| Integration: combined10 / relative10 | 1.010760 [.994905,1.032493] | .977134 / 1.028117 |
| Budget: combined20 / combined10 | .990503 [.965372,1.016035] | 1.268300 / 1.372151 |
| Combined20 / relative10 | 1.001160 [.983242,1.019267] | 1.250909 / 1.341902 |
| Combined20 / actual R | 1.439125 [1.328832,1.559773] | .489951 / 1.431232 |

Final20/original5 passes all8 former thresholds, with RSS ratio1.007686,
max orthogonality5.63e-15, worst individual EDR ratio1.140097. Improvements
are concentrated at d50; d20 median EDR is worse. Integration and extra
budget do not establish a quality gain over accepted relative10.
Twenty increases accepted inner steps1191->2602 versus combined10;
3/140 final inner calls certify stationarity, all outer stops are h_min.
R remains more accurate in each case median, with greater runtime at d20/50.

## Independent validation108800

150/150 successful fits,30 matched blocks; source fingerprints and all solver
settings exactly match development. No retuning or omitted failed outcomes.
Raw: benchmark_outputs/r_sdr/grassman-combined-heldout-2026-10-22/,
including combined_analysis.json, inputs/, source/, fits/ and manifest.json.

| Validation comparison | EDR geometric ratio [bootstrap95] | Time median / p90 ratio |
|---|---|---|
| Relative10 / original5 | .922158 [.863114,.977377] | .656330 / .908264 |
| Combined10 / original5 | .928334 [.871540,.981091] | .660171 / .908383 |
| Combined20 / original5 | .925311 [.861556,.983259] | .860129 / 1.189895 |
| Integration: combined10 / relative10 | 1.006697 [.998577,1.016538] | 1.006664 / 1.038341 |
| Budget: combined20 / combined10 | .996743 [.974926,1.023747] | 1.306993 / 1.367969 |
| Combined20 / relative10 | 1.003418 [.986369,1.028379] | 1.275697 / 1.421531 |
| Combined20 / actual R | 1.296449 [1.212391,1.393809] | .504139 / 1.433617 |

Combined20/original5 again passes all8 historical gates. Median time ratio
bootstrap95 [.777517,.920750], RSS1.004027, max orthogonality4.32e-15,
worst EDR ratio1.335523 (not a per-dataset improvement guarantee).
Twenty increases accepted steps1233->2614 versus combined10. Only1/140 inner
calls certifies stationarity, all outer stops h_min; no outer convergence claim.

| Case | Original5 EDR / seconds | Relative10 | Combined10 | Combined20 | Actual R MEANMAVE |
|---|---|---|---|---|---|
| d5 | .041120 / .30343 | .041183 / .19056 | .041183 / .19151 | .041180 / .20123 | .030941 / .15050 |
| d20 | .100680 / .62059 | .101316 / .39201 | .101316 / .40075 | .101259 / .54751 | .081787 / 1.08500 |
| d50 | .227016 / .83284 | .179425 / .71000 | .182847 / .67466 | .181903 / .88427 | .150784 / 4.53800 |

Entries are separate case medians; a paired median delta is not generally
the difference between these medians. Time is model.fit wall-clock, RSS is
isolated Linux-process /proc high-water sampling, including imports/runtime.

## Interpretation and public settings

The combined20 default is implemented at the user's explicit request for
integration and more compute. It improves EDR and pooled median runtime over
the original5 control on both stages. Extra components and budget do **not**
demonstrate an accuracy gain over the already accepted relative10; validation
EDR is .34% worse geometrically (CI includes equality), with27.57% median
extra time. The20/10 budget effect is only .33% lower EDR with CI including1,
at30.70% extra time. Do not describe these effects as established added gains.
R has better EDR in every case median; combined20 is faster at d20/d50,
slower at d5. Evidence is n500,m2,d5/20/50, not the full50-case catalog.

Current use: ADP_solver(grassman.solve) enables relative GN, quadratic3,
exact zero-U exclusion, max_steps20 and smooth_max_steps=None. To select the
earlier accepted low-cost option, explicitly pass relative_gn=True,
quadratic_trials=0,constant_centers=False,max_steps=10,smooth_max_steps=5.
For the original control use relative_gn=False,quadratic_trials=0,
constant_centers=False,max_steps=5. Do not duplicate lambda_prox in ADP_solver;
ADP_Config.lambda_penalty supplies it. No estimator/rank cutoff change.

R-SDR ADP-Grassmann stays an explicitly pinned original control;
ADP-Grassmann-relative pins the accepted10 settings; combined10/combined
record integration and larger-budget arms. Historical scalar candidate and
polish prototypes now explicitly pin old flags, preventing public defaults
from silently changing their meaning. Raw frozen outputs are unchanged.

## Verification

247 focused checks pass. Public-default/control replay120/120 exactly matches
saved bases/all inner diagnostics/stops; current source hashes match replay.
Ruff and17-file format checks pass. Whole-repository run has717 passes,
28 CUDA skips,7 unrelated failures and3 collection errors, reproduced with
the prior solver. Exact names, causes and reproducible control:
[COMBINED_VERIFICATION.md](COMBINED_VERIFICATION.md).
