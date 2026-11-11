# Current state

Task `grassman-combined-2026-10-22` **done**; PLAN.md preserves evidence.
User explicitly requested integrating prior joint accuracy/time ideas below
old adoption thresholds, then increasing computation. This authorizes the
new public defaults; it is not evidence of added quality over relative10.

## Public behavior

ADP/solver/grassman.py::solve now defaults to max_steps20,relative_gnTrue,
quadratic_trials3,constant_centersTrue,smooth_max_stepsNone.
Relative intrinsic GN retains audited rank/workspace guards and normalized
mu=lambda_prox*||J||F²/(q*m); smooth GN damping unchanged. Rank-one fallback
uses lowest actual Armijo-accepted quadratic trial with halving backup.
Only exactly zero U_j centers are reduced, restoring full coefficient rows,
loss/history/rank counts and original gradient stopping scale. Certificates
apply to the active objective plus exact constants; all-zero input retains
conservative prior status. Same estimator/rank cutoff/tol/local_ridge0.
GN extra workspace16MiB unchanged; reduction additionally copies active
input statistics O(JPd), documented separately from the GN workspace bound.

Current use: ADP_solver(grassman.solve). Do not pass lambda_prox in fixed
settings; ADP_Config.lambda_penalty supplies it (default.05).
Earlier accepted10: max_steps10,relativeTrue,quadratic0,constantFalse,smooth5.
Original control: max_steps5,relativeFalse,quadratic0,constantFalse.
R-SDR labels pin all flags: ADP-Grassmann=original control, relative=accepted10,
combined10=integration10/smooth5, combined=current20/no cap. DefaultMETHODS
unchanged. Historical scalar candidate and polish explicitly pin old settings.

## Evidence

COMBINED_{MATH,PROTOCOL,REPORT,VERIFICATION}.md under
experiments/grassman_accuracy_2026_10_18/; exploratory pilot_combined.json
(seed90800,24 fits,12/20/30 budgets). Freeze20 before fresh data.
Raw stages: benchmarks/results/outputs/r_sdr/grassman-combined-{dev,heldout}-2026-10-22/.
Each150/150 successful,3cases(d5/20/50,n500,m2)*10seeds*5methods, actual
R MAVE::MEANMAVE. Development107800/validation108800 have identical executed
source hashes/configs, matching30 input blocks/stage and no retuning.
Development20/original EDRgeo.918800 CI[.864972,.970248],time.842937/1.232421.
Validation20/original EDRgeo.925311 CI[.861556,.983259],time.860129/1.189895,
RSS1.00403,orth4.32e-15,worstEDR1.33552; all8 historical gates pass.
Validation20/relative10 EDR1.003418 CI[.986369,1.028379],time1.275697;
20/combined10 EDR.996743 CI[.974926,1.023747],time1.306993. No demonstrated
additional accuracy gain from components/budget; do not claim it. R has lower
EDR in all case medians; combined20 faster atd20/50, slower atd5. Validation
2614 accepted inner steps vs1233 combined10; only1/140 inner calls certified,
all outer stops h_min. No global/outer convergence claim or full-catalog claim.

Current public/control replay120/120 EXACT bases/all diagnostics/stops on
consumed validation inputs: combined_public_replay.json. Current source set
matches replay; grassman SHA6610d885741dac4d604fe49e3e73120cc08c71b55cd75afea7ae185771cebfd7.
247 focused tests pass; Ruff/format17 files and Pyright14 clean.
Whole suite717 passed/28 CUDA skips/7 failed/3 collection errors; same failures
with prior public solver. COMBINED_VERIFICATION.md names cases; full_pytest.txt
and unrelated_baseline_check.txt preserve output. Failures are manifold default
local_quadratic withm2 and multiv2 catalog KeyError mi-spokoini-m2-n.

Updated routes: ADP/README.md, solvers.md, multi-index.md, r-sdr-comparison.md;
SRC-GRASSMAN1..806, intrinsic323..354, solve355..760, block762..806.
Prior relative-GN plan archived in history/plans/; original105800/106800 evidence
retained. Old drifted partial50-case catalog must not be resumed. Luna quota
was exhausted, so bounded local analysis used while unavailable. No active run.
No commit/push performed; preserve existing user/research modifications.
