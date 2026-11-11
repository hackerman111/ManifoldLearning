# Integrate useful Grassmann candidates and raise the budget

Task `grassman-combined-2026-10-22`; status **done**.
User explicitly requests integration of earlier joint accuracy/time gains,
including candidates below the former promotion threshold, then more compute.
Goal: a usable combined public solver and a measured larger default budget.
Non-goals: change statistics, ridge, rank cutoff, estimator, outer ADP or R.
Prior completed plan: agent-notes/history/plans/grassman-relative-gn-2026-10-22-complete.md.

Evidence: relative GN10 passed independent 105800/106800 paired checks;
direct Armijo10, quadratic3/10 and zero-center exclusion5 had EDR ratios
.9555/.96196/.99973 and median time ratios .95545/.82290/.87490, respectively.
Micro-QR, CG and scalar-warmup did not jointly improve full-fit error/time.
Luna analysis is requested but its last attempt exhausted quota; use bounded
local analysis while unavailable. Historical thresholds are reference metrics,
not a veto against the user's explicitly requested integration/budget increase.

Classification: EXACT zero-U center reduction; APPROXIMATE combined optimizer
and iteration budget. For U_j=0 the profile is mass_j*||I_j||², g_j=0,
with zero gradient; restore full coefficients/loss/rank counts and original
gradient stopping normalization. All other trials use actual QR/SVD refits,
Armijo, unchanged rank/workspace guards and live final certification.
No dense d*d; GN scratch stays bounded; no convergence or EDR theorem.

Read set: grassman.py; backtracking.py rank-one proposals; constant_centers.py;
RELATIVE_GN_REPORT, QUADRATIC_MATH/REPORT, CONSTANT_REPORT; relevant Grassmann
tests; experiments/r_sdr/{design,runner,worker}; numerics/research contracts.

- [x] C1: test first; port quadratic proposals and exact zero-center exclusion
  as independent switches; expose smooth budget cap. Compare against old
  prototypes/direct references, stress validation and constant-loss restoration.
  82 numerical checks then79 combined/relative/R checks pass; Ruff clean.
- [x] C2: exploratory seed90800 full fits: old5, relative10, combined10,
  combined raised budgets12/20/30, and component ablations. Save input/source
  hashes and runtime. Select one budget >10 before fresh data; checkpoint.
  pilot_combined.json + saved sources:24 full fits. Freeze20/no smooth cap.
  Relative10->combined10 EDR nearly same d5, .10206->.10197 d20,
  .19903->.18998 d50; raised20 EDR .032923/.101947/.193999.
  Combined20 takes .2186/.4820/.8519s vs old5 .3844/.6327/.6854s.
  Thirty worsens d50 to .21473/1.0607s. Single seed is exploration only.
- [x] C3: freeze 107800 development and 108800 validation, n500/m2/d5,20,50,
  ten seeds/case, old5/relative10/combined10/combined-selected/actual R MEANMAVE.
  Isolate integration and budget effects; report paired EDR/time/RSS/CIs,
  worst outliers, hashes/stops/certificates. Preserve all failures and raw runs.
  Development107800 complete150/150, no bad blocks. Combined20/original
  EDRgeo .918800 CI[.864972,.970248], time .842937/1.232421, all8 historical
  gates pass. Integration10/relative10 EDR1.01076/time.97713; raised20/10
  EDR.990503/time1.26830; combined20/relative10 EDR1.00116/time1.25091.
  Added components/budget do not establish a gain over relative10. Continue
  unchanged108800 validation; do not retune to development.
  Validation108800 complete150/150, same source/config, no bad blocks;
  combined20/original EDR.925311 CI[.861556,.983259], time.860129/1.189895,
  RSS1.00403,worstEDR1.33552, all8 reference gates PASS. Budget20/10
  EDR.996743 CI[.974926,1.023747],time1.306993. Final20/relative10
  EDR1.003418 CI[.986369,1.028379],time1.275697. Components/budget have
  no demonstrated quality advantage over relative10; user's request still
  authorizes integration/default20. R remains more accurate on case medians.
- [x] C4: enable the requested combined defaults and selected larger budget;
  retain explicit legacy options. Public/R-SDR config tests and old-reference
  checks, Ruff/Pyright and relevant suite; update routes, report, STATE/DECISIONS.
  Public defaults20/relativeTrue/quadratic3/constantTrue/smoothNone. Explicit
  legacy/relative controls retained; historical scalar/polish prototypes pin
  old settings. Current public default/control replay120/120 EXACT bases,
  all inner diagnostics and stops; source hash6610d885... matches current.
  247 focused tests pass; Ruff/format17 files and Pyright14 files clean.
  Whole repository717 pass/28 CUDA skips/7 fail/3 collection errors; failures
  reproduce with prior public solver and do not execute modified Grassmann.
  COMBINED_VERIFICATION.md names all cases and retains raw/control evidence.
  Reports, API examples, source routes and durable state updated.

Stop/revise on invalid rank algebra, nonfinite outputs, broken pairing,
unbounded allocations or catastrophic regressions. Do not consume fresh seeds
until implementation/reference tests pass and settings are frozen. Do not
retune on validation or resume drifted historical outputs.
