# Independent adversarial review: all centers, 2026-11-13

Read-only reviewer inspected PLAN, ALL_CENTERS_MATH, changed study/audit harness,
tests and minimal public fit/calculus/initializer ranges before outcomes.
Candidate and ADP code were not changed or delegated.

Pre-freeze verdict: no blocking mathematical/protocol defect. Independently
ran32 focused tests and old-campaign read-back7348 checks without failures.
Verified finite-population indicator/covariance proof, array-byte table,
fresh protocol seeds, conditional validation controls (three in all-J mode),
old-schema fallback, actual job and effective trace N_J checks. Final trace
tampering coverage passes. Lead also obtained79 focused passes, clean Ruff,
Pyright0 errors/warnings. No candidate settings/gates were retuned.

Scope restriction carried into the report: all-J versus bounded-J measures
the FULL configuration change, including initializer, h, sketch assignment
and relative effects of absolute smooth-GN damping. It cannot isolate an
effect of center-sampling variance or batch size. No EDR theorem follows.

## Descriptive script review

Reviewer checked summarize.py before execution: target-minus-base deltas,
median of individual time/RSS ratios, full paired grid and all outer-call
diagnostics are correct. Preconditions require completion, successful rows
and audit PASS. SHA256 of the script is recorded. Median stage fractions
are separate descriptive medians and need not sum to one.

## Final outcome and claim audit

Independent final reviewer regenerated the selection audit and matched saved
results exactly:400/400 successful fits,80 complete pairs,8396 checks and2000
output hashes. Recomputed summary and CSV from raw rows. Confirmed bounded
original20 .395326 -> all20 .332018,66/80 improvements, time3.169533,
RSS1.100217 and worst all-J regression+.431130. Confirmed EMA meanEDR.358669,
time.843886 versus all20 and failed frozen quality gates. Neither validation
exists. All68 ADP files and candidate remain unchanged.

Confirmed0/2100 top-level certificates, EMA185/last235 starts and zero
finalization rollbacks. Report correctly limits evidence to selection and
the complete configuration effect; fixed-contribution variance identity
does not establish adaptive EDR or convergence. One terminology correction
applied: eight TASKS (seven families plus additive at d50), not eight families.
Final verdict: no blocking evidence/claim defect; close negative mini/EMA
selection, preserve the exact remaining gap and all held-out boundaries.
