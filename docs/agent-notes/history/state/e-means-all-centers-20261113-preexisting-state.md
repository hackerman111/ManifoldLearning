# Current state

Task e-means-adp-20261113 is **done: negative selection**, not a complete
mathematical solution improving EDR. User requested ideas of arXiv:2611.09501
while preserving the original anisotropic ADP, multiple families, exact
scoped proofs and adversarial checks. Public ADP source/API/defaults intact.

Read docs/agent-notes/e-means-adp.md -> benchmarks/experiments/e_means_adp/
{REPORT,MATH,REVIEW,README}. Four families assessed before selecting isolated
APPROXIMATE mini-batch Grassmann proposals, optional low-rank projector EMA,
original full-center finalization and full-value guard. Kernel moments,
center/sketch seed streams, initialization, alpha/h and outer best formulas
unchanged. New estimates still change the realized later adaptive weights.

Frozen selection:320/320 successful fits,8cases x10seeds x4arms,80 pairs/arm.
Original20 mean EDR.408012,36/80 at EDR<=.3. Full6 .453989,time ratio.662642;
mini .439041/.870907; mini+EMA .437432/.886412 (33/80 recoveries).
EMA delta+.0294205,worst+.420027; worst/per-case/bootstrap quality gates FAIL.
No arm selected. Validation seeds1132400..1132409 untouched, no validation
data generated, no retuning or promotion. All1680 inner calls uncertified;
new variants have0/420 stationarity-applicable calls each.

Evidence: benchmarks/results/verification/e-means-adp-20261113/
campaign/{PROTOCOL.json,environment.json,source,selection,completion.json};
analysis.json,by-case.csv,final-audit.json,final-{tests,types,lint}.log.
Independent final audit5747 checks,0 integrity failures,1600 unchanged
artifact hashes. All68 public ADP Python files match frozen hashes, no Git
ADP source changes.78 final targeted tests pass; Ruff/Pyright/format/diff/link
checks pass. Independent adversarial review found no remaining blocker.

Strongest scoped proof: gauge-correct PSD EMA and input-relative single-scale
profile descent under the stated exact-evaluation/rank domain. Implementation
guard compares numerical values, not interval-certified ideal F. No theorem
of original20/EDR superiority, adaptive convergence or universal speed.
Negative outcomes and exact remaining gap are explicit in REPORT/MATH.
Each new fit requires a fresh MiniBatchSolver; its RNG advances across scales.
Nested full_finalization describes an attempt, top-level certificate the
returned full-refit result. The prototype remains opt-in research material.

Future work needs a NEW hypothesis/protocol and fresh selection seeds.
Do not tune this failed arm on the untouched validation set. Corrected stale
SRC-GRASSMAN route in ADP/README to live372/405/813 ranges.

Previous dirty r_sdr work preserved. Prior PLAN/STATE archived under
history/{plans,state}/e-means-20261113-preexisting-*.md. Separate large-fourteen
campaigns were not stopped/restarted/analyzed; their last recorded launch
state is in that archive. No commit was made.
