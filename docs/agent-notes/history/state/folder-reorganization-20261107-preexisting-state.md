# Current state

Task adp-moment-integration-20261030 DONE. PLAN.md is plan-of-record.
Read agent-notes/adp-moment-integration.md ->
verification/adp-moment-integration-20261030/{REPORT,THEORY,R1_AUDIT,FINAL_AUDIT}.md,
PROTOCOL.json, AUDIT.json, raw selection/validation rows, paired_edr.png/pdf.

320/320 main fits completed without errors: 240 selection,80 validation.
Residual start/adp ratio.9425 upper1.0305FAIL; proposal/(20+20 restart)1.0034
upper1.0646FAIL; these validation arms untouched. Proposals accepted6/176calls.
WLS2/WLS1 selection.34365PASS; independent validation.44798 percentile
[.40309,.49591],pairedtime1.3207,RSS.9998,worstEDRdelta-.09369PASS. Quadratic
meanEDR.5623→.2607,recovery(e<=.3)0/16→12/16. Same residual start/ridge1e-4;
not a held-out ordinary-ADP comparison. Anisotropic stress1.3372→1.3334 remains
poor; all-inner stationary only1/16 WLS2 quadratic fits. No global theorem.

New experiments/adp_moment_integration.py and integration_study.py; four new
contract tests in tests/test_adp_moment_integration.py. 38 focused tests passed;
six new Python files Ruff/format/Pyright clean, diff-check clean. Independent
review accepted scoped proofs and all saved artifacts; projector metric error
<=7.8e-16, independent initializer projector agreement<=3.76e-14. Iterative fits
NOT rerun by audits; U/I and per-step indices not saved, so no full inner replay.
No production/default/API changes. Baseline explicitly fit_index+Grassmann20,
not presumed public wrapper default. Frozen source/snapshot hashes intact.

Default first window Euclidean independent of start mathematically; later
adaptive windows differ. Wrapper retains O(Jn) weights (1.6MB here), WLS O(Ed)
neighborhoods and O(d^2) complement; no production streaming improvement claim.
Private initializer patch CPU/new/all only, restored, not thread-safe. No edits
to frozen sources/tests under same task id; future hypotheses need fresh seeds.
Prior global weak-moment/profile study unchanged; its results are not evidence
for integrated ADP. Preexisting unrelated dirty work was preserved.
