# Current agent state

Completed: svd-simple-improvements-2026-09-30-r2; PLAN.md is done.

Live ADP/solver/SVD.py equals the validated candidate byte-for-byte.
NUMERICAL option precondition_v=True (default False): positive-ridge iterative
v system right-scales both augmented blocks and preserves the ridge center.
Original-coordinate certificate and strict unpreconditioned fallback retained.
Energy cache capped at 16 MiB; larger J*d uses streamed diagonal reduction.
Zero ridge and small-d direct path retain prior behavior. EXACT optimization
removes nested LinearOperator callbacks by default. Nonfinite certificate/norm
rejected. Existing estimator, rank modes and defaults retained.

Corrected selection: 108 fixed-g rows (seeds110..112) and 36 full fits
(n320,seeds120..122); all gates passed. Untouched validation: 108 fixed-g rows
(210..212) and 36 full fits (n320,220..222); all gates passed, no errors.
Validation scaled median time/pass ratios .2255/.2381; default isotropic time
ratio .9430; max relative B difference5.22e-7, projector disagreement2.76e-6,
full-fit quality change2.27e-8, identical outer step counts. Evidence is bounded
numerical equivalence/performance, not new EDR recovery or global optimality.

59 shared tests passed, including 18 SVD tests (9 new); Ruff/format/diff checks
passed. Configured Pyright timed out after 180s, exit124; full type checking is
unverified. Baseline/prototype bounded checks also did not finish.

Artifacts: experiments/svd_improvements_2026-09-30/{REPORT,AUDIT}.md,
VERIFICATION.json, baseline/candidate snapshots, manifests, raw rows and
selection_corrected_summary.json/validation_summary.json. Invalid original
n220/d150 fixture and failed initial selection summary retained; protocol
repair used fresh full-fit selection seeds without changing solver or gates.
Routes updated: agent-notes/ADP/multi-index.md and agent-notes/tex/README.md.
Durable rationale appended to agent-notes/DECISIONS.md.

No remaining implementation step in this task. Previous unfinished SVD
documentation task is archived in this artifact directory's previous_plan.md
and previous_state.md; its completion is outside the current scope. Existing
dirty TeX/recovery files were preserved. No commit made.
