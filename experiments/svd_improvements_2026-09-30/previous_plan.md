task_id: svd-consolidated-theory-2026-09-29
status: active
change_class: documentation; proposed variants explicitly classified

# Goal
Create SVD/SVD_solver.tex: one standalone Russian account of all SVD/ inputs,
with corrected derivations, limitations and three research directions:
second solver, GPU batching/parallelism, initialization/direction selection.
Preserve user source files and numerical code. Prior active recovery plan/state
are archived in SVD/documentation/previous_{plan,state}.md; that research is
unfinished and is not part of this documentation task.

Known evidence: rank(B)<m loses directions; rank(B-P) is a distinct objective
constraint; a fixed-g certificate does not establish adaptive ADP recovery.
Invariants: explicit shapes and lambda assumptions; distinguish proven facts,
conditional identities, measured results and new proposals; no unsupported
speed/convergence claims; Russian terminology; all input files accounted for.

Read set: six files in SVD/ (chat_2.md is empty), live SVD.py exact formulas,
LSMR.py operator contract, multi-index source locators, prior comparison REPORT,
research/numerics/workflow contracts. Primary external papers/docs only when
needed to verify cited theory or GPU proposals.

Work units:
- [x] D1: read-only corpus scout and targeted audit; map contradictions/coverage.
- [x] D2: write standalone TeX with derivations and three detailed directions.
- [ ] D3: verify identities on small deterministic examples; compile twice,
  check references/layout/Russian terminology; record coverage and findings.
  Algebra passed: check_math.py (22 identities, additional inequalities and
  two perturbation/candidate metrics); final stable PDF/layout pass pending.
- [ ] D4: update affected routing, PLAN/STATE/DECISIONS; deliver TeX and PDF.

Acceptance: all nonempty sources mapped; corrected rank/whitening/gradient/
stationarity statements; no opaque chat citations; complete TeX compilation;
three directions each have equations, restrictions and measurable checks.
Stop: flag unverified claims as hypotheses; do not implement or run full fits.
