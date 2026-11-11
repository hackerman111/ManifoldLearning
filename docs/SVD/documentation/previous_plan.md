task_id: svd-high-dimensional-recovery-2026-09-29
status: active
change_class: ESTIMATOR (explicit full-rank variant); NUMERICAL/EXACT optimization

# Goal
Improve SVD full-fit projector recovery at high d and reduce runtime, with
bounded memory and explicit solver certificates. Preserve historical matrix
and correction modes and the default ADP solver. Previous completed plan is
archived in experiments/svd_recovery_2026-09-29/previous_plan.md.

Evidence: SVD/problem.md establishes rank(B)<m penalty floor and missing
EDR directions. Existing correction mode has only fixed-g evidence. Saved
25-seed Spokoiny comparison has median SVD error .118 versus LSMR .0464.
Hypothesis: solving the full fixed-g proximal least squares removes rank
truncation/completion bias and avoids repeated rank-one alternating solves.
It does not prove recovery of the adaptive estimator.

Invariants: float64, fixed data/fit seeds and statistics per comparison,
nonnegative mass, finite rank-m public basis, explicit original-coordinate
normal residual, no d*d or (J,p,m,d) allocation in the candidate.

Read set: SVD/{problem.md,SVD.tex,SVD_form.tex}, ADP/solver/{SVD.py,LSMR.py,
_multi_operator.py}, tests/test_svd_solver.py, benchmarks/svd_vs_hybrid.py,
agent-notes/ADP/multi-index.md and numerical/research/engineering contracts.

Work units:
- [ ] R0/R1: derive full fixed-g step and bounds, independent audit, profile
  baseline; exploratory pilots on seeds 73000-73002, including correction.
- [ ] R2: isolated prototype; augmented dense reference, adjoint, rank,
  degeneracy and certificate checks before full-fit comparisons.
- [ ] R3: freeze selected variant and fingerprint, selection seeds 73100-73105
  on small control, Spokoiny d50 and matched Spokoiny d100 (n=800).
  Require no failures; median error improves >=20% on each Spokoiny case,
  worst error no worse by >.02, small median no worse by >.02; median time
  decreases on each hard case, RSS <=1.2x baseline. Report all per-seed deltas.
- [ ] R4: only if R3 passes, untouched validation seeds 73200-73209 with the
  same gates and configurations. No tuning from selection/validation.
- [ ] R5: promote passing explicit option, verify production/reference path,
  update source routing, PLAN/STATE/DECISIONS and Russian report.

Stop conditions: failed frozen gate ends that hypothesis (retain artifacts,
do not consume held-out seeds); revise plan for any distinct new hypothesis.
Worker timeout 240 s; one BLAS thread; fresh process per fit. No broad suite.
