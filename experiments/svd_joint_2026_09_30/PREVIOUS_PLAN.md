task_id: svd-gradient-gain-2026-09-30
status: done
change_class: APPROXIMATE (explicit search variant, same fixed-g objective)

# Goal
Implement a test variant selecting rank-one SVD gradient directions by exact
functional decrease, SVD/SVD_solver.tex eq:Q, eq:gain and subsection
"Выбор направления по уменьшению, а не только по градиенту".
Default alternating search remains unchanged. No joint factor optimization,
new estimator, preconditioned/random candidate families or recovery claim.

# Evidence and invariants
Baseline: 51 SVD/metric tests pass before edits. Q=-grad(F)/2; for unit a,v,
R=a.T Q v, D=sum mass*(g@a)^2*||U v||²+lambda,
sigma=R/D and gain=R²/D. Test all nonzero singular pairs of unprojected Q,
choose largest finite gain, then reuse QR/SVD compression and scale refit.
Preserve matrix/correction ranks, metric whitening, float64, finite guards,
monotone accepted raw objective, public basis/refit and existing dirty work.
No d*d or (m*d)^2 allocation; candidates require O(m*d+J*p) scratch.
Gradient search is heuristic and rank/gain stops are not optimality certificates.

# Exact read set
ADP/solver/SVD.py::{solve_fixed_coefficients,_solve_with_metric,
_solve_fixed_coefficients_validated,solve,_objective_from_factors};
SVD/SVD_solver.tex:195-309,975-1006; tests/test_svd{_solver,_metric}.py;
agent-notes/ADP/{solvers,multi-index}.md; numerics/research contracts; WORKFLOW.

# Bounded units and acceptance
- [x] Orient, verify baseline and derive gain/shape invariants.
- [x] Add explicit rank_one_search="gradient" (default "alternating"),
  exact candidate screening and truthful scalar diagnostics.
- [x] Dense independent gradient/gain tests, smaller-gradient/better-gain
  fixture, multi-rank descent, metric compatibility, degeneracy/failure,
  public full fit/default regression: 124 broader tests pass, focused Ruff
  and diff check pass. Pyright timed out at 30 s; type check unverified.
- [x] One-thread paired fixed-g timing/peak tracemalloc pilot at d=100/300,
  both targets: 24 rows, no errors, deterministic replay, final source SHA.
  Worse gradient objectives (about 2.4x at d100, 8–9x at d300); see REPORT.md.
- [x] Refresh affected routes, decision and final state; mark done.

# Stop conditions
Fix reference mismatch/nonfinite or objective growth before experiments.
Stop at failed checks; revise the hypothesis if two conceptual failures recur.
No expansive full-fit benchmark or promotion of the experimental default.
Final evidence: experiments/svd_gradient_2026-09-30/REPORT.md, runs.csv,
metadata.json and pilot.py; tests/test_svd_gradient.py and metric references.

# Previous evidence retained
Previous completed plan-of-record archived verbatim:
experiments/svd_gradient_2026-09-30/PREVIOUS_PLAN.md.
Documents: SVD/SVD_{cur,algorithm}.{tex,pdf}, each verified 2 pages.
Metric implementation/evidence: experiments/svd_a_metric_2026-09-30/.
