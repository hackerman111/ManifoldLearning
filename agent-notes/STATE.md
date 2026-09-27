# Current agent state

Active: manifold-generalization-2026-09-28, standalone experiment only.
Read PLAN.md. Prior unfinished single/multi study and its state are archived
in docs/experiments/manifold_generalization_2026-09-28/previous_{plan,state}.md;
existing dirty code and provisional artifacts remain untouched.

m>1 requires explicit estimator="manifold": live default local_quadratic
is m=1-only. Routing note default statement must be corrected.
Use frozen m=1,2,3 / curvature / noise protocol, analytic full-rank truth,
all-center and independent-query geometric checks (threshold 0.2).
Convergence is debugging info, not recovery. No tuning or solver changes.
Next: implement experiments/manifold_generalization.py, reference self-check,
bounded fits, report and checkpoints.
