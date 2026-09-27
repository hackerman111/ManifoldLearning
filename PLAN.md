task_id: manifold-generalization-2026-09-28
status: active
change_class: experiment only (no estimator/solver change)
---

# Manifold beyond m=1

Goal: runnable frozen experiment with m=1,2,3, flat/varying geometry,
noiseless/noisy responses, all-center and independent-query recovery.
Non-goals: estimator fixes, defaults, tuning, universal identifiability claim.
Previous active single/multi study is preserved in previous_plan.md and
previous_state.md under docs/experiments/manifold_generalization_2026-09-28/.

Evidence: old radial protocol is m=1; local_quadratic rejects m>1.
Live default is local_quadratic (routing note incorrectly says manifold).
Explicit estimator=manifold is necessary; fix that note.
Hypothesis: m>1 works on flat controls and moderately varying geometry;
curvature/noise may expose support/rank/locality limitations.

Invariants: float64, row-orthonormal truth (n,m,d), full-rank analytic
Jacobian, no truth passed to fit, separate data/noise/model/query seeds,
all failures in denominator; geometric threshold 0.2 unchanged;
convergence/completion/residuals diagnostic only. No estimator change.
Scalar response does not uniquely identify a varying rank-m distribution:
interpret recovery relative to the specified generating map.

Read set: research/engineering contracts, WORKFLOW, ADP/manifold.md;
ADP/core/manifold/ADP_Manifold.py; engine/manifol_engine/fit.py;
experiments/manifold_recovery_validate.py, manifold_recovery_probe.py
(_version), runner.py (_local_subspace_metrics).

- [ ] Implement standalone module, frozen manifest before fitting,
  incremental JSONL, per-case summary, source hashes, time/RSS/debug info.
- [ ] Verify analytical Jacobian by finite differences, basis rotation,
  deterministic generation and independence; run smoke/full within budget.
- [ ] Document protocol, observed limits, runnable commands, update routes
  and durable state; mark done when script and bounded evidence exist.

Main: n=600,d=8,J=40,P=40,N_loc=30,N_lin=40,N_manifold=10,
sync=3,lambda=0.5,cg_tol=1e-6,a=2**(1/m),h_min=3*mean(std)/sqrt(n),
scale_boundary=raise, m=1,2,3, curvature=0,0.35,0.8, noise=0,0.1.
5 independent seeds 81000..81004, 512 query points, one BLAS thread.
Smoke: same cases/parameters, seed 80000 (not main seed).
Stop: no tuning after outcomes; budget 600 seconds between fits marks
incomplete; numerical errors are recorded, never removed. A single fit
can exceed remaining budget. Query finite-sample coverage is not a proof
on a continuum. No held-out validation of an improved estimator is claimed.
