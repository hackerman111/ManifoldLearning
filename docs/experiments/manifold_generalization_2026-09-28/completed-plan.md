task_id: manifold-generalization-2026-09-28
status: done
change_class: experiment only; explicit existing scale_boundary variants
---

# Manifold beyond m=1

Goal achieved: standalone experiments/manifold_generalization.py, frozen
protocol, artifacts and Russian report for m=1,2,3, varying geometry/noise.
No production estimator/solver/default changes. Existing dirty work preserved.
Prior unfinished single/multi plan/state: previous_plan.md and
previous_state.md in docs/experiments/manifold_generalization_2026-09-28/.

Evidence/read set: research/engineering contracts, WORKFLOW, ADP/manifold;
ADP_Manifold API and engine fit hooks; existing manifold recovery validator,
probe version metadata and runner _local_subspace_metrics.
Live default correction: local_quadratic supports only m=1; explicit
estimator=manifold used throughout new experiment.

Invariants: float64, rank-m analytic Jacobian, orthonormal row truth,
truth never passed to fit, independent seed streams, all failures retained,
all-center and query RMS/max threshold 0.2; solver convergence debug only.
Scalar response does not uniquely identify arbitrary varying m>1 geometry;
finite query coverage does not prove continuum recovery.

- [x] Standalone script, manifest before fits, incremental JSONL/summary,
  source hashes/environment, time/process RSS, initial/partial diagnostics.
- [x] Finite-difference gradient, dense projector reference, basis rotation,
  reproducibility and noise-stream separation self-check passed.
- [x] Main raise: 180/180 fits, 0 recovery, 149 mass + 31 rank failures.
- [x] Separate exploratory paired stop: 180/180 fits, 28 recovery,
  31 rank + 1 feasible-bracket failures; all 148 finished at mass boundary.
- [x] Final smoke: 36/36 recorded fits; empty-budget incomplete path checked;
  raw counts/order/recovery and recomputed summaries verified.
- [x] Ruff/Pyright and git diff --check passed; no full pytest.
- [x] report.md, protocol.md, verification.json; affected routes/README/
  STATE/DECISIONS updated.

Protocol: n=600,d=8,J=P=40,sync=3,lambda=.5,cg_tol=1e-6,
a=2**(1/m),h_min=3*mean(std)/sqrt(n); m=1,2,3,curvature=0,.35,.8,
noise=0,.1. Support local=(N_lin=200,N_loc=80,N_manifold=10),
broad=(300,300,30). Seeds81000..81004,512 queries,1 BLAS thread,
600-second between-fit budget. Earlier smoke support changes are documented
and artifacts preserved. Stop variant selected after raise outcomes: explicitly
exploratory, not untouched validation. No further tuning/promotion.

Outcome: stop flat recovery m1=17/20,m2=9/20,m3=2/20; curved m>1=0/80.
Center errors also fail, and oracle nearest-chart fails curved query max.
No claim of stable general manifold recovery. Follow-up requires separate
local identifiability/support/chart coverage protocol, not gate relaxation.
