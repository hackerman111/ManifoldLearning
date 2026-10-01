task_id: svd-joint-rank-r-2026-09-30
status: active
change_class: APPROXIMATE (isolated experimental optimizer, unchanged objective)

# Goal and scope
Implement simultaneous X=A M V.T optimization, full r*r core, orthonormal
A,V, spectral initialization, core least squares, horizontal descent and QR.
Callable through ADP_solver in experiments/svd_joint_2026_09_30/prototype.py.
No default replacement, new metric, GPU, global optimum or recovery guarantee.

# Evidence, hypothesis, invariants
78 baseline SVD/gradient/metric tests pass. Fixed-g objective:
sum mass||I-U B.T g||² + lambda||B-P||². Matrix: B=X, prior=P;
correction: B=P+X, shifted I, zero prior. Retain all Hessian cross terms.
Core ridge center A.T prior V; omitted prior penalty is constant only for
core solve, not line search. Hypothesis: moving both spaces improves fixed-g
objective versus greedy, possibly at increased runtime. Float64, finite
checks, rank<=r, monotone objectives, complete basis/refit and dirty files.
Augmented QR/lstsq; no d*d or (md)^2 Hessian. QR chunks capped at 16 MiB
plus O(r^4); cached U V costs O(J p r).

# Exact read set
ADP/solver/SVD.py::{_FlatU,_objective,_complete_basis,solve};
ADP/solver/LSMR.py::{_validate_inputs,_local_refit,_normalize_index};
tests/test_svd{_solver,_gradient,_metric}.py; numerics/research contracts;
agent-notes/{WORKFLOW,ADP/solvers,ADP/multi-index}.md.

# Bounded units and acceptance
- [x] Baseline, derivation and active checkpoint.
- [ ] Prototype, dense core/gradient references, invariance, ill-conditioning,
  zero ridge/rank, failure and complete ADP fit checks.
- [ ] Frozen paired selection: fixed-g d100/300/600 seeds 73001..73003,
  rank2/m4, matrix/correction; full fits d10/50 seeds 73101..73103.
  Gate: no exceptions/objective increases; joint fixed-g objective <= greedy
  within 1e-6 relative on every pair; full-fit projector distance <= greedy
  +0.02 on every pair. Record time/memory; no speed requirement.
- [ ] If selection passes: untouched validation seeds 74001..74003 and
  74101..74103, same configurations. On failure leave untouched.
- [ ] Report actual outcomes, refresh routes/state/decision, mark done.

# Stop conditions
Fix reference mismatch/nonfinite/descent violation before pilots. Two repeated
conceptual failures require hypothesis revision. Selection failure closes
promotion gate; usable prototype/tests remain the deliverable.
Previous completed plan archived verbatim in the experiment directory.
