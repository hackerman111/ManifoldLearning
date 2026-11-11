task_id: grassman-solver-2026-10-01
status: done
change_class: APPROXIMATE optimizer; EXACT Schur/cache; tau>0 opt-in ESTIMATOR

# Goal / scope
Completed CPU ADP/solver/grassman.py from improvements I/II of
SVD/EDR_unified_theory.tex; paired heavy Spokoiny comparison assigned to Luna.
Existing solver defaults/statistics/outer selection preserved. New solver
default core_gn,max_steps5; no permanent chordal/coeff penalty at tau0.
Prior completed theory plan/state: experiments/grassman_2026_10_01/PREVIOUS_*.

# Invariants / approach
Same unpenalized minimum-norm finite-sketch profile as HPAO at tau0,
external mass, local cutoff, all q*m coefficient cross terms, orthonormality,
monotone accepted profile. SVD fixed-g rank-r objective remains distinct.
Full eq:dg core GN + polar Armijo; rank_one Schur/refit and adaptive spectral
controls. Rank/conditioning guards use value-based rank-one, workspace
fallback spectral. Boundary stationarity not certified; final live U check.
No dense d*d projector/Hessian; U[Y,V] cached within frozen basis only.

# Exact read set
AGENTS, WORKFLOW, numerical/research/engineering contracts;
agent-notes/ADP/{multi-index,solvers}; theory:288-583;
ADP/solver/{SVD,LSMR,_multi_operator}; pyproject; targeted solver tests;
benchmarks/svd_vs_hybrid.py:27-146 and old saved Spokoiny/LSMR-SVD protocols.

# Completed units / acceptance evidence
- [x] Heavy case n800 d50 m2 J800 p10, full162 outer steps and split seeds0–2.
- [x] Solver +18 independent tests: augmented lstsq/Schur, FD full Jacobian,
  FD-coordinate ridge, adjoint/gradient, gauge/rank/stress/noiseless recovery.
- [x] Final focused shared suite110pass. Ruff solver/tests/harness passes;
  Pyright solver zero errors/warnings (system paths, project standard config).
  Full collection4 existing errors; broad excluding4:388pass28skip19 existing
  failures, reproduced separately. Details VERIFICATION.md/focused_tests.txt.
- [x] Luna fullfits core20 and core5, each3 paired seeds/current SVD.
  core20 ratio1.260; core5 ratio.874, all3 lower projector errors, +13MiB RSS.
  Every core inner call hits cap, no stationary certificate; no heldout claim.
- [x] Frozen6methods*3seeds plus61-angle7warmed repeats. Schur/refit rank-one
  paired speedup5.72x; curve sharedprojection-inclusive23.2x; abs error3.41e-13.
  CoreGN .0949s vs adaptive spectral .2157s at similar loss. No observed
  adaptive-rank advantage; exact QR compression inapplicable p10<d+1.
- [x] Lead reviewed raw rows, aggregation/hashes, source formulas and claims;
  corrected preliminary report speedup and scalar projector wording.
- [x] Routes ADP/{README,multi-index,solvers}, STATE/DECISIONS updated.

# Evidence / limitations / stop conditions
experiments/grassman_2026_10_01/{REPORT,VERIFICATION}.md and benchmark/ raw
protocols, rows, snapshots, histories, retained4 harness setup errors.
SVD old25.734 vs now34.859 same seed/iterations/output; execution environment
changed (Python3.13/Clang→3.14/GCC; old BLAS unspecified); cause unisolated.
Use current paired times. New default5 is measured outer-budget tradeoff,
not generic convergence/global/recovery guarantee. Required evidence exists;
no unrelated catalog/manifold/dependency repair or further tuning required.
