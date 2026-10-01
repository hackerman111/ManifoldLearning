task_id: svd-simple-improvements-2026-09-30
status: active
change_class: EXACT hot-path changes; NUMERICAL explicit v preconditioner

# Goal
Improve SVD using the simplest ideas in SVD/SVD_solver.tex: bounded diagonal
right preconditioning of v least-squares, and profile-supported reuse/removal
of repeated work. Preserve current objectives, rank constraints, float64,
regularization center, public defaults, diagnostics and bounded workspace.
Non-goals: GPU, block/nonconvex-search changes, full solver, estimator changes,
new recovery claims, completing previous documentation/recovery tasks.
Previous PLAN/STATE archived in experiments/svd_improvements_2026-09-30/.

Known evidence: direct Cholesky d<=128 and U@V caching already implemented;
adaptive_krylov remains false after past quality regression. Rank(B) and
rank(B-P) are different constraints. Preconditioning benefit is a hypothesis.
For A_a=diag(repeat(sqrt(mass)*(g@a),p))*Uflat, use S=diag(H_a)^(-1/2)
and x=x_base+S*y in [A_a; sqrt(lambda)I]x=[target;sqrt(lambda)z].
Keep ridge target z=(prior-low_rank).T@a, never scalar damping after scaling.
Original-coordinate normal residual remains the acceptance certificate.
No J*p*d square temporary or d*d normal matrix on the large-d path.

Read set: AGENTS, numerics/research/workflow; multi-index SRC-SVD-SOLVER;
SVD_solver.tex sec:greedy, sec:algorithm (eq:diagprecond), sec:verification;
ADP/solver/SVD.py; tests/test_svd_solver.py; benchmarks/svd_correction.py;
_multi_operator.py; pyproject.toml. Reference: frozen baseline_svd.py.

Work units and evidence:
- [x] R0: profile baseline, freeze benchmark shapes/seeds/tolerances/gates.
- [x] R1: derive right scaling and have an independent read-only audit.
- [x] R2: implement isolated candidate; dense augmented LS, adjoint, nonzero
  warm start, unequal mass/column scale/collinearity, ridge/zero guards.
- [ ] R3: frozen paired selection (seeds 110,111,112); fixed-g d100/300/600,
  both rank modes, isotropic and unequal column scales; bounded full fits.
- [ ] R4: untouched validation seeds 210,211,212 only after R3 passes.
- [ ] R5: integrate explicit opt-in preconditioner; run focused/shared checks,
  Ruff/Pyright/format/diff checks; write report, notes, decisions, final state.

Acceptance: original objective history agrees with direct evaluation and is
nonincreasing; dense v relative error <=1e-7 for nondegenerate ridge fixtures;
original normal residual <=max(1e-7,10*inner_tol). Paired output relative
error <=2e-5, projector disagreement <=2e-5, identical outer step counts;
no new failures/rank losses. Preconditioning on scaled fixed-g inputs must
reduce median U passes and time by >=20%; isotropic default exact changes
must not regress median time >10% or grow peak allocations >10% + 1 MiB.
Full-fit projector quality difference <=2e-5 (numerical equivalence evidence,
not a new recovery claim). Record time, traced allocations/RSS, iterations,
U passes, diagnostics, objective and quality separately; failures retained.
Stop: failed math/reference gate stops candidate; failed selection gate saves
negative evidence and leaves held-out unused. No tuning on selection seeds.
