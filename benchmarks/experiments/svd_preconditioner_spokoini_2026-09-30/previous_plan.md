task_id: svd-simple-improvements-2026-09-30-r2
status: done
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
rank(B-P) are different constraints. The bounded selection/validation below
supports preconditioning on the tested column-scaled inputs.
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
- [x] R3a: fixed-g selection seeds 110..112, 108 rows passed. Initial full-fit
  fixture invalid at d150: N_lin=300 > n220; all 18 methods refused before
  solver execution, preserved in initial raw rows (selection_summary failed).
- [x] R3b: corrected full-fit selection n320, d50/150, fresh seeds 120..122;
  36 rows, unchanged algorithm/thresholds. Protocol repair, no solver tuning.
- [x] R4: untouched fixed-g seeds 210..212 and fit seeds 220..222 (n320),
  only after corrected R3 passes; same other fixtures/gates.
- [x] R5: integrated opt-in preconditioner; 59 focused/shared tests passed,
  Ruff/format/diff checks passed. Pyright timed out after 180s (exit 124);
  full type checking remains unverified. Report, routes, decisions and final
  state recorded in experiments/svd_improvements_2026-09-30/REPORT.md and
  VERIFICATION.json. Live solver equals the validated candidate byte-for-byte.

Final evidence: corrected selection and untouched validation each passed all
gates on 144 rows. Validation scaled fixed-g median time/pass ratios are
0.2255/0.2381; default isotropic time ratio 0.9430. Full-fit quality change
<=2.27e-8, identical outer step counts, no new numerical failures.

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
