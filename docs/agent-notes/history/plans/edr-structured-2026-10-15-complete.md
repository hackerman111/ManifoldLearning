# Plan: EDR structured solver candidates

- Task `edr-structured-2026-10-15`; status **done**.
- Goal: implement/evaluate Schur profile/preconditioning and exact U=CX from
  `SVD/edr_research_note.tex`; also check recycling, metric low-rank descent,
  exact VARPRO Newton/trust region. Lead owns numerical code; GPT-6 Luna owns
  independent tests and reproducible benchmarks.
- Non-goals: default/CUDA/legacy changes, weights/sketch/bandwidth changes,
  recovery guarantees, broad parameter search, dense d*d matrices.
- Known evidence: current core_gn caches UY/UV; local profile/refit dominates
  recorded d50 fits. Existing metric SVD has a different fixed-g objective.
- Classes: CX EXACT/NUMERICAL, Schur NUMERICAL; bounded optimizers/recycling/
  preconditioning APPROXIMATE opt-in for the same frozen objective. Positive
  local_ridge is an explicit ESTIMATOR setting, held fixed in each pair.
- Invariants: float64, row basis (m,d), orthonormality; finite-sketch
  sum mass*min_g(||I-UYg||²+tau||g||²); no double mass in legacy normalization;
  polar ridge metric; true-profile monotone acceptance; full live gradient
  for convergence; tau=0 rank guards; no explicit inverse.
- Read set: AGENTS/RTK, numerics/research/engineering contracts, WORKFLOW;
  solvers.md; note:190-446; grassman.py:23-200,268-565;
  statistic.py:29-213; utils.py data/weight validators; LSMR input/result
  helpers; tests/test_grassman.py; pyproject.toml; new files below.

## Work units and acceptance

1. [done] Freeze proof/interfaces under experiments/edr_structured_2026_10_15/;
   Luna writes independent failing reference tests before implementation.
2. [done] ADP/solver/statistic_operator.py: DenseOperator, FactorizedOperator,
   streamed CSR builder. Check dense moments both normalizations, adjoints,
   sparse/offset/invalid inputs; measure construction and storage as well.
3. [done] ADP/solver/_grassman_schur.py: full active-block Schur, analytic
   gradient/HVP, stable inactive elimination, metric direction, preconditioned
   Steihaug trust step. Check augmented-lstsq reference, finite differences,
   Hessian symmetry/negative curvature, tau=0 guards.
4. [done] ADP/solver/grassman_research.py: opt-in schur_gn/schur_newton/metric,
   recycling/block preconditioner. Check monotonicity, orthogonality, operator
   parity, full-gradient certification, budget/failure diagnostics, API smoke.
5. [done] Luna paired frozen benchmarks: isotropic/weak-index/correlated,
   seeds0/1/2; sparse/dense CX setup+solve; actual ADP statistics and bounded
   full-fit smoke. Same tau/tol/budget/thread, wall/RSS/iterations/operator
   actions/objective/gradient/error; retain every failure. Same-certificate
   selection requires >=10% median speed gain without quality regression.
   Otherwise retain explicit experiments; no default promotion.
6. [done] Lead audit; Luna focused/full pytest, scoped Ruff/Pyright; report;
   update routing, PLAN/STATE/DECISIONS.

Stop/revise on identity/adjoint/derivative disagreement, nonfinite results,
workspace violation. Tau=0 nonsmooth Newton rejects/falls back, never changes
ridge silently. Benchmark cap120s/arm; small kernels 3 repetitions, one BLAS
thread, no concurrent timed work. Failed gates stay in report.
Previous completed plan archived under agent-notes/history/plans/.

## Acceptance evidence

- `experiments/edr_structured_2026_10_15/{PROOF,REPORT,LEAD_AUDIT,README}.md`,
  protocol.json and raw results preserve identities, paired settings, timing
  boundaries, hashes, failed attempts, and usage. No default promotion.
- Focused new + existing Grassmann suite: 73 passed. Scoped numerical and
  harness Ruff format/check and Pyright pass. Review guard regression RED ->
  GREEN; final full-gradient certificate and public adapter smoke verified.
- Full suite has three catalog collection errors; excluding exactly these
  modules: 479 passed, 7 failed, 28 skipped. All seven failures reproduce in
  isolation; implicated existing sources unchanged. Exact names/logs under
  test_results/full-suite-status.md in the experiment folder.
- Distinct-K Schur kernel about 4.5x faster, excluding ~.76ms setup; whole
  solver gain absent. CX support8 storage .625x dense, setup+solve tied;
  support400 storage6.505x and setup+solve about2x slower.
- Recycling full adjoints31->23; CG preconditioning iterations789->763;
  neither improves wall time. Near-truth calibration: all methods certify,
  new methods slower. Targeted weak-signal pilot: metric3/3 versus core2/3,
  1.662x paired speedup on two certified pairs, lower projector error.
- Supplemental pilot has only three seeds. Full-fit smoke is budget-limited;
  positive-tau performance and CUDA were not measured. Positive-tau algebra
  is covered by reference/derivative tests. Task complete with these limits.
