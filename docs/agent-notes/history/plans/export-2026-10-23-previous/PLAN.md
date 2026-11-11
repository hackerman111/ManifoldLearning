# Joint low-rank correction accuracy search

Task `grassman-lowrank-2026-10-22`; status **done: bounded experiments; accuracy goal unachieved**.
No production promotion; fresh validation was gated off.

Goal: user-requested LoRA-like additive correction and joint rank-k SVD
correction, plus an independent residual-enrichment idea, to reduce full-ADP
projector recovery error with <=10% median fit-time cost versus live combined20.
Compare to actual R MEANMAVE. Luna owns routine runs/aggregation, lead owns math.
Non-goals: changing localization/sketch/estimator, GPU implementation, unrelated
repairs, promotion from inner objective or a single seed. Production unchanged
until paired development plus untouched validation pass.

Known evidence: previous nine candidates failed the bounded pilot; archived
plan `agent-notes/history/plans/grassman-accuracy22-screening.md`, report
`experiments/grassman_accuracy_2026_10_22/REPORT.md`. Existing SVD correction
is fixed-g greedy rank-one; existing Grassmann block_rank_k is a different
profiled angle update and has negative prior full-fit evidence. Neither tests
joint additive-factor refinement. Reference.py still equals live grassman.py.

Hypotheses (APPROXIMATE optimizer variants, same finite-sketch target):
L: optimize Delta=A C, rank<=k, both factors jointly via alternating LS;
S: choose k horizontal gradient SVD directions, jointly refit all k*m
correction coordinates (not sequential rank-one scale-only refits);
E: enlarge that coarse space using the remaining frozen-g normal residual.
Freeze current local coefficients only to propose steps. Polar retraction,
all-center live profile/gradient and true Armijo acceptance are mandatory.
Zero local ridge for these experiments; fixed-g surrogate is not substituted
for the profiled objective or used as a convergence certificate.

Invariants: float64, orthonormal output, full all-center final certificate,
finite/rank guards, bounded scratch (16MiB small LS; 4MiB adjoint blocks),
no dense d*d/Hessian/full ambient Jacobian, no U-sized extra copy, unchanged
public API/default. Capped LSMR is a proposal, never claimed converged.

Read set: AGENTS/STATE/WORKFLOW/numerics/research; ADP notes solvers and
multi-index additive/fixed-g sections; SVD/SVD_solver.tex680-774; grassman.py profile/gradient/
solve405-810; SVD.py correction301-690; existing accuracy22 reference,
candidates, benchmark; tests/test_svd_solver.py and accuracy22 tests.

- [x] A derive shapes/majorization, implement isolated lowrank.py and independent
  dense/adjoint/rank/monotone/stress tests; preserve reference and old artifacts.
- [x] B registered candidates and union full+multi-rank case catalogs in existing
  frozen-source R harness. Luna run exploratory seed90800 on d5/20/50,corr.9,
  rank3/4/5, baseline+actual R; compare k1/kfull and budget10/20 as bounded arms.
  77/77 low-rank fits +28/28 projected-Jacobian fits completed, all workers ok.
  Seven paired cases, exploratory90800 only. LoRA_k10 EDRratio.990/time1.017;
  projected10 .9859/.723, projected20 1.0085/.985. Other low-rank arms fail.
  LOWRANK_REPORT.md/raw grassman-lowrank22-{pilot,projected-pilot} preserve data.
  Added own second mechanism P: omit orthogonal-column-space residual part
  of the profile Jacobian. J_perp.T r is unchanged; smaller PSD curvature
  may avoid conservative GN steps. Independent QR/gradient/PSD checks first.
- [x] C FROZEN before new seeds: projected10 (only J_perp, max_steps10,
  other combined20 settings identical). Development111800..111809 on same
  seven cases, baseline20 + budget10 (identical production except max_steps10)
  +projected10+actual R, 280fits. Isolate iteration budget in interpretation.
  Development280/280 completed,70 paired datasets, zero worker failures.
  P10 EDR/base1.00065 [0.98131,1.02227], median/p90 fit .737/.885;
  budget10 .99895 [0.97675,1.02411], .742/.941. Both fail quality and
  case/individual regression gates. P10/budget10 EDR1.00169,
  CI[.98859,1.01678], time .990. No demonstrated direction benefit.
  Validation112800..112809 NOT RUN: failed development gate, no retuning.
  Bootstrap fixed before dev: 10000 seed-block resamples retaining all seven
  cases, RNG20261022, two-sided percentile95 (upper97.5%).
  Gates: EDRgeo<=.98/bootstrap95upper<1; median time<=1.10,p90<=1.25;
  RSS<=1.10; finite orthogonality<=1e-10; no case median EDR increase>.005
  or individual increase>.05, no failures. No validation tuning.
- [x] D update report/math/routing/DECISIONS/STATE;214 focused tests passed,
  Ruff/Pyright clean, git diff --check clean. Reference==production bytewise.
  LOWRANK_REPORT.md and per-run paired_analysis.json preserve385 fits over
  77 unique datasets (7 pilot+70 dev), actual R controls, failures/gates.

Stop: reject incorrect/unstable/budget-breaking algorithms before benchmarks;
Stop reached: no promotion after failed development gate. Negative screening is a
completed experiment, not the unachieved recovery improvement. Preserve all
failures and exact seeds/inputs/source/diagnostics in artifacts.
