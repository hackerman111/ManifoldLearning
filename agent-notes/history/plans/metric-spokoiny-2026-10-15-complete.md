# Plan: Metric low-rank descent on hard Spokoiny

- Task `metric-spokoiny-2026-10-15`; status **done**.
- Goal: determine whether the existing profiled metric descent improves hard
  Spokoiny recovery, solver convergence and total fit cost. GPT-6 Luna owns
  bounded harness/tests/runs; lead audits mathematics, protocol and conclusions.
- Non-goals: new optimizer, production/default changes, estimator tuning,
  replacing the benchmark, CUDA, broad hyperparameter search, catalog repair.
- Evidence: preceding three-seed weak-coefficient synthetic pilot was positive;
  ordinary synthetic calibration was slower. Neither establishes Spokoiny gain.
  Archive: history/plans/edr-structured-2026-10-15-complete.md.
- Hypotheses: coefficient metric helps unequal local slopes; rank-one truncation
  can lose weak directions; poor local ranks/outer statistics may dominate.
- Class: existing APPROXIMATE optimizer of identical frozen finite-sketch Phi;
  experiment-only controls, no ESTIMATOR change. tau=0 fixed in all arms.
- Invariants: float64, orthonormal row basis; fixed mass/weights/sketch per
  frozen comparison; full-gradient certificate normalized by sum mass*I^2;
  identical initial/data hashes; objective acceptance and rank guards retained;
  no dense ambient d*d numerical allocations or inverse.
- Exact read set: AGENTS/RTK, WORKFLOW, research/numerics/engineering contracts,
  solvers.md; benchmark grassman_benchmark.py::_make_problem and PARAMS;
  cited generator/config sources; grassman_research metric branch,
  _grassman_schur.metric_direction; prior PROOF/tests/report; pyproject.toml.

## Bounded work units

- [done] Verify exact hard benchmark and metric algebra/control semantics;
  freeze protocol in experiments/metric_spokoiny_2026_10_15/ before timings.
  Benchmark expected n800,d50,m2,J800,P10, original noise/schedule/init.
- [done] Luna harness: serial fresh workers, 1 BLAS/OMP/MKL thread, alternating
  arms, wall/RSS/stop/quality/inner summaries; separate snapshot capture and
  instrumentation from ordinary fit timings. Test independent metric reference,
  gauge equivariance, rank control and certificate. Lead reviews before runs.
- [done] Development seeds0/1/2: full fits core_gn, metric rank2, metric rank1
  at original five inner steps (9 fits). Separate core-path captures early,
  middle, late outer states; frozen solve comparisons at budgets5 and100
  for core/metric2/metric1/isotropic control. No retuning from results.
- [done] Analyze coefficient spectrum, full gradients, local rank failures,
  per-principal-angle recovery, objective-vs-truth disconnect. A fixed extra
  budget20 comparison core/metric2 on seeds0/1/2 (6 fits) checks budget effect.
- [done: skipped by failed gate] Held-out seeds3..7 core/metric2 at budget5 only if development gate:
  no per-seed projector regression >1e-4 and either >=10% paired median fit
  speedup or >=20% median projector reduction at <=1.5x median time. If gate
  fails retain validation untouched. No automatic promotion either way.
- [done] Lead audit raw pairing/hashes/aggregation; Luna focused tests, scoped
  lint/type; saved report and plots; update PLAN/STATE/DECISIONS/solver route.

Evidence: results_repaired/ has9 fullfit5,4 capture,90 frozen and6 budget20
successful rows; original90 pre-numerical launcher errors retained separately.
Development gate fails (seed1 regression5.666e-4); validation3..7 untouched.
Five jointly smooth-certified frozen pairs: metric/core median wall2.040x.
All9 sampled initial metric/isotropic proposals are below the angle cap;
proposal_audit.json verifies common-mass scaling and empirical R spectra.
Two supplementary untimed profiles are saved. LEAD_AUDIT.md and
results_repaired/lead_audit.json verify18 source hashes,14 imported files,
pairing/unique jobs/threads/monotone objectives and independent failed gate.
Final10 tests pass; scoped Ruff/format/Pyright clean. REPORT/plot verified,
MATH/DECISIONS/STATE/solvers route updated. No numerical/default changes.

## Verification and stops

Protocol must identify exact link, sampling/noise, kernel argument, bandwidth,
anisotropy, centers/directions/refresh/init, schedule, tolerances, seed streams,
code/environment hashes and timing boundaries. Count all failures/timeouts.
Timeout 180s per full fit, 60s per frozen arm; serial timed work, total bounded
stages above. Stop/revise on wrong generator, reference/certificate mismatch,
nonfinite/orthogonality failure, overwrite or uncontrolled threads. No speed
claim from uncertified frozen endpoints; outer h_min is not convergence.
Controls must disclose metric-floor step-scale confounding; never infer that
faster decreasing Phi implies better truth recovery. No unrelated full-suite
rerun or repairs: preceding task recorded existing collection/manifold failures.
