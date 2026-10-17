# Plan: current Grassmann solver stage timing

- Task grassman-stage-profile-2026-10-06; status done.
- Goal: measure dominant internal stage of production grassman.solve/core_gn
  on existing heavy Spokoiny full fits, three seeds (0,1,2).
- Non-goals: optimize or change estimator, tolerances, algorithm or API.
- Class: observational profiling; preserve float64, same inputs/options,
  finite results and orthogonality. Profile only solver calls; preparation
  belongs to full fit and must be reported separately.
- Known evidence: current production core-only API; existing reproducible
  benchmarks.svd_grassman_optim_benchmark.fit and grassman_benchmark generator.
- Read set: AGENTS, workflow/research/numerics contracts, solvers notes,
  grassman.py:321-565, two benchmark modules.
- Units: Luna create isolated profiling harness; one small smoke; three
  sequential full-fit seeds with uninstrumented reference and instrumented
  twin; lead audit stage accounting, parity, raw artifacts; checkpoint.
- Evidence: reproducible command/protocol with code hashes, env/threads,
  exclusive stage times, wall time, call counts, diagnostics, quality and RSS.
- Additional verification: cProfile added 24-41% solver overhead; confirm
  stage ranking using three timer-only twins against existing baselines.
- Acceptance: no double counting, full stage coverage with explicit remainder,
  same-seed output/diagnostic parity and measured profiling overhead.
- Stop conditions: nonfinite/parity failure; 900s per run; keep failures;
  no broad benchmark expansion or performance changes.

- Verified: three baseline/cProfile/timer-only pairs; exact input/basis/full
  diagnostics parity; exclusive accounting matches wall (lead_audit.json).
  Main stage local refit47.3% pooled,33.3–57.8% per seed. Full protocol/raw
  profiles/report: experiments/grassman_stage_profile_2026_10_06/.
  Failed instrumentation runs archived; production source unchanged.
