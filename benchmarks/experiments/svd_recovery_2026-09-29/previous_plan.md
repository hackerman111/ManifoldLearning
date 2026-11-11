task_id: lsmr-vs-current-svd-full-fit-2026-09-29
status: active
change_class: EXPERIMENT; solver objectives differ, no estimator/default changes

# Compare current LSMR and SVD on paired full multi-index fits

Goal: measure full-fit runtime, process peak RSS, recovery error, and solver/
outer-loop diagnostics for LSMR versus current SVD on three frozen workloads:
small n=500,d=20,m=3 (100 paired seeds), medium n=900,d=60,m=3 (50 pairs),
and the saved heavy Spokoiny point n=800,d=50,m=2,N_J=800 (25 pairs).

Non-goals: change either solver, tune after inspecting results, or claim their
inner objectives are equivalent. SVD's fixed-g rank-r objective and LSMR's
HPAO objective differ; projector recovery is an end-to-end descriptive metric.

Known evidence: prior three-seed SVD/HYBRID full fits exist for the small and
medium synthetic workloads and the heavy Spokoiny point. On the heavy point,
SVD took median 47.56 s and HYBRID 58.99 s; this does not predict current LSMR.
The heavy historical saved experiment uses LSMR but is from a different run
state and will not substitute for paired current runs.

Invariants: paired methods receive identical generated X/y/truth and fit seed;
float64; one BLAS/OpenMP thread; isolated fresh worker process per fit; rank=2
for m=3 and rank=1 for m=2; no fit-time tracing; capture process high-water
RSS. Keep every failure and incomplete pair. Compare per-cell distributions,
paired differences/ratios, diagnostics, and descriptive projector error.

Exact read set: `benchmarks/svd_vs_hybrid.py`, `ADP/solver/{LSMR,SVD}.py`,
`agent-notes/ADP/multi-index.md`, `experiments/{data.py,models.py,runner.py}`,
the prior small/medium report, and saved heavy point `series.json`.

Bounded work units:
- [x] Add an LSMR/SVD paired-run protocol for these cases and seed counts;
  validate solver wiring on small seed 0 and the Spokoiny LSMR generator path
  on seed 25 (outside the frozen analysis range). Both workers completed and
  emitted finite full-fit metrics.
- [x] Run frozen seed ranges: small 0-99, medium 0-49, heavy 0-24. Do not
  retune or replace failed seeds; retain all rows.
- [x] Verify raw records and aggregates; write per-case report with time,
  RSS, projector recovery, convergence/failure counts, and paired statistics.
  Update `PLAN.md`/`agent-notes/STATE.md`; run Ruff and `git diff --check`.

Stop conditions: any data/seed mismatch, duplicate/missing pair, changed
parameters between methods, or repeated worker failures; preserve failed rows
and report incomplete evidence. Safety timeout is 240 seconds per full fit.

Outcome: 350/350 fits succeeded with exact paired seed coverage and no
duplicates. SVD was faster on every pair (median speedup 8.6x small, 7.9x
medium, 1.51x large). Median projector error favored SVD slightly on small,
was inconclusive on medium, and strongly favored LSMR on the large workload,
where several SVD seeds had high error. Peak RSS was consistently higher for
SVD by about 3.0, 5.2, and 2.0 MiB, respectively. No LSMR run met its HPAO
stopping certificate; SVD inner solves converged in 82/100, 36/50, and 7/25
fits. See `experiments/lsmr_vs_svd_fullfit_2026-09-29/REPORT.md` for full
statistics and the objective/convergence caveats. Ruff, format, and diff
checks passed; no tests were requested or run.

status: done
