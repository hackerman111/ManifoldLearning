task_id: svd-vs-hybrid-heavy-30s-2026-09-28
status: done
change_class: APPROXIMATE comparison; no estimator changes

# Compare SVD with HYBRID on the 30-second heavy multi-index point

Goal: rerun the prior `mi-spokoini-m2-dhigh` point with paired seeds 0,1,2
using the full SVD rank-1 and HYBRID fit loops. Report wall time against the
30-second target, recovery quality, process peak RSS, traced Python peak,
outer/inner convergence and failures.

Non-goals: stop a fit at 30 seconds (full-fit quality and memory still matter),
change parameters after seeing results, or claim equivalent inner objectives.

Frozen point: Spokoini m=2, d=50, n=800, tau=1, beta(1,tau) features, Gaussian
noise sigma_eps=0.1, `N_loc=10`, `N_lin=100`, `N_J=800`, `N_phi=10`,
`a=exp(1/100)`, `h_min=1`, local initialization, `select_step=last`, no outer
step cap, solver max steps 5. Recreate data and model seeds with the same
`experiments.data._make_seed_bundle` scheme as the saved experiment. SVD uses
rank r=1; HYBRID uses its saved strict defaults. Both use float64 and one BLAS
thread, with a safety timeout of 240 seconds; 30 seconds is a target checked
after each complete fit, not a censoring deadline.

Known limitation: SVD's fixed-g rank-r objective differs from HYBRID's HPAO
correction-penalty objective, despite the shared numeric lambda. Its second
basis direction is completed from the incoming prior. Compare end-to-end
recovery descriptively and include this caveat.

Exact read set: saved point in
`benchmark_outputs/experiments/spokoyni-12/mi-spokoini-m2-dhigh/series.json`,
`experiments/{multi.py,data.py,runner.py,models.py}`, current SVD/HYBRID solver
APIs, and the existing paired runner `benchmarks/svd_vs_hybrid.py`.

Work units:
- [x] Extend runner for exact Spokoiny point and matched seed bundle; run 6
  fits in fresh processes.
- [x] Verify all raw records, compare paired metrics and 30-second target,
  report time/quality/memory/convergence/failures.
- [x] Save protocol/report, update plan and `agent-notes/STATE.md`, run Ruff and
  `git diff --check`.

Evidence: all 6 fits completed with no errors or safety timeouts; 0/3 fits per
method met the 30-second target. Median SVD/HYBRID times were 47.56/58.99 s,
median normalized projector distances 0.0951/0.0481, and median process peak
RSS 98,172/125,848 KiB. All fits reached `h_min` after 162 outer steps;
HYBRID's convergence diagnostic was false on all runs, and SVD's rank-one
inner loop hit its 20-step cap on 22/49/43 outer updates. Full paired results,
protocol, raw records, and traces are in
`experiments/svd_vs_hybrid_spokoini_dhigh_30s_untraced_2026-09-28/`.
`uv run ruff format --check benchmarks/svd_vs_hybrid.py`,
`uv run ruff check benchmarks/svd_vs_hybrid.py`, and `rtk git diff --check`
passed. Traced Python peak was intentionally not collected for this timing
target; process high-water RSS is reported instead.

Stop conditions: preserve and report any failure/timeout; do not resample or
retune. If either fit is censored by the 240-second safety timeout, report the
comparison as incomplete for that pair.
