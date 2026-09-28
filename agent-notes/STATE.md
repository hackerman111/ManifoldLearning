# Current agent state

Completed task `lsmr-vs-current-svd-full-fit-2026-09-29`; details and
acceptance evidence are in `PLAN.md`. The paired benchmark contains 350/350
successful fits: 100 small pairs (n=500,d=20,m=3), 50 medium (n=900,d=60,m=3),
and 25 large saved Spokoiny pairs (n=800,d=50,m=2,N_J=800). Frozen seeds are
0-based and paired exactly. Full distributions, paired bootstrap intervals,
solver diagnostics, and caveats are in
`experiments/lsmr_vs_svd_fullfit_2026-09-29/REPORT.md`; raw records and
machine-readable aggregates are alongside it.

SVD was faster on every pair: median speedups were 8.6x, 7.9x, and 1.51x from
small to large. Projector error favored SVD slightly for small, had no clear
paired difference for medium, and favored LSMR strongly for large; SVD had a
long high-error tail in the large case. SVD process peak RSS was higher by
about 3.0, 5.2, and 2.0 MiB. None of the LSMR fits met the current HPAO
stopping certificate. SVD inner convergence counts were 82/100, 36/50, and
7/25. These are end-to-end descriptive comparisons: objectives and convergence
diagnostics differ, and SVD completes its remaining basis direction from the
prior.

The harness is `benchmarks/svd_vs_hybrid.py` (`--comparison lsmr-svd`); report
generation is `benchmarks/analyze_lsmr_svd.py`. Ruff, format, and `git diff
--check` passed. No tests were requested or run. Existing independent dirty
changes in the worktree were preserved.
