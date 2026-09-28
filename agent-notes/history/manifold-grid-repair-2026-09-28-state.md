# Current agent state

Completed task: `manifold-grid-repair-2026-09-28` in `PLAN.md`. The prior
`manifold-generalization-repair-2026-09-28` work remains pending, archived
in `agent-notes/history/plans/manifold-generalization-repair-2026-09-28-pending.md`.
Its dirty code, notes, paired selection, and flat validation artifacts are
untouched. Do not infer that its final validation or handoff is complete.

Two interrupted `experiments.manifold_grid --profile full` outputs have 41
and 20 rows, all the same first cell `(200,3,1,.35,.1)` and all the same
function-mass infeasibility error. Its first row reached four successful
projector updates with solver residual around `1.35e-14`; the exception is
at the next scale with strict `scale_boundary="raise"`, not an initial fit
or solver failure. `scale_boundary="stop"` is an existing explicit estimator
variant that locates the feasible scale boundary.

The old grid used a raw nearest-chart query recovery gate. Prior analysis in
`docs/experiments/manifold_generalization_2026-09-28/repair.md` shows curved
`m>=2` generator charts are unidentifiable from scalar Y and raw query
error includes nearest-chart discretization. The repaired grid uses the
corrected assessable target and records all query errors separately.

Reference on seed 81000, `(n,d,m,c,sigma)=(200,3,1,.35,.1)` with one BLAS
thread: strict raises after 4 successful updates. Existing `stop` mode
returns `stop_reason=function_mass_boundary` after 5 updates and 2 scales;
final mass is 80.0000000002 and residual 9.1e-15. Center RMS/max principal
sine is 0.330/0.642 and same-center chart RMS/max is 0.311/0.642, so this
fit is numerically valid but does not meet recovery criterion.

Grid implementation now defaults to explicit `scale_boundary="stop"` and
offers `--scale-boundary raise` for the original strict protocol. Schema 2
records corrected center/chart-estimation/raw-query/oracle-query metrics;
`recovered` applies only to `m=1` or flat `c=0`. The first dimension-series
CLI check completed 4/4 fits with no errors but 0/4 recovery. Focused
`tests/test_manifold_grid.py`, Ruff, Pyright, and `git diff --check` pass.
The production estimator/default and old grid artifacts are untouched.

All 22 unique cells were sampled. After deduplicating one shared case-seed,
there are 82 attempts, 61 completed fits and 21 rank failures. All 61
completed fits stopped at the function-mass boundary. Of 40 identifiable
attempts, two recovered, both in the flat `(400,10,2,0,.1)` cell; curved
`m=1` recovered 0/35 attempts. Non-HD cells have five development seeds;
the remaining HD cells have one seed. The full 5500-fit run and held-out
validation were not done. Exact commands, environment, failure classes,
timings and limits are in
`docs/experiments/manifold_grid_2026-09-28/report.md` and three linked
artifacts. Full/default and strict dry-runs, focused pytest, Ruff, Pyright
and `git diff --check` pass. The next agent may resume the archived prior
manifold-generalization task or run the full grid without changing this
checkpoint.
