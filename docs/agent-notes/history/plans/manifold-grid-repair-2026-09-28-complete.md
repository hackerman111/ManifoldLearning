task_id: manifold-grid-repair-2026-09-28
status: done
change_class: ESTIMATOR option selection and experiment metric/protocol correction
---

# Repair the manifold experiment grid

Goal: make the supplied `--profile full` command use a feasible, explicit
boundary policy and report whether manifold fits recover an identifiable
local subspace. Keep each numerical failure visible.

Non-goals: change production manifold defaults or solver certificates, run
all 5500 full fits during debugging, rewrite old result artifacts, or claim
universal recovery from a scalar response.

Known evidence: two interrupted strict grid artifacts contain 61/61 identical
`function mass target is infeasible even with alpha=0` errors on the first
`n=200,d=3,m=1` cell. The first failed row has four successful projector
updates and a small solver residual. The live `scale_boundary="stop"` mode
locates this feasibility boundary. Earlier generalization analysis proves the
chosen generator chart is not identifiable for curved `m>=2`, and the raw
nearest-chart query gate includes discretization error.

Hypotheses: explicit boundary stop lets the first cell return an assessable
fit; a corrected chart-estimation gate gives a valid empirical recovery
measure on identifiable profiles. Rank and other failures may remain.

Invariants: fixed 22 deduplicated cells, data and seed scheme, float64,
original estimator/default in production, rank/residual checks, bounded
memory, all failed rows and stop reasons, old artifacts untouched. Preserve
strict `raise` as an explicit grid option. Keep `recovered=null` for curved
`m>=2` while retaining descriptive distances.

Read set: `agent-notes/ADP/manifold.md`; research/numerics contracts;
`Manifold exp.md`; `experiments/{manifold_grid,manifold_generalization,
manifold_generalization_repair}.py`; `ADP/engine/manifol_engine/{fit,weights,
utils}.py`; relevant `tests/test_manifold.py`; `experiments/README.md`.

Work units:
- [x] Run a one-seed reference with `scale_boundary="stop"` and compare the
  strict failure, update count, trace, and mass-boundary status. Seed 81000:
  strict raises after 4 updates; stop returns after 5 with mass 80.0000000002,
  residual 9.1e-15, center RMS/max 0.330/0.642; fit does not recover.
- [x] Make the grid boundary policy explicit; correct recovery/diagnostic
  metrics, manifest, and summary with a focused check. Four dimension fits
  complete with zero errors; a one-test integrated check passes, including
  strict failure and `recovered=null` on curved m=2. Ruff, Pyright and
  `git diff --check` pass on the touched code.
- [x] Run a bounded multi-seed grid across representative identifiable and
  difficult cells; report fits, recovery, numerical failures, and limits.
  All 22 cells sampled: 82 unique cell-seed attempts, 61 fits, 21 rank
  failures, 40 identifiable attempts, two recoveries (flat m=2). All
  successful fits stopped at the function-mass boundary. HD-only cells have
  one seed each; full 250-seed profile remains unrun.
- [x] Refresh affected routing/docs and checkpoint `STATE.md` and decisions.
  See `docs/experiments/manifold_grid_2026-09-28/report.md`; focused pytest,
  Ruff, Pyright, strict/default full dry-runs, and `git diff --check` pass.

Verification: no first-cell mass exception under default command; strict
mode remains reproducible; `recovered` only applies to identifiable cells;
fit and metric failures remain recorded. Run focused tests, lint and type
checks on touched code, then `git diff --check`. Stop if the boundary mode
still fails before an evaluable fit or if checks show an estimator change
outside the explicit grid option.
