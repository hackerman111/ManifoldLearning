task_id: manifold-spec-grid-2026-09-28
status: done
change_class: experiment only; production estimator and solver unchanged
---

# Manifold grid from `Manifold exp.md`

Goal: implement a runnable experiment for the specified Gaussian curved-index
model and parameter grid, recording every fit and numerical failure.

Non-goals: estimator changes, comparison with other methods, launching the
5,500+ fit production grid, or changing the prior Spokoini implementation.

Known evidence: the current standalone `manifold_generalization` protocol uses
different cells and 5 seeds. The specified table has 26 rows; exact tuple
deduplication gives 22 unique configurations, not 23. The live manifold API
requires `d+1 < N_lin < n`, so fixed `N_lin=200` is invalid for several listed
cells (`n=100`, `n=200`, and `d=200`). The implementation uses the nearest
legal `N_lin` and preserves the specified `scale_boundary=raise`. The user's
full run reached 41/5500 fits before interruption; all 41 for the first cell
failed with `function mass target is infeasible even with alpha=0` after four
successful projector updates. This is the strict scale-boundary failure
recorded by the spec; `stop` is a separate diagnostic protocol.

Invariants: float64; `X~N(0,I_d)`; `u=XQ`; exact specified `z`, `f`, response
standardization, and additive noise; current `estimator="manifold"`,
`solver="cg"`, and `scale_boundary="raise"`; all fit errors retained; fixed
seed protocol; no comparison code or production estimator changes. The runner
represents configurations by exact unique data tuples and uses the nearest
legal `N_lin=max(d+2,min(200,n-1))` so every cell can run.

Read set: `Manifold exp.md`; `agent-notes/ADP/manifold.md`;
`experiments/manifold_generalization.py` (data, geometry, diagnostics);
`ADP/core/manifold/ADP_Manifold_utils.py`; `ADP/engine/manifol_engine/fit.py`;
`experiments/README.md`.

Work units:
- [x] Implement the exact unique grid, series selectors, seed profiles (30
  development / 250 full), fixed estimator configuration, and preflight view.
- [x] Record fit outcomes, failures, metrics, source/environment manifest, and
  incremental summaries without comparison logic.
- [x] Document commands and configuration caveats; update state/decision log.
- [x] Verify CLI preflight and source syntax/style; inspect the user's partial
  full-profile artifact without modifying it.

Stop conditions met: CLI reports exact selected cells and planned fits; the
manifest, per-fit JSONL, and incremental summaries are implemented. The user
launched the full profile and interrupted it at 41/5500; the partial artifact
is `benchmark_outputs/experiments/20260928T022740464081+0300-manifold-grid`.
It is incomplete and is not a full-grid result. Each failed seed remains a
valid recorded outcome under the specified strict boundary rule.
