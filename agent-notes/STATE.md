# Current agent state

Status: grid implementation is complete. The isolated runner is
`experiments/manifold_grid.py`; it reuses the existing exact geometry/data
generator and `DiagnosticManifold` hooks without changing production estimator
code. It exposes eight series, development/full repetition profiles, series
selection, dry-run planning, and incremental manifest/runs/summary output.

Two inconsistencies in the specification are resolved explicitly in the
runner and documented in `experiments/README.md`:

- Its 26 listed rows deduplicate to 22 exact configurations (four overlaps),
  not the stated 23.
- Fixed `N_lin=200` fails current live validation for `n=100`, `n=200`, and
  `d=200`. The runner uses nearest legal target mass
  `max(d+2, min(200, n-1))`, recorded per fit and in the manifest.

No comparison code or production estimator changes are included. The previous
Spokoini task had an unfinished 5,450-fit run; its plan was archived at
`agent-notes/history/plans/spokoini-grid-2026-09-28-interrupted-before-full-run.md`
and should not be mistaken for executed results.

Verification: full-profile dry-run reports 22 unique configurations / 5,500
fits; a selected-series dry-run reports seven deduplicated points / 1,750
fits. `ruff check`, `py_compile`, and `git diff --check` pass. No full fit was
launched by the implementation session. The user later started the full grid
and interrupted at 41/5,500 fits; see
`benchmark_outputs/experiments/20260928T022740464081+0300-manifold-grid/`.
All 41 completed fits are recorded as errors with `function mass target is
infeasible even with alpha=0`; diagnostics show four successful projector
updates before the function-mass boundary was crossed. This follows the spec's
fixed `scale_boundary=raise`; the separate `stop` behavior is only for a
diagnostic variant. Preserve these rows and the strict setting when interpreting
the specified grid. The current runner does not resume a partial artifact.

The earlier Spokoini grid's 5,450-fit run is still pending.
