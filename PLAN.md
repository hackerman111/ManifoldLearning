---
task_id: hybrid-package-decomposition-2026-09-27
status: done
change_class: EXACT
---

# HYBRID solver package decomposition

## Goal and scope
Keep the original HYBRID implementation and move it into
`ADP/solver/HYBRID/`: shared core, explicit single/multi routes, and separate
manifold solver. Remove the parallel experimental solver and its integrations
and artifacts. Explain current and legacy LSMR contracts.

## Invariants and known evidence
Current HYBRID HPAO is multi-index-only and delegates outer iterations to
`LSMR.solve`; manifold is an independent linear subproblem. Single-index uses
current HPAO-LSMR. Preserve objectives, tolerances, certificates, diagnostics,
shapes, existing import paths, and original HYBRID's own workspace recycling.

## Read set
`agent-notes/ADP/solvers.md`, `ADP/solver/HYBRID/`,
`ADP/solver/LSMR.py`, `ADP/solver/legacy_lsmr.py`, import callers,
`tests/test_hybrid*.py`, `ADP/cli/main.py`, `pyproject.toml`, and
`agent-notes/WORKFLOW.md`.

## Bounded work and acceptance
- [x] Remove only parallel-variant additions while preserving unrelated dirty work.
- [x] Split shared linear algebra, multi-index workspace/entry point, and
  manifold solve; add a single-index route that explicitly reuses current LSMR.
- [x] Preserve `ADP.solver.HYBRID` public symbols and update internal imports,
  focused monkeypatch locations, and solver routing notes.
- [x] Confirm no parallel-variant traces remain; verify syntax/imports and
  `git diff --check`. Do not run pytest unless explicitly requested.
- [x] Update `STATE.md`, append the architecture decision, mark plan done.

## Stop conditions
Stop if extraction needs a formula, tolerance, result-contract, or default
change. Resolve such changes as a separate task.
