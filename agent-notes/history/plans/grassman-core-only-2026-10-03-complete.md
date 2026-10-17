# Plan: core-only Grassmann production API

- Task grassman-core-only-2026-10-03; status done.
- Goal: grassman.solve exposes only verified core_gn; all selectable alternative
  algorithms/Schur-refit ablations live in grassman_exp.
- Non-goals: change profile/estimator/budget/rank/memory safety or rerun selection.
- Class EXACT API separation, user-authorized signature change. Preserve float64,
  profile minimum norm, guards, core default outputs and diagnostics. Internal
  rank-one/spectral fallback steps remain part of validated core algorithm.
- Evidence/read set: prior promotion REPORT; solver solve ranges and shared
  kernels; four focused tests; benchmark/capture callers; solvers/multi-index notes.
- Units: remove production method/angle_backend selectors and refit ablation;
  migrate callers/tests; verify core parity incl guards, alternatives in exp;
  refresh routing notes/state/decision.
- Acceptance: focused tests, exact core parity vs pre-change, Ruff/Pyright,
  benchmark routing checks. Stop on same-input core output regression.

- Verified: 51 focused tests; 72 exact core pairs incl rank/workspace guards
  (experiments/grassman_promotion_2026_10_03/core_only_parity.json);
  benchmark construction/reference routing; Ruff/format and Pyright pass.
