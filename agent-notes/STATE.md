# Current state

Task grassman-core-only-2026-10-03 done. User clarified that production
must expose only the verified core_gn algorithm.

- grassman.solve now has no method or angle_backend parameters; it always
  executes core_gn. Rank-one/spectral safety fallbacks remain internal parts
  of the validated core algorithm. Objective/budget/guards unchanged.
- All selectable rank_one/spectral/Schur-refit ablations, flat_gn and
  solve_full_gn live in grassman_exp; numerical kernels imported from grassman.
- Test and benchmark callers migrated. Archived original reference preserved;
  historical raw experiment rows unchanged. Prior promotion plan archived at
  agent-notes/history/plans/grassman-promotion-2026-10-03-complete.md.
- 51 focused tests; 72 exact core pairs including regular, rank-deficient,
  workspace-limited cases at d20/100/300/600,m1/2/4,ridge0/0.2; benchmark
  routing/construction checks; Ruff/format and Pyright0errors0warnings pass.
- Evidence: experiments/grassman_promotion_2026_10_03/core_only_parity.json;
  REPORT.md contains initial promotion and this correction. Previous120 relocation
  pairs describe initial promotion, before production API restriction.
- Source routes: grassman.py:1-565; grassman_exp.py:30-458 flat/orchestration,
  :459-704 full tangent. Notes solvers/multi-index refreshed.
- No new estimator/performance/recovery claims or heldout runs. User dirty work
  preserved. No blocker or pending implementation.
