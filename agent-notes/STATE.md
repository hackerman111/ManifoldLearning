# Current agent state

## Task

- ID: diagnostic-adp-benchmarks-2026-09-23
- Status: done; D1–D3 complete.
- Goal: парные диагностические бенчмарки single/multi/manifold с отдельными seed для выбора и проверки параметров.
- Plan: `PLAN.md`; previous completed CPU optimization plan archived at `agent-notes/history/plans/optimize-manifold-projectors-hpao-lsmr-2026-09-23-complete.md`.

## Established facts

- `experiments/diagnostic.py` runs base/noise/correlation/scarce scenarios with local one-factor estimator variants and paired data/initialization seeds. Full default: 12 seed per variant, 1440 fits across all modes/scenarios; smoke: 48 fits. `--dry-run` and `--analyze` are available.
- Report files under `benchmark_outputs/diagnostic/<run>/` are `run.json`, `diagnostics.{csv,json,md}` and normal `series/*` artifacts. New runs record a code SHA-256 and reject source changes during execution. Selection uses the first half of seed; validation uses the rest. Numerical failures remain in recovery denominators and now retain fit wall-clock time in `experiments/runner.py`.
- Full base run (360 fits, 12 seed per variant) is at `docs/experiments/diagnostic_2026-09-23/20260923T184703547414-all/`. Its report was regenerated with the final selection rule; its numeric fits preceded the code-hash field, so this initial artifact has Git/dirty provenance in `series.json` but no run-time code hash. Single baseline: 6/6 validation recoveries; faster unregularized candidate failed one validation trust certificate, so baseline retained. Multi `solver_max_steps=80`: 6/6 selection, 5/6 validation recoveries versus baseline 5/6 and 4/6; preliminary only, with wide Wilson intervals. Manifold: 0 validation recoveries for every variant, frequent rank-deficient local slopes, no recommendation.
- 48-fit smoke across all three modes and four scenarios completed before the final failure-timing change; it is a functionality check, not scientific evidence for stressed regimes. Fresh 4-fit manifold smoke with seed 4 confirmed all numerical failures have fit time and empty traced memory; fresh run/analysis code SHA-256 matched.
- Scientific classification: tested parameter changes are explicit experimental `ESTIMATOR` variants; production estimator/defaults unchanged. Details and limits are in `experiments/README.md`; selection decision is recorded in `agent-notes/DECISIONS.md`.
- Ruff, Pyright (new diagnostic and runner), and `git diff --check` passed. No existing historical artifacts were overwritten. Earlier dirty CPU optimization work remains intact.

## Next action

No required work remains. The full noise/correlation/scarce scenarios are available for a later, separately budgeted run; no parameter recommendation should be inferred from their smoke results.
