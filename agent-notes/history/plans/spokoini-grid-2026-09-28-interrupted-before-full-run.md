task_id: spokoini-grid-2026-09-28
status: active
change_class: experiment only; production estimator and solver unchanged
---

# Spokoini multi-index grid

Goal: add a runnable, named grid for the Hristache–Juditsky–Polzehl–Spokoiny
design in `Spokoiny.md`, using only the current ADP estimator. Preserve the
specified Beta design, links, truth bases, noise levels, grid points, and
replication counts. Record per-iteration quality and requested checkpoints.

Non-goals: ADE/SIR II/PHD implementations or comparisons; changing current
ADP defaults; reinterpreting existing `tau` experiments; claiming a numerical
replication of the historical implementation when its schedule differs from
the current engine.

Known evidence: the shared `experiments` runner already records reproducible
data seeds, traces, fit time, failures, and summaries. Its existing `tau`
means a common Gaussian factor, and it standardizes responses; neither matches
`Spokoiny.md`. The current engine exposes `a` and `h_min`, but not the paper's
separate `rho` schedule or explicit initial/max bandwidth. Its multi-index
adaptation factor starts at one, matching `rho_1`; only `a_h` maps directly to
the current `a` field. Keep the current bandwidth initialization and floor.

Invariants: float64; independent coordinate draws
`X=2*Beta(1,tau)-1`; exact m=1/2/3 links and fixed orthonormal truth; additive
Gaussian noise at the specified sigma; same current ADP implementation for all
fits; all errors retained; independent deterministic seeds; no production
estimator/API/default changes. Main metric is the live normalized trace score;
report the corresponding geometric loss `m*(1-trace_score)` for every fit
checkpoint (mean and IQR).

Read set: `Spokoiny.md`; `agent-notes/ADP/multi-index.md`;
`experiments/{models,data,multi,registry,runner,profiles,suite}.py`;
`ADP/cli/experiment_utils.py`; `ADP/core/{ADP_Config,multi/ADP_multi_index}.py`.

Work units:
- [x] Add isolated Spokoini feature/link generation without changing old
  `tau` or standardized-link behavior.
- [x] Add named m=1/2/3 grid selectors with exact cells and 250/100 runs;
  use current multi ADP defaults, set the directly mappable `a_h`, and record
  the remaining schedule mismatch.
- [x] Add requested 1/2/4/8/last loss mean/IQR summary and usage notes.
- [x] Inspect the run plan and generated diff; dry-run preview confirms exact
  cells and fit count. No comparison implementation or test suite.
- [ ] Launch the full 5,450-fit grid with one BLAS thread; retain all series,
  failures, and checkpoint summaries.
- [ ] Update durable state and decision record with completion status and
  output directory; retain prior completed plan.

Stop conditions: no baseline-comparison implementations and no estimator
tuning. Stop after the full grid is recorded and the suite manifest has a
terminal status.

Verification: full-profile dry-run reports 7 series and 5,450 fits. A separate
dry-run confirms the existing `mi-boundary-nd` catalog still has 63 points and
630 fits. Python syntax compilation and `git diff --check` pass. The full grid
is the remaining work unit.
