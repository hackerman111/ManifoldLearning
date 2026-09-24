# Current agent state

## Task

- ID: improve-multiindex-quality-2026-09-24.
- Status: active; plan is written in `PLAN.md`.
- Goal: diagnose and improve Multi-index recovery on representative v2 points using paired configuration experiments and untouched validation seeds.

## Established facts

- The inspected Multi v2 artifact `benchmark_outputs/experiments/20260924T012524585775-multiv2/` planned 22 series but records only 7 completed series (1590 fits). None formally converged; 197 passed `trace_score >= 0.95`; all complete runs stopped at `outer_steps`. The suite manifest still says `running`; no runner process was found at inspection. The multiplicative-frequency folder has 24 rows for its first point and is incomplete.
- The recorded Multi v2 setup uses `outer_steps=3`, LSMR, and `solver_max_steps=5`. The d=10 point passes the quality threshold in 30/30 runs; the n=1000,d=100 point has median trace score about 0.575. These are candidate sentinel conditions, not generalization evidence.
- A prior paired diagnostic at d=6,n=240 found a preliminary benefit from `solver_max_steps=80` (5/6 validation recovery vs 4/6 baseline). It does not validate transfer to the high-dimensional v2 points. See `agent-notes/DECISIONS.md`.
- Quality, convergence, recovery, and numerical failure remain separate outcomes. The primary metric is schema-9 `trace_score`, higher is better; recovery requires convergence and passing the unchanged 0.95 threshold.
- The prior completed catalog-trimming plan is archived at `agent-notes/history/plans/multiv2-complete-multi-catalog-7points-2026-09-24-complete.md`. Preserve all other dirty workspace changes.

## Current hypothesis

The five-step inner solver cap and three-step outer loop may explain the lack of formal convergence. Inspect the recorded inner solver statuses first; test solver and outer budgets independently before tuning estimator parameters.

## Active routes

- `agent-notes/ADP/multi-index.md`, `agent-notes/ADP/index-pipeline.md`, `agent-notes/ADP/solvers.md`.
- `experiments/diagnostic.py`, `experiments/multiv2.py`, `experiments/multi.py`, `experiments/runner.py`, `experiments/data.py`.
- `ADP/engine/common/index_fit.py`, `ADP/solver/LSMR.py`; focused tests under `tests/`.
- Artifact root: `benchmark_outputs/experiments/20260924T012524585775-multiv2/`.

## Next action

Execute T1 from `PLAN.md`: inspect per-run traces and freeze exact sentinel configurations. Do not run fits or edit estimator code before that diagnosis.
