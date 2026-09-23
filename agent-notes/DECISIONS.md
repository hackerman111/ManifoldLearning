# Decision log

Append only durable research/architecture decisions. Routine edits do not belong here.

Use this form:

```markdown
## YYYY-MM-DD — <short decision>

- **Decision:** <what was chosen>
- **Reason/evidence:** <why>
- **Alternatives rejected:** <only material alternatives>
- **Affected:** <paths, APIs, formulas, experiment protocol>
- **Status:** active | superseded by <entry/date>
```

## 2026-09-23 — Isolated CPU fit profiling protocol

- **Decision:** Use fixed synthetic inputs and one child process per repeat, with one BLAS thread, a warm-up, ten representative repeats, time-only hooks for phase timing, process peak RSS, and separately measured `tracemalloc`/`cProfile` overhead.
- **Reason/evidence:** Both public models completed control and representative baseline fits with identical outputs within each case. Per-process `ru_maxrss` avoids carry-over from previous fits; existing multi phase timers and manifold private hooks expose the fit phases without changing the estimator.
- **Alternatives rejected:** Reusing one process for RSS peaks makes later peaks ambiguous; using only `tracemalloc` misses native NumPy/BLAS memory and perturbs runtime.
- **Affected:** `benchmarks/fit_bottlenecks.py`, `docs/experiments/bottlenecks_2026-09-23/`, `PLAN.md`.
- **Status:** active

## 2026-09-23 — Require HPAO convergence in the multi profiling protocol

- **Decision:** Set HPAO `max_steps=50, tol=1e-6` explicitly for final multi runs and require `solver.converged=True` on every outer step.
- **Reason/evidence:** The default five-step limit returned `converged=False` for J=48 and J=96. Paired trial fits with a 50-step limit converged on both steps at both shapes.
- **Alternatives rejected:** Interpreting incomplete five-step fits as ordinary runtime bottlenecks would violate the plan's solver stop condition.
- **Affected:** `benchmarks/fit_bottlenecks.py`, multi profiles under `docs/experiments/bottlenecks_2026-09-23/`.
- **Status:** active
