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

## 2026-09-23 — Optimize rank-one manifold and weighted HPAO actions under the current objectives

- **Decision:** Implement the EXACT `m=1` manifold recovery and rank-one penalty action as a specialized path, retaining the current `m>1` path. Implement the EXACT HPAO A/A* action with preweighted local coefficients and cache `Aᵀr` within each correction. Keep current solver tolerances, certificates and acceptance rules. Retain old reference functions for paired comparison.
- **Reason/evidence:** Captured real manifold normal/RHS agree with dense reference to `3.6e-15`/`2.7e-15`; direct `m=1` recovery gives projector distance `1.18e-16` and 4–5× lower per-target recovery time. Rank-one penalty action agrees exactly and is about 20% cheaper per call. Captured HPAO A/A* action agrees within `5.4e-15`, preserves a certified correction and reduces three local solve timings by 7–14% without changing the 16 LSMR iterations.
- **Alternatives rejected:** Dense `(md)²` manifold solve has unbounded quadratic memory in `d`. Exact diagonal right scaling of augmented HPAO ridge reduced 16→12 Krylov iterations but was slower in three paired micro-runs. The existing bounded dense HYBRID variant remains explicit.
- **Affected:** `ADP/engine/manifol_engine/optimisation.py`, `ADP/solver/LSMR.py`, `docs/manifold_hpao_optimization_math.md`, paired benchmark artifacts. Acceptance of production changes remains conditional on full-fit quality, time and RSS measurements.
- **Status:** active

## 2026-09-23 — Retain exact CPU optimizations after paired full-fit validation

- **Decision:** Keep the rank-one manifold and preweighted HPAO implementations in the default CPU paths. Preserve the general manifold, old benchmark reference, unchanged objective and certificates. Record GPU runtime as unverified on this host.
- **Reason/evidence:** On one checkout and 10 child processes per variant/shape, manifold full-fit median fell 7.45%, 6.48% and 8.91% for base/J2/control with projector distance ≤`3.22e-15`. Multi fell 2.83%, 5.37% and 2.82% in the first series; reverse-order base/J2 remained positive at 4.01%/5.15%, while control improved only 1.14% within measurement spread. Single paired fit fell 46.316→42.676 ms. No material RSS growth or solver failure occurred; 305 tests passed, 28 GPU tests skipped for no device.
- **Alternatives rejected:** Applying the scaled augmented HPAO ridge candidate by default lacked a local timing gain. A conditional threshold for small multi problems is unsupported by the present noisy control series and would add complexity without clear benefit.
- **Affected:** `ADP/engine/manifol_engine/optimisation.py`, `ADP/solver/LSMR.py`, `docs/manifold_hpao_optimization_math.md`, `docs/experiments/manifold_hpao_opt_2026-09-23/`.
- **Status:** active

## 2026-09-23 — Paired diagnostic benchmark with held-out seed validation

- **Decision:** Keep parameter tuning as explicit one-factor experimental variants in `experiments/diagnostic.py`. Share data and initialization streams within each scenario; select only on the first half of seed and validate on the rest. Count numerical failures in recovery denominators. Treat quality differences up to 0.01 as practically tied, retain baseline unless a candidate is at least 10% faster on paired selection fits, and require validation recovery ≥0.8 with no numerical errors before a preliminary recommendation.
- **Reason/evidence:** Existing suites cover many factors but do not validate a selected configuration on independent seed. In the 360-fit base run, a naive quality tie-break selected `N_loc=30` for single although baseline also recovered 6/6 validation seed; the revised rule retained baseline after the faster unregularized candidate failed a validation trust certificate. Multi `solver_max_steps=80` recovered 6/6 selection and 5/6 validation versus baseline 5/6 and 4/6. Manifold had zero recoveries in all ten variants and repeated rank failures, so no setting is recommended.
- **Alternatives rejected:** Ranking all fits on the same seed would report a selection-biased winner. A full Cartesian hyperparameter search is expensive and cannot be interpreted as a local parameter effect. Treating tiny quality differences as decisive is unsupported by six validation seed.
- **Affected:** `experiments/diagnostic.py`, `experiments/runner.py`, `experiments/README.md`, `docs/experiments/diagnostic_2026-09-23/`.
- **Status:** active
