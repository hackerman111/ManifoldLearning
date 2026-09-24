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

## 2026-09-23 — Isolate a local quadratic manifold estimator for validation

- **Decision:** Implement an explicit `ADP_Manifold(estimator="local_quadratic")` variant for `m=1`: a 60-neighbor second-order pilot, quadratic correction of directional moments, a self-only center graph, and an observation-mass floor of six. Keep the existing estimator accessible as `estimator="manifold"` while the candidate is validated on untouched seed.
- **Reason/evidence:** `docs/experiments/manifold_recovery_2026-09-23/{reference.md,selection.md}` show the live normal operator matches a direct reference; even true projectors plus moments degrade under the broad graph. On fixed selection seed 0–9 the combined variant recovered 10/10 with zero errors and RMS sine 0.034–0.076, while the baseline recovered 0/10 with six failures. Each component alone failed the threshold or still had numerical failures.
- **Alternatives rejected:** Treating the inconsistent manuscript penalty mass as an established bug is unsupported. Floor 12 had no material quality gain over floor six. Two or three graph neighbors reintroduced large directional bias on the radial geometry. No validation seed was used to select this variant.
- **Affected:** `ADP/core/manifold/ADP_Manifold.py`, `ADP/engine/manifol_engine/{fit.py,weights.py}`, the experiment probe and focused tests. The default and any broader recovery claim remain contingent on M4 validation.
- **Status:** active

## 2026-09-23 — Retain the manifold default and publish the validated opt-in estimator

- **Decision:** Keep `estimator="manifold"` as the public default and expose the validated `estimator="local_quadratic"` option for `m=1`. Store second-order coefficients in packed symmetric form rather than per-center dense `d×d` matrices. Describe `h_min` plus certified inner solves as scheduled completion, not outer stationarity.
- **Reason/evidence:** Frozen validation seed 100–119 gave the option 20/20 recovery, zero numerical errors and maximum all-center RMS sine 0.1485 versus the 0.2 threshold; the original gave 0/20 with ten failures. A different noiseless varying geometry gave 5/5; CG/hybrid quality agreed within 1.34e-9. The packed representation changed held-out quality by at most 1.10e-15 with identical outcomes. The new estimator assumes a locally quadratic response and removes cross-center pooling, so the evidence is insufficient to replace the general default. Fixed-shape median fit cost increased from 0.0708 to 0.0981 s with similar measured memory.
- **Alternatives rejected:** Replacing the default globally would silently change every `m=1` estimator based on two tested designs. Keeping dense Hessian arrays would violate the plan's working-memory invariant. Relabeling scheduled completion as fixed-point convergence is unsupported by the trace.
- **Affected:** `ADP/core/manifold/ADP_Manifold.py`, `ADP/engine/manifol_engine/{fit.py,weights.py}`, `agent-notes/ADP/manifold.md`, `docs/experiments/manifold_recovery_2026-09-23/`.
- **Status:** active

## 2026-09-24 — Expand manifold dimension coverage

- **Decision:** Keep five joint points at `d=3,4,6,8,10` in the primary `manifold` series and retain all five in `overview`. Use a separate one-factor `manifold-d` grid with 10 values and five distributed overview values.
- **Reason/evidence:** The prior default overview exposed only three dimension values in each series. Dry-run confirms five primary points, five dimension-sweep overview points, and 10 full dimension-sweep points. The primary series co-varies sample size and noise, so `manifold-d` remains the isolated dimension comparison.
- **Alternatives rejected:** Extending only the full `manifold-d` grid leaves the primary series unchanged.
- **Affected:** `experiments/manifold.py`, `experiments/profiles.py`, `experiments/README.md`, `tests/test_experiment.py`.
- **Status:** active

## 2026-09-24 — Make every manifold scenario a seven-level boundary probe

- **Decision:** Use exactly seven valid levels in each of the 13 manifold series and retain every level in `overview`. Put configuration sweeps near their feasibility limits and extend noise, correlation, scale, penalty, and iteration axes into stressed regimes.
- **Reason/evidence:** The requested `--runs 30 --solver hybrid` dry-run now schedules 210 fits per series and 2730 total. The chosen feasibility-edge values satisfy the live manifold limits; reports retain per-fit recovery, quality, convergence, and failures. Actual failure thresholds remain to be measured by running the suite.
- **Alternatives rejected:** The generic overview's three-point reduction hides transitions; interior-only levels do not probe the known feasibility edges.
- **Affected:** `experiments/manifold.py`, `experiments/profiles.py`, `experiments/README.md`, `tests/test_experiment.py`.
- **Status:** active; supersedes 2026-09-24 — Expand manifold dimension coverage

## 2026-09-24 — Probe effective manifold limits without hidden center changes

- **Decision:** In the sample-size sweep, hold `N_J=36`; start the local-mass sweep at `N_loc=10` and the center-count sweep at `N_J=12`. These values keep the runner's automatic minimum-center adjustment from merging adjacent levels.
- **Reason/evidence:** `_effective_config()` raises `N_J` to at least `ceil(n/N_loc)`. With `n=240` and `N_loc=1`, the nominal local-mass change also raises effective `N_J` from 24 to 240; values from 10 preserve 24 centers. For the center sweep, 12 is `ceil(240/20)`. `N_J=36` is feasible for every sample size 61–720 and stays fixed.
- **Alternatives rejected:** Using nominal minima 1/7 or leaving `N_J=24` in the sample-size sweep would conflate factors through the effective configuration.
- **Affected:** `experiments/manifold.py`, `experiments/README.md`, `PLAN.md`.
- **Status:** active; supersedes the effective-boundary choices in 2026-09-24 — Make every manifold scenario a seven-level boundary probe

## 2026-09-24 — Isolate all multi series in a seven-point v2 suite

- **Decision:** Expose every source `multi.catalog()` series under a `multiv2-*` selector and keep its full grid in v2 `overview`. Expand each short binary comparison over four noise levels; expand the link comparison over four frequencies for both link functions. Keep v2 outputs separately labeled and leave the original multi catalog/profile unchanged.
- **Reason/evidence:** The source catalog has 38 series, eight with two points. The v2 dry-runs show a minimum of seven distinct points, 2255 fits for its 24-series default, and 9812 fits for all series at one run. The original multi dry-run remains 930 fits.
- **Alternatives rejected:** Mutating original selectors would alter established `multi` output grids. Keeping generic three-point overview reduction would violate the requested v2 coverage. Adding unsupported values to binary factors would not represent valid estimator settings.
- **Affected:** `experiments/multiv2.py`, `experiments/suite.py`, `experiments/profiles.py`, `experiments/README.md`, `tests/test_experiment.py`.
- **Status:** active

## 2026-09-24 — Exclude two expensive sweeps from Multi v2

- **Decision:** Remove `mi-3` and `mi-5` from the v2 catalog and default selector list; retain the original experiments in `experiments.multi`.
- **Reason/evidence:** At 30 repetitions they add 8640 fits (48 and 240 points). Removing them reduces the v2 default to 4890 fits while preserving every other v2 grid and the original multi suite.
- **Alternatives rejected:** Hiding them only from defaults would leave the same expensive series in `--experiment all`, contrary to the request to remove them from v2.
- **Affected:** `experiments/multiv2.py`, `experiments/README.md`, `tests/test_experiment.py`.
- **Status:** active; supersedes 2026-09-24 — Isolate all multi series in a seven-point v2 suite

## 2026-09-24 — Stop Multi v2 tuning at the inner HPAO convergence gate

- **Decision:** Keep current Multi-index defaults. Stop the planned outer-step and estimator sweeps, and leave validation seed 2000–2019 untouched because no candidate passed the inner convergence and hard-point quality gates.
- **Reason/evidence:** On paired selection seed 1000–1009, HPAO limit 80 recovered 8/10 at `d=10`, but 0/10 at `d=100,n=1000`. At the hard point, 16/30 inner HPAO calls exhausted 80 steps and 13 of those retained riemannian gradient above `1e-6`. All accepted linear corrections passed the normal-residual certificate; no numerical failure occurred. All 10 hard-point fits missed `trace_score >= 0.95` at every tested limit. See `docs/experiments/multi_quality_2026-09-24/selection_inner.md` and the recorded `summary.json`/`runs.csv`.
- **Alternatives rejected:** Increasing outer steps or tuning localization before inner HPAO convergence would confound the diagnosis. Treating the automatic selection ranking as a recommended candidate would ignore zero hard-point recoveries and the untouched validation gate.
- **Affected:** `PLAN.md`, `agent-notes/STATE.md`, `experiments/multiv2_quality.py`, `experiments/README.md`; no ADP estimator/default code changed.
- **Status:** active; closes the current plan at its prespecified stop condition.

## 2026-09-24 — Plan solver search around common stationarity and frozen statistics

- **Decision:** Compare existing HYBRID and at most two mathematically justified reduced-objective solvers on frozen multi-index tasks before full-fit selection and independent validation. Preserve the objective and use a common external convergence evaluator.
- **Reason/evidence:** Prior cap-80 hard-point runs certified only 14/30 HPAO calls despite accepted linear-correction certificates. Faster linear solves alone may not address slow nonlinear progress. Reduced-objective methods remain hypotheses, especially near changing local rank.
- **Alternatives rejected:** Selecting by raw iteration count, timing only successful runs, or immediately changing estimator settings would confound numerical efficiency and statistical quality.
- **Affected:** PLAN.md, agent-notes/STATE.md; prior completed plan archived. No solver implementation changed in this planning task.
- **Status:** planned; experiments and candidate decisions await execution of S1–S7.

## 2026-09-24 — Require proofs and adaptive hypotheses in solver search

- **Decision:** Per user request, require a written proof with explicit assumptions and a separate audit before candidate implementation. Empirical/reference checks remain necessary but do not substitute for proof. Reassess and record new falsifiable hypotheses during selection-stage research.
- **Reason:** Correct formulas, rank-truncation behavior and convergence claims require justification beyond observed fits; intermediate evidence may invalidate the initial candidate list.
- **Constraints:** New candidates pass the same gate; hypothesis predictions and bounded budgets are recorded before trials. Validation is never reused for tuning.
- **Affected:** PLAN.md and agent-notes/STATE.md; documentation only.
- **Status:** active planning requirement.

## 2026-09-24 — Admit structural local rank in reduced-solver proofs

- **Decision:** The reduced-objective proof must cover constant structural rank below `m` and guard the nonzero singular spectrum. A blanket full-rank assumption for every local projected system is not an admissible domain for the 12 frozen selection tasks.
- **Reason/evidence:** S1 found 1–21 rank-deficient projected centers per frozen task, and each had `rank(U_j)<2`; some `U_j` are zero. No observed projected rank loss came from a full-rank `U_j`. See `docs/experiments/multi_solver_search_2026-09-24/s1.md` and the frozen manifest.
- **Alternatives rejected:** Dropping these centers or changing their mass would alter the objective; treating `rcond=None` as an exact pseudoinverse theorem would hide numerical rank transitions.
- **Affected:** `PLAN.md`, `agent-notes/STATE.md`, S2 proof/guards and common certificate. The production estimator is unchanged.
- **Status:** active for S2; differentiability beyond the guarded domain remains to be proved.

## 2026-09-24 — Reject existing HYBRID for this solver-search selection

- **Decision:** Do not advance existing HYBRID to full-fit selection on the two planned points; proceed to the proof gate for a reduced-objective nonlinear method.
- **Reason/evidence:** All 24 frozen trials passed, but median paired HYBRID/LSMR time was 1.037 on d100 and 1.919 on d10, with unchanged certification (3/6 and 6/6) and AO counts. HYBRID reduced Krylov iterations on d100 but not wall time. See `docs/experiments/multi_solver_search_2026-09-24/s2_hybrid.md`.
- **Alternatives rejected:** Promote a solver by iteration count alone; rerun or tune HYBRID after its prespecified hypothesis failed.
- **Affected:** S2/S4 candidate selection and `hypotheses.md`; production default remains unchanged.
- **Status:** active selection decision; no universal claim about HYBRID.

## 2026-09-24 — Admit one guarded reduced L-BFGS prototype

- **Decision:** Implement one isolated reduced-objective Riemannian L-BFGS solver after the S2 proof gate; do not select Gauss–Newton yet. Require rank/cutoff and exact-objective error guards, external stationarity checks, and explicit failure at the boundary.
- **Reason/evidence:** The proof and independent audit in `docs/experiments/multi_solver_search_2026-09-24/proofs/` derive the truncated objective gradient, geometry and conditional stationarity convergence. Synthetic reference tests pass; on 12/12 frozen inputs rank stayed stable under small tangent perturbations and finite-difference errors decreased. Existing HYBRID did not improve the nonlinear certification count.
- **Alternatives rejected:** An unguarded full-rank-only derivation excludes actual structural rank-deficient centers; promoting HYBRID by Krylov count alone did not meet wall-time criteria; a second prototype lacks current evidence.
- **Affected:** S3 implementation, `hypotheses.md`, proof/audit documents; estimator and default remain unchanged.
- **Status:** active, conditional on S3 focused checks.

## 2026-09-24 — Close the bounded multi-index solver search without changing the default

- **Decision:** Reject HYBRID, pure guarded reduced L-BFGS, and one-step HPAO plus guarded reduced as replacements on the selected multi-index points. Stop before full-fit selection and held-out validation; keep the public HPAO solver/default and statistical estimator unchanged. Retain both reduced methods as isolated research prototypes with explicit limitations.
- **Reason/evidence:** HYBRID had median paired wall ratio 1.037 on d100 with no certification gain. Pure reduced certified 6/6 d100 frozen tasks but had median ratio 0.883, 4.244× d10 slowdown and three materially worse d100 objectives. The preregistered second/last prototype H5 certified only 5/6, had median ratio 0.971 and objective +1.214274 versus cap80 / +1.292610 versus cap320 on the distinguishing task. See `docs/experiments/multi_solver_search_2026-09-24/{s2_hybrid,s4_initial,s4_final}.md` and their raw artifacts. All 67 relevant regression tests and Ruff checks passed.
- **Alternatives rejected:** Promoting a method by certificate count alone, tuning HPAO warm-start length or reduced parameters after seeing six frozen trials, and using held-out seed to select a method would violate the predeclared objective/speed gate and budget.
- **Affected:** `PLAN.md`, `agent-notes/STATE.md`, `hypotheses.md`, S4 report; no public API/default or estimator change. Seed 3000–3019 and 2000–2019 remain unused.
- **Status:** final for this bounded plan; a distinct hypothesis requires a new plan and independent selection evidence.

## 2026-09-24 — Reopen solver search after causal failure diagnostics

- **Decision:** User-authorized retry starts with certificate-directed precision in the existing explicit HYBRID relative mode; no change to defaults. No warm-length sweep.
- **Evidence:** `docs/experiments/multi_solver_retry_2026-09-24/r1.md`: same-lambda tighter solves remove four normal rejects; three worse reduced endpoints have positive local curvature, and the slow reference evaluator costs 8–10x more than a diagnostic batched envelope.
- **Scope:** H6 proof/audit precedes implementation, bounded frozen gate retained; possible H7 requires a separate proof. Prior negative results remain valid for their methods.

## 2026-09-24 — Close the solver retry after both frozen gates failed

- **Decision:** Do not promote certificate-directed HYBRID or the reduced Gauss–Newton prototype. Restore the pre-retry production HYBRID path, preserve its rejected change as `docs/experiments/multi_solver_retry_2026-09-24/h6_rejected.patch`, and keep GN only as an explicit experiment. Do not run full fits or held-out validation for these candidates.
- **Reason/evidence:** R1 established that some normal-residual rejects came from LSMR stopping accuracy at unchanged lambda, but H6 halved rejects with no d100 AO/certificate improvement and 1.264× median wall. H7's full residual Jacobian passed dense/adjoint tests but gave only 4/6 d100 certificates, 3.883× median spent wall, 1.748× RSS, one worse certified objective and one line-search failure near rank cutoff. See `docs/experiments/multi_solver_retry_2026-09-24/completion.md` and linked raw artifacts.
- **Alternatives rejected:** Relaxing the original rank/cutoff guard, promoting by certificate count while ignoring objective/time/failure, or retuning warm length/tolerance on the same selection data. These would change the mathematical contract or exceed the preregistered search.
- **Affected:** Research-only `experiments/reduced_gauss_newton.py`, diagnostic/test/report files, `PLAN.md`, `STATE.md`, solver route. Public solver/default and estimator unchanged; held-out seed3000–3019 unused.
- **Status:** final for this bounded retry; another attempt needs a distinct derived hypothesis and fresh selection gate.
