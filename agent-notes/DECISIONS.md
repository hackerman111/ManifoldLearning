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

## 2026-09-24 — Start a separate multi-index center-selection study

- **Decision:** Seek fewer, more informative centers as an explicit `ESTIMATOR` variant, with random `J₀` and random same-`J` controls. Keep solver/default fixed. Gate promotion on paired external `trace_score` and recovery, including all failures and total selection cost.
- **Reason:** The completed solver retry found no passing candidate. Live center selection is random, and changing its output also changes initialization, localization and outer-step SSE; a solver objective cannot certify quality across different center sets.
- **Alternatives rejected:** Selecting by oracle projector/recovery, comparing raw losses across different `J`, or claiming universal non-degradation from a surrogate score.
- **Affected:** `PLAN.md`, `agent-notes/STATE.md`; previous solver plan archived. No numerical code changed in this planning task.
- **Status:** planned; H1/H2 and acceptance gate await C1–C4 evidence.

## 2026-09-25 — Continue center selection after C1 pilot

- **Decision:** Proceed to C2 with one frozen, training-only selection rule; do not infer EDR quality from C1.
- **Evidence:** The precommitted neighborhood-heterogeneity gate passed on 10/10 selection seeds at both d10 and d100, with 20/20 complete pilots, 8.636 s total wall and 97.40 MiB peak RSS. See `docs/experiments/multi_center_selection_2026-09-25/c1_result.md` and raw CSV.
- **Alternatives rejected:** Stopping for lack of structural variation, or promoting a center rule from neighborhood metrics alone.
- **Affected:** `PLAN.md`, `agent-notes/STATE.md`, C1 experiment artifacts; solver and public estimator unchanged.
- **Status:** active; H2 and full-fit recovery await C3.

## 2026-09-25 — Stop the center-selection branch at C2

- **Decision:** Reject the frozen coverage/logdet center rule and end this plan before C3/C4. Keep random center selection and all public defaults.
- **Evidence:** C2 reference test passed and 20/20 selection jobs completed without numerical failures, but only 0/10 d10 and 9/10 d100 yielded `J<500`. The precommitted absolute coverage threshold 0.99 was unreachable for the feasible center pool in 11 jobs; direct recomputation confirmed d10 seed4000 feasible coverage 0.959 and d100 seed4008 0.989. See `docs/experiments/multi_center_selection_2026-09-25/c2_result.md` and raw CSV.
- **Alternatives rejected:** Relaxing the threshold after viewing these selection results, proceeding to paired fits with missing candidates, or treating an internal logdet ratio as recovery evidence.
- **Affected:** `PLAN.md`, `agent-notes/STATE.md`, `experiments/README.md`; isolated experiment scripts and tests only. No `ADP/` production edits.
- **Status:** done with a negative result; H2, full-fit recovery and held-out validation remain untested.

## 2026-09-26 — Start proof-first search for a shared multi/manifold ADP idea

- **Decision:** Begin with a bounded ADP-specific proof of orthogonal directional blocks (H1) for multi and manifold. If its mathematical or full-cost gate fails, consider one variance-reduced stochastic solver (H2) for a fixed objective. Do not implement or run candidate full fits before the proof and independent audit; keep all production defaults unchanged.
- **Reason/evidence:** Both live paths use directional `I/U` moments, giving a possible common mechanism. Previous multi solver retries and center-selection failed their frozen gates; neither tested this mechanism. Existing orthogonal-feature and Riemannian-stochastic literature supplies ideas, not an ADP theorem or a performance claim.
- **Alternatives rejected:** Retune rejected solver/center rules on their selection data; infer EDR recovery from a sketch-variance or inner-solver theorem; transfer manifold `m=1` validation to `m>1`.
- **Affected:** `PLAN.md`, `agent-notes/STATE.md`, archived prior plan; research documentation only.
- **Status:** active research plan; H1/H2 remain unproved hypotheses.

## 2026-09-26 — Add a SPINN-inspired separable neural estimator hypothesis

- **Decision:** Add H3 to the active proof-first search: jointly learn an EDR basis and a low-rank separable response surrogate, then form gradient-based projectors. Evaluate a multi-index version first; admit local manifold charts only with a separate derivative-error theorem and full training-cost argument. At R0 choose at most two ideas for full proof and one eventual prototype. H2 remains a reserve.
- **Reason/evidence:** The user specified Separable Physics-Informed Neural Networks. Its per-axis factorization suggests a candidate approximation family, while ADP observations are irregular and have no governing PDE. The existing MLP pilot only initializes a basis; freezing that basis in H3 could not repair its span. Prediction fit alone cannot certify derivative or EDR quality.
- **Alternatives rejected:** Import the PDE residual or paper's grid speedup as an ADP claim; train factors on a frozen pilot projection and call the resulting projector improved; add a neural default before a gradient-error and eigengap proof.
- **Affected:** `PLAN.md`, `agent-notes/STATE.md`; research documentation only, no estimator code or public API change.
- **Status:** active hypothesis, unproved and untested.

## 2026-09-26 — R0 selects isotropic orthogonal blocks only

- **Decision:** Take H1 into R1 for normalized multi-index with independent isotropic redraw and for the original manifold estimator. Treat the changed joint direction law as an `ESTIMATOR` variant. Do not advance H3 or H2 in this cycle.
- **Evidence:** Exact fixed-weight ADP moment factorization and live objectives are recorded in `docs/experiments/proof_first_shared_2026-09-26/r0.md`. The two selected paths use independent unit-Gaussian directions, so one common orthogonal-block argument is possible. SPINN's Cartesian-grid evaluation saving has no analogous coordinate reuse on irregular projected ADP samples, and response MSE alone cannot control gradients.
- **Alternatives rejected:** An anisotropic-multi variance claim without a separate proof; freezing a pilot basis in H3; starting a neural training implementation without derivative-error control; mixing stochastic H2 and estimator H1 in one comparison.
- **Affected:** `PLAN.md`, `STATE.md`, R0 dossier, manifold ADP/TeX route correction. No production algorithm or API change.
- **Status:** R0 done; H1 proof and independent audit pending.

## 2026-09-26 — Admit isotropic H1 to one isolated CPU prototype

- **Decision:** Pass R1 only for pointwise fixed-weight variance reduction and conditional stability of one certified quadratic step. Build one CPU prototype of H1 and a small dense reference. Consider compact full-block representation only as a separately verified numerical representation of the same H1 Gram.
- **Evidence:** `docs/experiments/proof_first_shared_2026-09-26/h1_proof.md` derives Haar-block expectation/variance for `P=kd+r`, correctly aggregates shared manifold sketches, and bounds a fixed quadratic step and projector under a positive gap. Independent read-only audit in `r1_audit.md` confirmed those formulas and forced explicit limits on multi rank, HPAO endpoint, manifold sync reuse and `P≥d` memory.
- **Alternatives rejected:** Treating pointwise variance as full-fit recovery, claiming exact numerical equivalence of compressed rows across rank cutoffs, testing anisotropic multi without a theorem, or using `P=d=1000` at prohibitive memory cost.
- **Affected:** Research dossier, PLAN/STATE; production defaults/API unchanged.
- **Status:** R1 done; R2 reference, stress and frozen protocol pending.

## 2026-09-26 — Freeze H1 full-fit gate after R2 reference

- **Decision:** Run only the isolated compact-H1 estimator against original isotropic multi and explicit original manifold baselines, with six paired selection seeds, same-row iid controls where compacting occurs, and the precommitted per-fit quality/convergence/completion/failure and full wall/RSS gates in `r2_protocol.md`. Hold validation seeds unused unless a case passes.
- **Evidence:** Five small reference/gate tests, Ruff, Pyright and 64 MB direction-generation stress passed. The R0 manifold CLI timing was actually the `local_quadratic` default, so it was corrected and excluded. Original manifold m=2 repeatedly failed local rank on diagnostic seeds despite larger neighborhoods; keep this failure as a stress instead of tuning it away.
- **Alternatives rejected:** Compare H1 to the different `local_quadratic` estimator, weaken quality tolerance after selection, tune m=2 further on diagnostic rank failures, infer full-fit recovery from the variance theorem, or use held-out seeds for selection.
- **Affected:** `experiments/proof_first_h1.py`, `tests/test_proof_first_h1.py`, `docs/experiments/proof_first_shared_2026-09-26/{r0,r2_protocol}.md`, `PLAN.md`, `STATE.md`; no production algorithm/default/API changes.
- **Status:** R2 done; R3 paired selection authorized by plan.

## 2026-09-26 — Stop orthogonal directional H1 after negative R3

- **Decision:** Reject H1 as a full-fit improvement under the frozen quality/cost gate; do not add a production option or run held-out validation. Preserve the mathematical result and raw negative selection evidence. H2/H3, if revisited, require a separate proof and fresh protocol.
- **Evidence:** All 66 prespecified fits were recorded on seed 61000–61005 with the frozen fingerprint. Quality regressed on 2/6 multi d10, 1/6 multi d100, 2/6 manifold m1 and 2/3 completed manifold m2 pairs; m2 had three rank failures per variant. Median H1/baseline wall ratios were 0.820, 1.028, 1.009 and 1.000 (m2 only three completed), with zero certified recoveries. Same-row iid control at d10 had comparable cost and better quality than H1 on five of six seeds. See `docs/experiments/proof_first_shared_2026-09-26/r3_result.md` and raw `selection/` artifacts.
- **Alternatives rejected:** Relaxing per-fit tolerance or the 0.8 speed bar after looking at selection, discarding failed m2 fits, treating nonconverged multi quality as recovery, or reusing held-out seed 62000–62019 for tuning.
- **Affected:** Research report and `PLAN.md`/`STATE.md`/`experiments/README.md`; no production ADP/default/API changes.
- **Status:** final negative result for this bounded H1 cycle.

## 2026-09-27 — Formalize β functional and retain exact compression as isolated research

- **Decision:** Preserve current estimator/solver/default and document a conditional exact row compression using thin QR of [U,I], without rank truncation. Do not substitute gradient-PCA or a common-metric spectral solve for arbitrary local metrics. Keep tested NumPy variants in an isolated benchmark.
- **Evidence:** `docs/experiments/beta_functional_2026-09-27/report.md` derives the original/constrained objective, fixed-C proximal subproblem, full cross-block Hessian and QR residual isometry. Frozen d10/P40 compresses to 11 rows with fixed-C ridge errors <=7.63e-17; d100/P40 cannot reduce. Proportional local metrics are disproved. Final profiles/9 paired repetitions in `audit_final.json` show existing matmul actions dominate d100; substitutions give no compelling general trajectory gain. High-mass local-gradient stress exposes threshold-dependent rounding changes.
- **Alternatives rejected:** Independent per-β solves that discard cross terms; gradient averaging without local metrics; automatic direct solve(B,V,y) with changed P-dependent cutoff; reporting reduced design allocation or faster stationarity kernel as full-fit acceleration; discarding the existing manifold regression failure.
- **Affected:** PLAN/STATE, solver and multi-index TeX routing notes, isolated `benchmarks/beta_functional_audit.py`, 8 new reference tests, report/raw artifacts. Previous dirty work preserved; production `ADP/` unchanged.
- **Status:** completed formalization and bounded audit. Future QR integration needs original numerical cutoffs/normalizations and a separate trajectory/full-cost verification protocol.

## 2026-09-27 — Split HYBRID by solver route

- **Decision:** Move the original implementation into `ADP/solver/HYBRID/`; keep shared linear helpers in `HYBRID.py`, the multi-index workspace and HPAO route in `HYBRID_multi.py`, the manifold linear solver in `HYBRID_manifold.py`, and an explicit single route in `HYBRID_single.py` that delegates to current HPAO-LSMR. Preserve the package facade and public imports.
- **Reason/evidence:** Current HYBRID HPAO accepts only a multi-index basis; single-index uses the current LSMR solver, while manifold solves a separate B-subproblem. Separating these paths expresses the existing contracts without changing their objectives or tolerances.
- **Alternatives rejected:** Keep a mixed 700-line module or duplicate the single-index HPAO algorithm inside HYBRID.
- **Affected:** `ADP/solver/HYBRID/`, CLI/import callers, focused solver tests, `agent-notes/ADP/` routing.
- **Status:** active

## 2026-09-27 — Keep current ADP objective solves matrix-free for audited profiles

- **Decision:** Do not convert current single, multi, or manifold objective solves to sparse-matrix storage/factorization based on the audited profiles; retain the existing matrix-free paths. Reconsider only with evidence of structurally sparse operators on representative workloads.
- **Reason/evidence:** Across 10 small and 10 medium tasks per family, exact design density was at least 99.9375% and all normal matrices were 100% dense. Rare design zeros appeared only in medium zero-inflated-70 profiles. Manifold CSR graph density medians were 51.56%/64.23% and did not make its feature-space B-system sparse. See `docs/experiments/sparse_solver_suitability_2026-09-27/report.md` and `audit.json`.
- **Alternatives rejected:** Infer useful solver sparsity from input zero-inflation, near-zero coefficients, or the graph's CSR container without a sparse feature-space objective.
- **Affected:** Solver research direction and audit artifacts; no production code, estimator, or defaults changed. Runtime and memory remain unbenchmarked.
- **Status:** final for the audited synthetic profiles; not a universal claim about other data or operators.

## 2026-09-27 — Define recovery by geometric quality, report convergence separately

- **Decision:** Recovery requires finite-fit geometric quality only: single `abs(cosine)`, multi `trace_score`, and manifold aggregate local-projector quality plus a worst-center local-projector check. Solver convergence/stationarity remain diagnostics and do not gate recovery or candidate admission.
- **Reason/evidence:** The user requested that solver convergence stop being mandatory for recovery. Existing experiment records already preserve quality and solver diagnostics separately. For the supported manifold experiments (`index_dim=1`), maximum principal sine over all centers is the maximum local-projector distance and prevents a poor region from being hidden by the full-center RMS average.
- **Alternatives rejected:** Counting a high-quality estimate as unrecovered solely because an iterative solver stopped by its native status; judging manifold recovery by an average over centers without checking the worst center.
- **Affected:** `experiments/runner.py`, experiment report/analysis code, recovery study gates, manifold validation, `PLAN.md`, and `STATE.md`; no ADP estimation formula or solver is changed.
- **Status:** adopted; the existing 105/108 selection snapshot retains its old protocol and is incomplete.

## 2026-09-28 — Manifold generalization experiment at m>1

Use standalone analytic full-rank row(Dz) geometry with m=1,2,3,
flat/curved controls, noise and fixed local/broad support. Explicit
estimator=manifold is required because live default local_quadratic
is m=1-only. Record all-center and independent-query geometry separately
from solver debug; finite queries/scalar response do not prove universal
identifiability or continuum recovery.

Strict raise protocol failed180/180 (149 mass,31 rank); exploratory existing
stop option yields flat recovery17/20,9/20,2/20 for m1,m2,m3, and curved
m>1 0/80. This does not justify default changes. Preserve both protocols
and failures rather than weaken gates. Source/report:
experiments/manifold_generalization.py and
docs/experiments/manifold_generalization_2026-09-28/report.md.
Prior unfinished single/multi study plan/state archived in that directory.
Status: active experiment interpretation; no algorithm decision/promoted variant.

## 2026-09-28 — Spokoiny multi-index grid uses isolated current-ADP data design

- **Decision:** Add named `mi-spokoini-*` series to the existing multi-index experiment runner. Generate independent `2*Beta(1,tau)-1` coordinates, fixed m=1/2/3 truth bases and links, and unstandardized signal plus Gaussian noise. Preserve all fit traces/errors and summarize geometric loss at iterations 1, 2, 4, 8, and last. Do not implement ADE/SIR II/PHD comparisons.
- **Reason/evidence:** The existing `tau` generator is a common Gaussian factor and existing responses are standardized, so changing them would alter established experiments. An isolated feature distribution and named links preserve old semantics. Full-profile dry-run exposes the expected 7 series / 5,450 fits, and an existing-selector dry-run confirms `mi-boundary-nd` remains 63 points / 630 fits.
- **Alternatives rejected:** Reuse the old `tau` path or claim exact reproduction of historical algorithm settings. The current engine maps only `a_h` to `a`; it lacks separate `rho_min`, `a_rho`, `h_1`, and `h_max` controls, so its other bandwidth behavior remains current.
- **Affected:** `experiments/{data,multi,models,registry,profiles,runner,spokoini}.py`, `ADP/cli/experiment_utils.py`, `experiments/README.md`, `PLAN.md`, and `STATE.md`; no estimator or solver behavior changed.
- **Status:** implementation accepted; full grid execution is pending.

## 2026-09-28 — Implement the unified manifold grid with feasible live settings

- **Decision:** Add a separate `experiments.manifold_grid` runner, preserve the existing `manifold_generalization` protocol, deduplicate configurations by `(n,d,m,c,sigma)`, and use `N_lin=max(d+2,min(200,n-1))` so every listed cell passes the current API's size constraints.
- **Reason/evidence:** The specification lists 26 rows but four exact cross-series overlaps, leaving 22 unique configurations. Live `effective_config` requires `d+1 < N_lin < n`; fixed 200 is invalid for `n=100`, `n=200`, and `d=200`. Full-profile preflight therefore plans 22*250=5,500 unique fits.
- **Alternatives considered:** Repeat an identical configuration to reach the stated 23/5,750 total, or preserve invalid `N_lin=200` and record guaranteed pre-fit validation failures. Neither changes the specified data model; the runner records the selected feasible values per fit and in its manifest.
- **Affected:** `experiments/manifold_grid.py`, `experiments/README.md`, `agent-notes/ADP/manifold.md`, `PLAN.md`, and `STATE.md`; no production estimator/default or comparison code changed.
- **Status:** implemented; user clarification may revise the point count or boundary configuration.

## 2026-09-28 — Preserve strict manifold scale-boundary failures in the grid

- **Decision:** Keep `scale_boundary="raise"` in the specified grid and record each infeasible function-mass boundary as a failed fit. Treat `scale_boundary="stop"` only as a separate diagnostic variant.
- **Reason/evidence:** `Manifold exp.md` fixes the strict mode and explicitly excludes `stop` from the main grid. The user's partial full-profile run recorded the same infeasibility for 41 independent seeds in the first cell after four successful projector updates; the runner retains every per-seed outcome.
- **Alternatives rejected:** Automatically skip remaining seeds after repeated errors or switch the main grid to `stop`, which would censor the failure rate or change the specified estimator protocol.
- **Affected:** Interpretation of `experiments/manifold_grid.py` outputs and future run instructions; no estimator, solver, or runner behavior changed.
- **Status:** adopted for this grid.

## 2026-09-28 — Diagnose manifold generalization without changing the estimator

- **Decision:** Keep the current manifold estimator and frozen generalization protocols intact. Treat replacing relative spectrum by unit weights as a separate ESTIMATOR hypothesis requiring its own selection and validation; do not call it a universal fix.
- **Reason/evidence:** The paired flat broad `m=3,c=0` diagnostic changes recovery from 0/10 to 10/10 and exposes spectral localization feedback, but exact-gradient initialization on the broad curved graph still has center RMS 0.268–0.649. Exact center charts also fail the curved query gate. See `docs/experiments/manifold_generalization_2026-09-28/diagnosis.md` and its runnable `diagnosis.py`.
- **Alternatives rejected:** Attribute geometric failures to solver recovery, silently replace spectral weights in production, or relax the frozen recovery threshold after observing failures.
- **Affected:** Interpretation of manifold generalization results and future experiment design; no production code or frozen outputs changed.
- **Status:** adopted for this diagnosis.

## 2026-09-28 — Use explicit feasible boundary in the runnable manifold grid

- **Decision:** For `experiments.manifold_grid`, default to the existing explicit `scale_boundary="stop"` variant and retain the original strict `raise` protocol behind `--scale-boundary raise`. Keep all other estimator parameters and every numerical failure. In schema 2, assess recovery only for `m=1` or flat `c=0`, using center and same-center chart-estimation errors; retain raw/oracle nearest-chart distances as diagnostics.
- **Reason/evidence:** The user's interrupted full run and an earlier partial run contained 61/61 identical mass-boundary exceptions on the first cell. Paired seed 81000 reaches four updates then raises in strict mode; stop returns after five updates with target mass 80.0000000002 and solver residual 9.1e-15, allowing geometric assessment. It still fails recovery (center RMS/max 0.330/0.642). The prior generalization proof shows curved `m>=2` generator charts are not identifiable from scalar Y, and the raw query gate includes chart discretization.
- **Supersedes:** The earlier same-day decision to keep strict `raise` as the grid default; that mode remains reproducible on request. This is an explicit grid protocol/ESTIMATOR variant change, not a production ADP default change.
- **Affected:** `experiments/manifold_grid.py`, grid tests, `experiments/README.md`, manifold routing, `PLAN.md`, and `STATE.md`; previous output artifacts remain unchanged.
- **Status:** active for the repaired grid; broad recovery remains an empirical question.

## 2026-09-28 — Use only identifiable scalar-response targets for curved manifold

- **Decision:** Keep scalar Y. For curved m=1, assess the one-dimensional local chart away from stationary points; for flat m=1/2/3, assess the fixed global EDR space. For curved m=2/3, assess containment of the observable gradient in the estimated local space and keep generator `row(Dz)` distances descriptive, with no full-chart recovery flag.
- **Reason/evidence:** The smooth gauge change `z(x) -> R(x)z(x)` preserves `f(x)=||z(x)||²/2` at every x but changes `row(Dz(x))` for m>=2. The user explicitly selected an honest scalar-response criterion. The raw nearest-chart query gate also includes nonzero discretization error with exact center charts.
- **Alternatives rejected:** Claim full rank-m curved chart recovery from scalar observations; redefine a failed generator-chart gate as success.
- **Affected:** New curvature diagnosis/validation protocol, result interpretation, and `PLAN.md`; production estimator and old artifacts remain unchanged.
- **Status:** active.

## 2026-09-28 — Retain unit-spectrum localization as a flat broad variant

- **Decision:** Keep `localization_spectrum="unit"` opt-in. It is supported for the frozen broad flat m=1/2/3 protocol but is not promoted as a general default.
- **Reason/evidence:** Complete development/validation paired runs: broad/unit recovered 30/30 flat validation fits without numerical failure; broad/relative recovered 14/30. Local/unit recovered only 19/30 with five failures, and the curved development gate failed. The complete run manifests match all 11 live source hashes.
- **Alternatives rejected:** Change the production default or claim curvature repair from the flat result.
- **Affected:** Experiment interpretation and `docs/experiments/manifold_generalization_2026-09-28/repair.md`; no production code change.
- **Status:** validated only in the frozen flat broad setting.

## 2026-09-28 — Stop the curved scalar-response estimator search at the failed gate

- **Decision:** Do not promote the current manifold estimator, its `unit` spectrum option, or the Gaussian-Stein/quartic prototype as a stable curved m=1/2/3 recovery method. Keep the scalar-response criterion as all-center RMS and maximum gradient-containment sine <=0.2; full curved `row(Dz)` for m>=2 remains a descriptive generator distance.
- **Reason/evidence:** The finalized 240/240 development rows and matching seven source hashes show broad/unit 60/60 numerically completed fits but 0/60 gradient-containment passes. The original broad graph with exact source gradients also passed 0/60; local/unit had 21/60 rank failures. The one isolated polynomial candidate computed 12/12 preflight profiles and passed 0/12. The smooth gauge example proves full chart nonidentifiability for m>=2 under scalar `f=||z||²/2`; `tests/test_manifold_identifiability.py` checks the counterexample numerically. See `docs/experiments/manifold_curvature_2026-09-28/report.md`.
- **Alternatives rejected:** Count low solver residual or `scale_boundary="stop"` as geometric recovery, weaken the all-center maximum gate after observing failures, infer a general curved method from flat validation or the mild d=4 curved test, or run untouched curved validation after failed preflight.
- **Affected:** Curved-manifold scientific claim and experiment routing only. Production estimator and defaults stay as recorded; the opt-in flat broad `unit` result remains valid for its frozen profile.
- **Status:** closed for this bounded hypothesis; further recovery requires a new identifiable model/signal assumption and a new selection protocol.

## 2026-09-28 — Isolate the SVD.tex rank-r multi-index step

- **Decision:** Implement the fixed-g rank-r objective as an opt-in CPU `ADP_solver` method. Return a full m-row public basis by retaining B's recovered right singular directions and completing them from the projected prior basis; refit local coefficients before the existing spectral canonicalization.
- **Reason/evidence:** `SVD.tex` penalizes `||B-P||²` and constrains `rank(B)<m`, unlike live HPAO's correction penalty and full-rank output contract. The rank-r matrix alone cannot identify all m EDR directions. Small dense reference checks establish the v solve, adjoint, objective descent, and conditional scale stationarity; a public m=2/r=1 fit completes.
- **Alternatives rejected:** Replace default LSMR, treat arbitrary SVD null-space vectors as recovered EDR directions, or claim the greedy rank-r result is a global optimum.
- **Affected:** `ADP/solver/SVD.py`, `tests/test_svd_solver.py`, `agent-notes/ADP/multi-index.md`; existing default solver and API unchanged.
- **Status:** active experimental variant.

## 2026-09-29 — Use a certified direct v solve for small positive-ridge SVD problems

- **Decision:** In the opt-in SVD solver, use bounded Cholesky normal equations when `lambda_penalty>0` and `d<=128`; retain strict flat LSMR for larger `d` and nonpositive ridge, with explicit normal-residual certification and LSMR fallback. Keep adaptive Krylov tolerance opt-in; use an objective-preserving warm start on the iterative path.
- **Reason/evidence:** The user-shaped fixed-g d=400 case improved from 1.0372 to 0.6263 s median under strict LSMR. At the heavy `(Jp,d)=(8000,50)` geometry, direct/LSMR kernel time ratio was 0.58. Three full heavy seeds completed in 24.92/25.10/25.20 s with projector distances 0.11831/0.08987/0.09291 versus old 0.11831/0.08987/0.09507 and zero direct fallbacks. Adaptive tolerance reduced time but worsened seed-2 projector distance to 0.12634. See `experiments/svd_hot_path_2026-09-28/` and `experiments/svd_hot_path_full_fit_2026-09-28/`.
- **Limit:** The direct threshold is empirical and conservative; the final small algebraic cleanup after full-fit timing has focused dense/public verification but no repeated full-fit timing. Peak RSS rose about 4–5 MiB in the heavy direct fit. Pyright did not finish within several minutes and was interrupted.
- **Affected:** `ADP/solver/SVD.py`, focused tests, SVD benchmarks and route notes; the default ADP estimator remains unchanged.
- **Status:** done for this bounded optimization task.

## 2026-09-29 — Add low-rank correction as an explicit SVD mode

- **Decision:** Keep `low_rank_target="matrix"` as the existing default and add `low_rank_target="correction"` to `solve` and `solve_fixed_coefficients`. The shared factor loop uses `I-forward(U,P,g)` as its data base and zero ridge target in correction mode; `rank=0` returns P. The public correction result QR-orthonormalizes `P+Delta` and refits local coefficients.
- **Reason/evidence:** `SVD_corr.tex` changes the rank constraint from `rank(B)<=r` to `rank(B-P)<=r`, an ESTIMATOR variant. Shared `_FlatU`, direct/LSMR v solves, certificate, QR/SVD compression and scale re-fit avoid duplicate numerical paths. Nine focused tests pass, including dense objective, conditional scale derivatives, direct/iterative certificates and public multi-index fits. The bounded three-seed fixed-g comparison is recorded under `experiments/svd_correction_2026-09-29/`.
- **Alternatives rejected:** Silently change the old SVD default, overwrite the default HPAO LSMR solver, or duplicate the entire greedy loop. High reference correction energy `E_2=0.884–0.929` did not imply greedy optimality: rank-2 correction objective improved over matrix on only two of three tested seeds.
- **Affected:** `ADP/solver/SVD.py`, focused tests, benchmark, and SVD routing notes. Stochastic sampling remains outside this deterministic implementation.
- **Status:** active opt-in variant; no general recovery or speed claim.

## 2026-09-30 — Add bounded opt-in Jacobi scaling to the SVD v subproblem

- **Decision:** Implement `precondition_v=True` as a NUMERICAL option, default False, for positive-ridge iterative v solves. Scale both data and ridge blocks by the inverse square root of the normal diagonal, preserving the objective, regularization center and warm start. Keep original-coordinate certification and strict unpreconditioned fallback. Disable scaling at zero ridge to preserve minimum-norm semantics; retain the small-d direct path. Cache center-wise column energy within 16 MiB and stream larger cases.
- **Reason/evidence:** The simplest applicable preconditioner in `SVD/SVD_solver.tex` passed independent algebra/adjoint audit, dense references, 59 shared tests and paired selection/untouched validation (144 rows each). On validation column-scaled fixed-g tasks, median time/pass ratios versus the frozen baseline were 0.2255/0.2381. Full-fit projector quality changed by at most 2.27e-8, outer step counts matched, and no new numerical failures occurred. See `experiments/svd_improvements_2026-09-30/REPORT.md` and raw manifests/rows.
- **EXACT companion change:** Fuse augmented LinearOperator callbacks by directly using `_FlatU` actions. This applies by default and preserves the mathematical operator; validation isotropic fixed-g median time ratio was 0.9430, with no traced-memory regression beyond the frozen bound. Reject nonfinite certificates/norms explicitly.
- **Alternatives rejected:** Scalar damping after right scaling would change the penalty; scaling zero ridge can change the minimum-norm solution. Dense d*d preconditioners, GPU, block updates and new search strategies add complexity beyond this task. Existing ADP default LSMR, rank semantics and estimator remain unchanged.
- **Protocol boundary:** Initial full-fit n220/d150 settings were infeasible before solver execution. Preserve all 18 refused rows and the failed initial summary; corrected n320 selection used fresh fit seeds120..122 before validation220..222. No solver/gate tuning followed the protocol error. The result establishes bounded numerical equivalence/performance, not new recovery or global optimality.
- **Verification limit:** Ruff/format/diff checks passed; configured Pyright timed out after 180s, exit124. Bounded baseline checks also did not finish, so full type checking is unverified.
- **Affected:** `ADP/solver/SVD.py`, `tests/test_svd_solver.py`, SVD experiment artifacts and routing notes, `PLAN.md`, `STATE.md`.
- **Status:** done for this bounded optimization; diagonal scaling remains opt-in.

## 2026-09-30 — Record preconditioner behavior on the 25-second Spokoiny fit

- **Decision:** Keep the existing `direct_max_dimension=128` behavior and the opt-in preconditioner unchanged. On the saved full-fit task (`n=800,d=50,m=2,N_J=800`), compare the same three seeds with the flag off/on using isolated one-thread processes.
- **Reason/evidence:** Median paired on/off time ratio was 1.0022 (+0.22%); projector error and outer-step count matched in all three pairs. Diagnostics show zero LSMR iterations and 1977/2407/2238 direct solves. Since `d=50` is on the direct Cholesky path, the experiment does not exercise Jacobi scaling. See `experiments/svd_preconditioner_spokoini_2026-09-30/REPORT.md`.
- **Limit:** This is evidence about the option's no-op on this direct-path workload; it does not assess preconditioning for high-d iterative LSMR fits. Process RSS is not an isolated solver-memory measure.
- **Affected:** Experiment interpretation and future workload selection only; production solver/API unchanged.
- **Status:** done.


## 2026-09-30 — Keep spectral SVD proximal metrics experimental

- **Decision:** Keep explicit `metric_power=0/.5/1` and `metric_floor` in the opt-in truncated-SVD solver. Preserve Frobenius at p=0 and do not change defaults.
- **Reason/evidence:** Whitening has an exact dense reference and 91 focused/regression tests pass. Selection p=1 improved the paired quality median under both rank constraints, but held-out correction failed the frozen gate: median projector-error delta +0.0000703 and worst +0.01061. Matrix mode passed this small pilot, but there are only three seeds per size and all held-out p=1 fits failed the inner convergence criterion. Details and raw runs: `experiments/svd_a_metric_2026-09-30/REPORT.md`.
- **Limit:** This is a bounded SVD-solver pilot, not a recovery theorem or global rank-r optimization result. Whitening stores a second U-sized tensor. Pyright did not finish within 30 seconds.
- **Affected:** SVD solver, current multi-index metric context, dense-reference tests, experiment report and routing notes.
- **Status:** done; option remains experimental.

## 2026-09-30 — SVD gradient pairs screened by exact decrease

Active: explicit `rank_one_search="gradient"`, default `"alternating"`.
Class APPROXIMATE; preserve the fixed-g matrix/correction objective and metric
whitening. Use unprojected Q=-grad(F)/2 SVD pairs, exact R²/D ranking and R/D
step, shared compression/scale refit; skip conditional a/v solves. This is a
finite candidate heuristic, not joint L/R optimization or a global optimum.
Do not interpret zero v residual as a certificate: applicability is false.
A useful candidate without rank growth remains a heuristic stop of the
existing greedy loop. Dense gain/derivative, anisotropic ordering, rank stops,
metric whitening and public-fit tests verify this boundary. No default
promotion; pilot evidence is in experiments/svd_gradient_2026-09-30/.

## 2026-10-01 — Unify EDR/SVD through profiled operators and reduced geometric coordinates

- **Decision:** Use separable operator least squares with variable projection and reduced coordinate maps as the common mathematical framework. Keep fixed-g matrix-rank, correction-rank, profiled Grassmann and spatial-field estimators distinct. Treat tangent SVD as commuting plane rotations of one fixed step; model spatial low rank before retraction rather than claiming low Tucker rank for the resulting basis tensor.
- **Reason/evidence:** Allowed seven-note SVD corpus and original VARPRO/Grassmann/GROUSE/weighted-rank/CP-Tucker literature. The standalone synthesis has short proofs and 203 independent checks in 35 groups. A three-seed, seven-repeat frozen angle-value comparison gives total time ratios .160–.167 versus batched augmented SVD including the shared projection, with curve discrepancy <=1.7e-16. This is evidence for scalar profile evaluation, not a full optimizer or recovery guarantee.
- **Alternatives rejected:** Ordinary SVD for general summed-Kronecker Hessians; identifying fixed-g ambient optimization with profiled subspace optimization; covariance Sigma/N for general kernel weights; low-rank spatial factors without spatial coupling/integrability assumptions; claiming global angle minimization or full sequence convergence from a local formula/descent bound.
- **Affected:** SVD/EDR_unified_theory.tex, SVD/theory_2026_10_01/{REPORT.md,check_formulas.py,formula_results.json,angle_microbenchmark.py,angle_results.json,build_audit.json}; affected TeX/multi-index routes. No production solver changes. Prior unfinished joint optimizer scope saved verbatim in PREVIOUS_PLAN.md/PREVIOUS_STATE.md.
- **Status:** active theory/design decision; proposed optimizer integrations and recovery improvements remain unvalidated.

## 2026-10-01 — Explicit profiled Grassmann solver with reduced full-core GN

- **Decision:** Add opt-in CPU `ADP.solver.grassman.solve` using full residual
  Jacobian in leading horizontal gradient subspace, all q*m core coordinates,
  polar retraction and true-profile Armijo. Keep Schur rank-one and spectral
  adaptive-q turns as ablations. Default local_ridge=0; lambda is damping only.
- **Reason/evidence:** EDR_unified_theory eq:dg retains coefficient cross terms;
  small independent lstsq/finite-difference tests pass. This bounds the dense
  solve by q*m instead of m*d. Cache operator actions only within frozen basis.
- **Alternatives rejected:** Dropped T.T*r term, row-independent solves, permanent
  coefficient/chordal ridge or replacement objective would change the target.
  Full ambient GN needs more operator passes; current low m favors small core.
- **Guards:** No full-rank Jacobian at deficient/ill-conditioned tau=0; value
  rank-one fallback. Explicit workspace fallback to spectral; no certified
  stationarity at rank boundary. Final certificate uses live U.
- **Affected:** ADP/solver/grassman.py; tests/test_grassman.py; EDR theory I/II.
- **Status:** active; performance/recovery advantage pending paired benchmark.

## 2026-10-01 — Five-step Grassmann budget for outer warm starts

- **Decision:** Set Grassmann default max_steps=5, retain configurable larger
  budgets and explicit converged/gradient diagnostics. Select core_gn5 for
  the measured heavy outer workload; no unconditional method recommendation.
- **Evidence:** Three paired seeds on n800/d50/m2 full162 steps: core5 times
  31.72/33.48/34.03s vs SVD36.32/40.74/37.52s; median paired ratio .874.
  Projector errors .05417/.04728/.04494 vs .11831/.08987/.12586.
  Core20 was slower than SVD (ratio1.260) and recovery slightly worse on all3.
- **Limits:** Both budgets exhaust final inner cap and fail stationarity;
  core5 gradients .0188/.0282/.0335. Three reused ablation seeds are preliminary,
  not untouched-seed validation. RSS +12–14MiB; objectives differ from SVD.
- **Affected:** grassman default, explicit-budget paired artifacts under
  experiments/grassman_2026_10_01/benchmark/fullfit_core5 and pilot_after_adapterfix.
- **Status:** active; larger standalone frozen solves may need larger budget.

## 2026-10-01 — Accept bounded Grassmann implementation and measured I/II evidence

- **Decision:** Keep explicit Grassmann variant and five-step outer budget;
  retain rank-one Schur, fixed/adaptive spectral and larger core caps for audit.
- **Evidence:** 18 independent numerical tests, 110 focused regressions;
  lead verified Luna raw paired results and source hashes. Schur rank-one
  paired speedup5.72x vs refit, projection-inclusive61-angle curve23.2x,
  max absolute error3.41e-13. Full core GN frozen .0949s vs spectral .2157s
  at similar loss; adaptive rank showed no useful advantage on this point.
- **Limits:** Three seeds, differing SVD objective, all core calls capped;
  five-step gain12.6% costs ~13MiB processRSS, no certified inner stationarity.
  p10<d+1 makes exactQR compression irrelevant. Retained harness failures
  distinguish setup errors from numerical failures. No default solver promotion.
- **Timing audit:** Historical SVD25.734 vs current34.859 has identical work
  diagnostics/output; Python/compiler changed and old BLAS unrecorded. Do not
  infer algorithm regression or a proven environmental cause from these runs.
- **Affected:** REPORT.md/VERIFICATION.md in experiments/grassman_2026_10_01,
  benchmark harness/results, solver/default budget and maintained routes.
- **Status:** active; heldout validation is follow-up, not claimed here.

## 2026-10-01 — Separate numerical CPU optimization of Grassmann

- **Decision:** Preserve grassman.py as small auditable reference; add
  grassman_optim.py with unchanged optimizer/default budgets/tolerances.
  Use reorthogonalized local QR only for safe m<=2; retain original SVD for
  sensitive/small/rank-deficient/large-m systems. Full augmented GN remains
  direct lstsq; no normal-equation optimization or estimator change.
- **Evidence:** Luna17 independent/parity/FD tests pass before last projection
  slice; frozen5-step paired gain~1.7x with projector differences<2e-15.
  Gradient adjoint reassociation .24ms vs1.13ms; transpose GEMM .33ms vs1.08ms
  (copy-inclusive final measurement pending). Original flat GEMM has no gain.
- **Memory/invalidation:** Gradient scratch capped4MiB using center chunks;
  only frozen-basis caches and per-solve invariant scalars/root(mass).
  Wide GEMM copies only (J,p,k), never another U-sized representation.
  Existing core workspace guard and live final certificate preserved.
- **Rejected:** Reduced iterations, relaxed tolerances, damping-only normal
  equations, unchecked analytic Gram inversion would confound CPU comparison
  or weaken numerical correctness.
- **Affected:** grassman_optim.py; tests/test_grassman_optim.py; new benchmark
  and experiment directory; ADP solver/multi-index/source routing notes.
- **Status:** active; heavy paired final comparison pending.

## 2026-10-01 — Accept Grassmann CPU optimization with measured trajectory limit

- **Decision:** Keep separate grassman_optim.py and untouched reference; accept
  EXACT/NUMERICAL CPU changes after35 optimized numerical checks and127 shared
  passes, final paired fullfits and common-input ill-conditioned diagnosis.
- **Evidence:** Final BLAS1 heavy Spokoiny n800/d50/m2/J800/p10: median paired
  runtime ratio.941939 (5.8% reduction), frozen core5 speedup2.5204x, final
  projector differences<=6.90e-10 and profile objective differences<=3.49e-8.
  Every seed has162 outer calls/810 accepted steps, all capped/unconverged.
  GN/rank fallback counts match; profile evaluation counts differ (roundoff).
- **Limits:** Seed2 transient profile loss difference3.436 (10.4%) at call106
  comes with differing outer inputs; common-input cross-solves differ only
  <=1.66e-8 in objective and<=2.30e-8 in projector. Incoming basis3.60e-8,
  U8.60%/I12.99%/mass2.19%; precise upstream amplification unisolated. No
  identical trajectory/convergence/recovery improvement claim. Fresh-process
  RSS delta median+92KiB, range-5872 to+7040KiB; no stable memory gain claim.
- **Scope:** Original cProfile solver share~16.3% of instrumented fullfit;
  local statistics dominate. Do not expand this task into upstream changes.
  Wide GEMM and QR measured on this BLAS/workload; m>2 retains SVD.
- **Affected/evidence:** grassman_optim.py; experiments/grassman_optim_2026_10_01/
  {fullfit_final_hash,frozen_final_hash,call106_diagnostic,REPORT,VERIFICATION};
  earlier and invalid-derived benchmark artifacts retained with correction notes.
- **Status:** active; documentation/static review completion pending.

## 2026-10-02 — Use response-anchor dMAVE discretization for package comparison

- **Decision:** Represent dMAVE in the benchmark by fitting package CSMAVE to
  Gaussian response-kernel pseudo-responses at five response quantiles and
  aggregating the fitted subspace projectors. Keep the implementation isolated
  in the R benchmark worker and label it as an anchor discretization.
- **Reason/evidence:** MAVE 1.3.12 has no dMAVE method selector. Its
  MEANMAVE-on-density-feature pilot recovered poorly on three paired seeds
  (median projector overlap below 0.5); a five-anchor CSMAVE projector
  aggregate recovered overlap 0.9990 on the frozen seed-0 input. The dMAVE
  literature defines the kernelized response `K_h(Y-t)` as the MAVE response.
- **Alternatives rejected:** Calling a nonexistent package method; treating
  MEANMAVE on one n-dimensional response vector as an adequate dMAVE fit after
  poor observed recovery; silently substituting MEANOPG.
- **Affected:** `benchmarks/adp_mave_methods.R`,
  `benchmarks/adp_mave_benchmark.py`,
  `experiments/adp_mave_comparison_2026-10-02/` (pilot), and the final
  `experiments/adp_mave_comparison_2026-10-02_anchors_final/` directory.
- **Status:** active benchmark-only approximation; not a production estimator.

## 2026-10-02 — Keep rMAVE as a separate benchmark implementation

- **Decision:** Add a benchmark-only refined-MAVE worker initialized by the
  package's CSOPG fit. Each iteration uses Gaussian weights in the current
  projected space, fits local intercepts/gradients, updates a shared ambient
  basis by weighted least squares, then orthonormalizes and shrinks bandwidth.
- **Reason/evidence:** The installed MAVE package does not expose an rMAVE
  selector. The implementation preserves the defining projected-space
  refinement while keeping the package call and custom steps explicit. On the
  three-seed benchmark it completed all fits with median overlap `0.9974` and
  median fit time `0.893 s`.
- **Alternatives rejected:** Claiming a built-in package rMAVE call or
  substituting CSMAVE without the reduced-space refinement.
- **Affected:** `benchmarks/adp_mave_methods.R`,
  `experiments/adp_mave_comparison_2026-10-02_anchors_final/`.
- **Status:** experimental benchmark implementation; no production change.


## 2026-10-02 — glm.tex full tangent GN remains research-only

- Task: glm-grassmann-integration-2026-10-02.
- Decision: retain the mathematically checked matrix-free profiled full GN
  in experiments/glm_grassmann_2026_10_02/full_gn.py; do not change public
  grassman/grassman_optim or run held-out validation after failed R3.
- Evidence: 40 focused tests, targeted Ruff/Pyright; 30 complete paired rows.
  Frozen cap-30 solves rejected certificates 45/45 and cost 11-20x extra;
  all 27 adaptive inner calls rank-guarded. Full-fit time ratio 1.2143,
  max projector regression 5.51e-8; quality/time/advantage gates fail.
- Scope: this rejects the bounded unpreconditioned candidate, not the full
  tangent GN algebra. Preconditioning/enrichment/active-center rank treatment
  need a fresh hypothesis/proof and seeds. Positive local_ridge, kappa/GLS
  and chordal penalty are estimator changes, not equivalent speed patches.
- Sources/artifacts: SVD/glm.tex; PROOF.md, protocol.json, runs.jsonl,
  pairs.json, gates.json under experiments/glm_grassmann_2026_10_02.

## 2026-10-02 — joint angles in one cached Grassmann flat

- Task: grassman-flat-2026-10-02; user explicitly requested implementation
  and GPT-6 Luna testing/benchmarking.
- Decision: add explicit APPROXIMATE `method="flat_gn"` to grassman_optim,
  preserve existing core_gn default and profile. Optimize all q angles of
  one SVD-selected flat jointly with full coefficient derivative and
  exact cached Exp/profile; use existing QR/SVD, augmented LS and Armijo.
- Reason: several small angle refinements reuse U[Y,V] without repeated
  ambient passes. q unknowns suffice for this fixed flat. A q*m polar core
  or one scalar spectral path solves a different constrained subproblem.
- Boundaries: local box and iteration cap, no global torus/recovery promise;
  unsafe rank -> value-only rank_one; memory/no descent -> spectral;
  local_ridge keeps its estimator meaning. Heldout depends on PLAN gate.
- Evidence/verification route: experiments/grassman_flat_2026_10_02/PROOF.md,
  tests/test_grassman_flat.py, benchmarks/grassman_flat_benchmark.py.
  Numerical checks/benchmarks owned by Luna; results pending at this checkpoint.

## 2026-10-02 — joint flat verified, selection fails full adaptive ADP

- Task: grassman-flat-2026-10-02; completes the preceding implementation entry.
- Evidence: Luna45 tests and targeted Ruff/format/Pyright pass;120 unique,
  error-free hash-paired rows. Frozen24 flat/core pairs: objective ratio0.586,
  gradient0.606, time2.557; all q>1,741 inner accepts, zero fallbacks.
- Fullfit6 pairs: median time1.8555 fails<=1.10;4 projector deltas exceed
  +1e-4, max+0.071155; RSS passes. All inner calls capped/unconverged.
  Fullfit final orthogonality not serialized; mark that benchmark criterion
  unverified, independently tested at solver level. Do not rerun to conceal
  the already decisive negative quality/time gates.
- Decision: retain explicit experimental flat_gn, preserve core_gn default;
  stop validation/promotion, leave heldout74200-74202 untouched, no retuning.
  Frozen profile improvement does not establish adaptive EDR recovery benefit.
- Artifacts: experiments/grassman_flat_2026_10_02/{PROOF,REPORT}.md,
  freeze.json, baseline_profile.txt, raw runs/protocols and gates.json.
  Root independently verified row uniqueness/input/code hashes and gate count.

## 2026-10-03 — verified Grassmann core promotion (active)

User requested current most effective verified Grassmann as grassman.py and
experimental modes in grassman_exp instead of grassman_optim. Promote existing
EXACT/NUMERICAL CPU optimization, preserve core_gn default and existing original
settings/budget/objective. Evidence: grassman_optim_2026_10_01 REPORT/LEAD_AUDIT,
frozen2.52x and fullfit median0.942 with close final projectors. This is scoped
measured evidence, not a universal optimum claim. Failed-selection flat/full
tangent GN stay explicit in grassman_exp (solve/solve_full_gn); shared kernels
come from grassman. Original SVD reference moved to experiment reference.py;
no persistent grassman_optim alias. Executable callers migrated; historical
series labels and raw artifacts retained. 51 tests and120 exact relocation
pairs passed; report in experiments/grassman_promotion_2026_10_03/REPORT.md.

## 2026-10-03 — core-only production API (active; supersedes selectable production modes)

User clarified that grassman.py must expose only the most effective verified
version. Remove method/angle_backend selectors and refit ablation from its
solve; alternatives remain in grassman_exp. Internal rank-one/spectral guards
are part of core_gn and must stay to preserve numerical stability and memory
limits. Explicit user authorization overrides selector compatibility. 51 tests
and72 exact core comparisons (including rank/workspace guards) pass; benchmarks
migrated, numerical outputs/diagnostics unchanged. See core_only_parity.json.

## 2026-10-06: production Grassmann stage profiling

Use timer-only semantic stage timings as primary evidence for this workload;
cProfile adds24–41% solver overhead and remains supplementary for hotspots.
Local profile/refit is largest on all3 Spokoiny seeds (pooled47.3%).
Rank fallback prevalence materially affects shares; do not generalize across
solvers/dimensions or infer an optimization without paired correctness checks.
Evidence: experiments/grassman_stage_profile_2026_10_06/report.md and lead_audit.json.
Production algorithm/API/estimator unchanged. Status active measurement policy.

## 2026-10-15 — structured EDR candidates remain opt-in (complete)

- Task edr-structured-2026-10-15, user requested Schur/CX, recycling, metric
  SVD/descent and exact VARPRO Newton/TR, with Luna tests/benchmarks.
- Add separate grassman_research entry point, statistic_operator and small
  Schur kernels; production grassman/core_gn and defaults stay unchanged.
  CX preserves finite-sketch statistics; bounded optimizer variants target
  the same profile. Positive local_ridge stays an explicit estimator setting.
- Choose full q*m solver chart for direct core_gn comparison; lower-level
  SchurCore supports A=(m,a), inactive elimination and q*a charts separately.
  Newton is exact only in the chosen chart, without a full Grassmann Newton or
  quadratic convergence claim. Preconditioner changes CG only. Recycling
  never certifies stationarity without a fresh full ambient gradient.
- Evidence: main 30-step synthetic suite has no certified endpoints. Separate
  near-truth 100-step calibration certifies all six methods on three seeds;
  candidates are slower (Schur GN~1.9x, Newton~4.6–5.4x). Do not promote or
  retune on these seeds. Schur inactive kernel can benefit repeated small
  trials, which does not establish whole-solver speed.
- CX backend stays explicit: sparse support8/d100 storage .625x dense U,
  support400 storage6.505x and slower applications; setup must be counted.
  No automatic backend policy inferred from this small experiment.
- Independent review found stale Schur screening after a conditioning guard;
  two-case RED regression established the issue, lead clears core before
  value-checked metric fallback. Final evidence lives in experiment REPORT,
  raw results and Luna test logs, linked from STATE/PLAN.
- Artifacts: experiments/edr_structured_2026_10_15/{PROOF,REPORT,README}.md,
  protocol.json, results/; tests/test_edr_{operators,schur,research}.py.

### Same-task supplemental evidence

- A separately frozen weak-signal calibration (second planted coefficient
  scaled .03, three seeds) supports a narrow positive metric result:
  3/3 certificates versus core_gn2/3, paired speedup1.662x on two certified
  pairs, projector median9.00e-6 versus9.22e-4. Keep as a three-seed pilot;
  it does not override default/no-promotion decision or establish full-fit
  recovery. Earlier weak_index case is only an anisotropic proxy.
- Corrected Schur microbenchmark changes K each trial: evaluate+HVP .175–.188ms
  versus cached polar/full refit .810–.851ms, setup~.76ms. The initial ~10x
  same-K result reused Schur factors and is retained as cache-biased evidence,
  not a new-trial performance claim. Distinct-K benefit is about4.5x locally.
- Recycling reduces median full adjoints31->23 and projects32->24 in primary
  Newton runs, but produces no overall wall-time gain. Preconditioning reduces
  total CG iterations789->763 (unpre789, pre763), with
  no wall-time gain; pass/iteration reduction alone is insufficient.

## 2026-10-15 — hard Spokoiny metric study: keep validation untouched

- Task metric-spokoiny-2026-10-15 uses original hard _make_problem point
  n800,d50,m2,J800,P10, original data/seed/init/weights/schedule, solver tau0.
  Existing metric search is an APPROXIMATE optimizer, no estimator/default
  change. Separate full-fit, frozen-objective and truth-recovery conclusions.
- Freeze PROTOCOL.md before timing: development0/1/2, explicit budget5/20,
  rank1/2 and frozen isotropic/spectral-matched controls. Held-out3..7 require
  per-seed nonregression plus a predeclared speed or recovery improvement.
- Original budget5 main9fits completed. Metric2 seed1 projector error .0478484
  vs core .0472818, regression .0005666 >1e-4 allowed. Gate fails; preserve
  seeds3..7 untouched regardless of other development results. Continue only
  already-frozen diagnostic/budget experiments, without retuning/promoting.
- MATH.md separates coefficient Gram R from ambient covariance and outer
  localization eigenvalues. For exact generator the population OPG is
  (5557/1155)I2 (lead rational proof, independent Luna proof/quadrature), so
  the earlier artificially weak-coefficient pilot is a different geometry.
- Initial frozen90 launches failed before numeric work (path/snapshot key
  mismatch). Preserve original manifest/errors and successful cold fullfit/
  capture rows; repair only launcher in a separately manifested result tree,
  with independent argv-route regression. This is harness repair, not a
  changed solver or a valid negative numerical result.
- Evidence: experiments/metric_spokoiny_2026_10_15/{PROTOCOL,MATH}.md,
  results/fullfit5.jsonl, capture.jsonl, frozen.jsonl; subsequent repaired
  artifacts carry their own manifest and import provenance.

### Final hard Spokoiny evidence (active, no promotion)

- Complete9 budget5 fits,4 diagnostic captures,90 repaired frozen solves,
  6 budget20 fits. Independent lead audit verifies18 source hashes,14
  byte-identical imports, paired input hashes, unique jobs, one BLAS thread,
  monotone frozen objectives and max orthogonality error6.22e-15. Only run.py
  changed between original/repaired manifests. Ten focused tests pass.
- Metric2/core paired median whole-fit ratio.981649 at5steps,1.015490 at20.
  Seed1 five-step projector regression5.666e-4 fails selection. Keep heldout
  seeds3..7 untouched; do not retune or promote from development results.
- Five jointly smooth-certified frozen pairs have median time ratio2.039975;
  recovery/objectives nearly coincide. Budget5 has no certified endpoints.
  Rank1 does not establish a stable gain. Separate lower iteration counts
  from actual solver/fit costs and frozen Phi from truth recovery.
- Nine sampled input R condition numbers1.057..4.801, floor/caps inactive.
  Common mass multiplication by c preserves tau0 minimizer but divides
  metric proposal by sqrt(c). Matched isotropic control removes per-call
  spectral magnitude difference, not trajectory or diagnostic overhead.
  No stable recovery advantage from orientation is demonstrated.
- Supplementary seed1/call160 cProfiles: metric36 accepted vs core48,
  but249/50 profile calls and247/48 polar calls. Extra trial work explains
  cost on this example; profiler timings are not benchmark evidence.
  Proposal scale calibration and cheaper profile trials are untested future
  hypotheses, requiring a new protocol; no optimizer changes in this study.
- Evidence: experiments/metric_spokoiny_2026_10_15/{REPORT,MATH,LEAD_AUDIT}.md,
  results_repaired/{lead_audit,proposal_audit}.json and profiles/.

## 2026-10-16: explicit block-rank experiment, existing mechanisms retained

- Active APPROXIMATE variant: grassman.block_rank_k forwards into experimental
  shared loop; production solve signature and numerical path unchanged.
  Spectral already supplies common-step rank-k geodesic; flat_gn supplies joint
  angle optimization. Reuse both instead of duplicating their mathematics.
- New degrees of freedom are fixed/energy/bounded-cost rank policy and two
  joint angle steps by default. Cost policy has at most two pilot evaluations
  once per solve; includes measured overhead and may be timing sensitive.
- Fresh development81600..81602/validation81700..81704; retain regressions and
  compare complete EDR fit against core and spectral. Validation will run for
  a frozen candidate even after a negative development result, explicitly as
  negative-result validation under this new user's requested experiment.
- Exponential error law is conditional on unestablished smoothness/PL/step
  assumptions; numerical F_ref is no global-minimum proof. Full EDR recovery
  has priority over frozen objective. No default promotion authorized.
- Evidence: experiments/grassman_rank_k/{PROTOCOL,MATH}.md; independent refit,
  finite-difference and geometry tests. Timings/results pending at decision.

## 2026-10-16: rank-k validation candidate frozen after development

- All54 development fits completed with paired input/first-stat hashes and
  controlled threads. All four optimized rank policies eligible on pooled
  EDR/completion; none meets .8 fit/core gate. Predeclared selector chooses
  block1 (median paired fit/core .9991409237). Selection pins development
  SHA588f87e083fb8edf40f2610958c2fb19bf2482d0669ec43349c34214f8c85bee.
- Heldout30 will validate this unchanged candidate plus core/spectral. No
  retuning. MI original case has rank loss4..9/250 at initial statistics;
  core/optimized use rank guard on all30 updates/fit. Keep case and disclose
  that it cannot distinguish joint-angle policies; spectral has a distinct
  unguarded path and worse EDR. Do not silently regularize to avoid guard.

## 2026-10-16: rank-k full-fit hypothesis not confirmed; no default promotion

- Completed84 full fits,792 inner workers and5 independent reference polishes.
  All workers ok, geometry/profile monotonicity/pairing/source hashes pass.
  71 focused tests pass; unrelated full-suite catalog collection failures remain.
- Frozen block1 heldout fit/core median .99963, bootstrap95%[.98914,1.01554];
  no20% gain. Median EDRdiff0 but one+.000120 regression. Selection of k1
  does not establish simultaneous multiple-direction recovery advantage.
  Preserve experimental explicit API only; production core_gn unchanged.
- Spectral already implements common-step block update; flat already jointly
  optimizes angles. Reused mechanisms cannot count as novel algorithmic gains.
  Extra block GN/refits and cost pilots do not amortize in measured full fits.
  Original MI rank-guard case retained; no statistics/ridge modification.
- One-step F improves in all200 adjacent-k comparisons; approximate exponential
  trend predicts heldout subset, with task-dependent c. No global guarantee:
  smooth nonoptimal stationary counterexample exists. Numerical Fref has22/24
  stationary certificates after separate independent polish, never global proof.
  Preserve340 lower-F/worse-EDR pair comparisons and failed recovery cases.
- Evidence: experiments/grassman_rank_k/REPORT.md, results/{selection,
  fullfit_summary,inner_descriptive,lead_audit}.json; analysis_polished and
  analysis_certified CSV/plots; source_snapshot.zip and reproducibility manifest.
  New tuning needs fresh protocol/seeds; current heldout has been consumed.

## 2026-10-17: experimental adaptive centers P0-P3

Active. All features use explicit AdaptiveCentersConfig; defaults/solver remain
unchanged. P0 and center selection are ESTIMATOR; P1 is exact compact-support
search modulo roundoff; projected moments preserve current weights. Preserve
full-pool alpha, initialization, per-center sketches and step-error population.
Use normalized logdet(I+sum uuT/eps), because the TeX uncentered logdet ratio can
have a nonpositive denominator. Cap min centers by pool size. Lazy greedy
guarantees only the fixed-vector surrogate; scalar PoU mass conservation is
not operator equivalence or recovery proof. Sources/evidence: experimental
geometry/centers/step modules, tests/test_adaptive_centers.py (24 passed),
experiments/adaptive_centers/MATH.md. Luna runs fresh paired full-fit experiments
before any claim of empirical benefit; no default promotion in this task.

## 2026-10-17: adaptive-centers development rejected global promotion

- Completed Luna120 paired fullfits(4cases,3seeds,10arms) and24 localization
  workers under separate frozen protocols. Inputs/init/threads/source/mapping
  audits pass. Every fullfit reaches h_min; zero inner convergence certificates.
  All global development gates fail; selection=null,heldout92900..4 untouched.
- P0 floor25 median paired EDRdiff0,time.994baseline; additive30 median
  .4190->.2486 but other cases do not support a general improvement. Floor50
  recovery2/12 vs baseline1/12 but pooled paired EDR+.00150. Do not promote.
- P1 fullfit/floor25 time1.682 atn500; P2 moment diagnostics add1.179time
  without EDR change. P3 LogDet+coverage reduces J500->60/80/100 but baseline
  EDR+.03968,time2.487,peakRSS1.191. Vs equal-J random EDR-.03773,time1.338;
  surrogate selection improves that control but does not justify replacement.
- P1 localization atn10000,d100,J500,h.4 yields projected/baseline time.493
  weights,.414alpha+weights; exact support/weights/alpha on3 paired seeds.
  Common D excluded from timer and retained in memory; CSR output common.
  This supports workload-dependent localization benefit only, not faster fits
  or lower memory than native streamed baseline. No automatic dispatch added.
- Same-state P1 I/U agree<1.4e-15 but one fullfit EDR differs-.01036 under
  adaptive iterative sensitivity. Exact geometry is not bitwise trajectory
  equivalence. Singleton eta diagnostic can differ1 and is unused by solver.
- Post-timing seed=None regression fixed using root SeedSequence entropy.
  For frozen integer seeds streams identical; two current-source fullfit
  hashes replay. Original snapshots immutable; compatibility overlay/patch
  AST audited and saved separately. Final27feature/86sharedtests pass;
  known unrelated full-suite catalog/manifold failures remain.

## 2026-10-17: paired ADP versus R SDR protocol

- Active. Isolate experiments/r_sdr from production ADP and historical benchmark
  wrappers. Compare ordinary ambient-kernel MAVE separately from refined MAVE
  and compiled MAVE::MEANMAVE. Classical SIR uses covariance whitening and fails
  explicitly on singular designs; do not relabel package KSIR as classical SIR.
- Installed MAVE1.3.12 has no dMAVE/rMAVE selectors. Use auditable R references
  instead of relabeling CSMAVE or averaging separately estimated projectors.
  Density MAVE fits one joint shared-B objective over all observed responses;
  reference variant explicitly has fixed response bandwidth and no trimming.
  Streaming QR/response-space compression checked against dense stacked LS;
  no ridge, normal-equation shortcut or silent target subsampling.
- Freeze oracle rank/config/seeds/source/environment before fits; paired float64
  X/Y hashes, original-coordinate subspace metrics, >=10 attempts per cell.
  Keep failures in recovery denominators; pair only jointly valid same-seed
  results and disclose pair counts/conditional inference. Fit and process time
  separate; sampled /proc peak includes runtimes and has a polling boundary.
  Resume skips all recorded outcomes, including failures, rejects protocol drift
  and runs frozen sources. No automatic hyperparameter tuning by true basis.
- ADP inner solver certificates do not establish outer convergence. Protocol
  schema2 records all_inner_certified separately and outer converged=null.
  Do not pool initial schema1 validation with schema2. Finite R stopping criteria
  and package-unknown convergence remain explicit, separate from EDR recovery.
- Evidence: tests/test_r_sdr.py(12passed),tests/r_sdr_reference.R; review fix
  confirmed; validation-final120/120 valid fits plus replay/hash/metric/resume
  audits in benchmark_outputs/r_sdr/validation-final/verification/. Full3000 and
  stress360 catalogs unrun. No comparative speed claim or default promotion.
- Evidence:experiments/adaptive_centers/{REPORT.md,verification.md},supplement/
  REPORT.md,results/selection.json,verification/{lead_audit,trajectory_audit,
  compatibility_overlay}.json. No solver/default promotion or heldout tuning.

## 2026-10-17 — Compare Grassmann directly with MAVE references

- **Decision:** Add `ADP-Grassmann` only as an explicit method in the isolated R-SDR runner, and compare it with ordinary MAVE and package `MAVE::MEANMAVE`. Use Grassmann as the paired-report baseline and `local_ridge=0`.
- **Reason/evidence:** The live Grassmann solver has the current `HPAOResult` contract and the same unpenalized finite-sketch local-refit profile at zero local ridge. Its documented five-step warm-start budget is the prespecified setting; it is not semantically equivalent to LSMR's 80-step cap.
- **Alternatives rejected:** Replacing public ADP defaults or including HPAO-LSMR in the requested three-method comparison would change the requested scope. Do not claim matched iteration budgets or pool with the completed LSMR run.
- **Affected:** `experiments/r_sdr/{design.py,__main__.py,runner.py,worker.py,report.py,README.md}`, `tests/test_r_sdr.py`, `PLAN.md`, `agent-notes/ADP/r-sdr-comparison.md`.
- **Status:** active pending the paired benchmark result.

### Outcome — 2026-10-18

- **Evidence:** `benchmark_outputs/r_sdr/multi-rank-grassman-2026-10-17/` recorded 90/90 valid fits, zero failures and 10 paired seeds at each of ranks 3, 4, and 5. Ordinary MAVE's paired median time was 25.14/32.70/35.74x Grassmann; MEANMAVE was 1.481/1.202/0.9816x. All paired EDR-delta intervals include zero and no fit met the recovery threshold.
- **Solver caveat:** all 150 inner calls reached the five-step cap; 745/750 accepted steps used the rank-one fallback; only one call reached core GN; all outer schedules stopped at `h_min`. This is a comparison of the configured Grassmann entrypoint, not evidence of converged Grassmann fits.
- **Validation:** 16 focused tests passed, scoped Ruff/format and `git diff --check` passed. The full suite has three unrelated collection errors caused by missing catalog key `mi-spokoini-m2-n`.
- **Status:** complete; no production default or estimator change.

## 2026-10-18 — Add Grassmann to the main R-SDR catalog comparison

- **Decision:** Run the full 50-case R-SDR catalog for `ADP-Grassmann`, ordinary MAVE, and `MAVE::MEANMAVE`, with 10 paired seeds per case (1,500 fits) and the already documented five-step Grassmann budget.
- **Reason:** The user's request is for Grassmann in the main benchmark alongside the two MAVE references. Keeping the same full catalog/seed protocol gives paired evidence across its broad `n,d,m` range while avoiding unrelated methods and preserving a separate artifact.
- **Alternatives rejected:** Reusing the small `m=3,4,5` run or rerunning all six default methods would not answer the requested focused comparison.
- **Affected:** `PLAN.md`, `agent-notes/STATE.md`, `experiments/r_sdr/README.md`, new benchmark output directory.
- **Status:** active until 1,500-fit run is analyzed.

## 2026-10-18: reject ten-step localized Grassmann accuracy candidate

The frozen R-SDR development run completed90/90 paired fits without failures. EDR geometric ratio .9093 passed quality, but median/p90 time ratios1.368/1.729 failed the fixed1.10/1.25 gate. No production/default promotion; heldout92800..92809 stays untouched. Saved sources/journal: benchmark_outputs/r_sdr/grassman-accuracy-dev-2026-10-18/. Next independent hypothesis uses only exactly-zero operator centers at five steps and fresh development seeds.

## 2026-10-18: reject constant-center accuracy candidate

Exact constant-center elimination at five steps passed time/RSS but EDR ratio.9997287 failed frozen.95. 90/90 fits complete and hashes match; validation94800..94809 untouched. Retain experiment without production promotion. Next hypothesis is direct true-profile Armijo to spend saved line-search time on rank-one iterations, using fresh development95800.

## 2026-10-18: reject direct Armijo accuracy candidate

Repaired development90/90 successful; EDR ratio.955496 failed fixed.95, time.95545/1.12086 passed. Preserve worst EDR ratio1.9680 at d20/95809. Original metadata parser failure fixed separately with regression coverage. No production promotion; heldout96800 untouched. Next independent hypothesis localizes QR/SVD profile fallback while retaining nonsmooth guards and accurate scalar search.

## 2026-10-18: reject quadratic accuracy candidate

90/90 successful paired fits; EDR ratio.96196 (95%CI[.92384,.99003]) failed fixed.95 despite fit median/p90.8229/1.09843. Preserve observed3.8% improvement/17.7% speed gain without claiming the frozen gate passed. Validation98800 untouched. Next distinct direction hypothesis uses regular-center GN with full true-profile acceptance for numerical intrinsic-rank centers; mathematical caveat in INTRINSIC_RANK.md.

## 2026-10-18: reject intrinsic GN after independent validation

Development99800 passed all gates (EDR.89504,time.66773), but unchanged heldout100800 failed EDR.98546 CI[.87116,1.18394]. A10.069x EDR regression at d5/100803 is material despite finite/orthogonal successful output. First candidate inner profile50.6433 versus baseline8.16165 suggests insufficient initial progress, not numerical success. Exact basin cause is unproven. Keep production unchanged. Next distinct candidate preserves one original bounded scalar first step before intrinsicGN; new101800/102800 data, same gates plus2xoutlier bound, no tuning on consumed validation.

## 2026-10-19: reject one-step intrinsic warmup

Complete101800 development90/90 fails EDR1.01325 CI[.83220,1.34436] and individual28.7518x at d5/101809; time.7551/.9099 passes. All paired/source hashes match, numerical certificates do not establish recovery.102800 heldout untouched. Keep full artifacts/defaults. Next hypothesis preserves the ENTIRE original five-step solve, then adds bounded intrinsicGN polishing; budget chosen only on90800, fresh103800/104800 selection/validation.

## 2026-10-19: close baseline-first polish pilot

Original5+intrinsicGN extra2/3/5 passed7 reference tests but90800 pilot total7/8/10 yields EDR.971/.969/.947 and time1.069/1.118/1.123; d50budget10 costs1.50x. No joint quality/time budget apparent from this exploratory input; do not claim a paired-selection rejection or promotion.103800/104800 still untouched. Next candidate conditionally reuses the existing joint spectral gradient geodesic at rank guards; smooth GN remains unchanged, no rank classifier/full-rank derivative at deficient centers.

## 2026-10-22: freeze conjugate rank-guard budget before fresh data

Pure spectral90800 pilot has no joint quality/time winner and is closed at exploration. PR+ projection transport adds bounded O(dm+J) history with rank-change and actual-descent resets; it remains an APPROXIMATE optimizer with full true-profile trials/live certificates.31 focused tests and independent Luna contract review found no blocker. Conjugate90800 budgets5/10/15 give EDRgeo1.38123/.94045/.96922 and median time.55456/.92183/1.28427. Freeze10 steps (initial smooth cap5) for103800 development/104800 validation under unchanged quality/time/RSS/2x-outlier gates. CONJUGATE_PROTOCOL.md; no production promotion from this pilot.

## 2026-10-22: close conjugate development on time, inspect exact cost next

Complete103800 development90/90 confirms EDRgeo.92707 CI[.88901,.96282],26/30wins and no severe outlier (max1.12455x). Time median/p901.13220/1.39751 fails fixed1.10/1.25; no promotion/validation.104800 unused. Preserve accuracy evidence without describing this as an accepted quality/time tradeoff. Next bounded measurement asks whether exact contractions/caching can retain the direction with lower cost; no budget retuning on consumed103800. New candidate would need fresh105800/106800 data.

## 2026-10-22: close conjugate QR cost pilot; investigate relative GN damping

Even after guarded QR factor reuse,90800 QR-CG is1.093..1.128x slower than CG.
Close before fresh selection; no speed claim from isolated profile improvements.
Retain numerical helper/tests and immutable pilot artifacts. New distinct
APPROXIMATE proposal normalizes intrinsic-GN damping by ||J||F^2/(q*m), preserving
conditional response/mass scale invariance. All-center acceptance, rank guards,
smooth paths and final certificates remain original. No data-driven quality
claim yet;105800/106800 remain untouched, fixed gates unchanged.

## 2026-10-22: freeze relative intrinsic GN at ten steps

64 reference/scaling/fallback checks pass; exploratory90800 EDRgeo at5/10/15
is1.06879/.95377/1.04254, median time.48757/.60728/.70264. Freeze10 before105800
development as the best pilot quality with time headroom. The reused pilot does
not establish the .95 gate; all unchanged frozen quality/time/outlier gates
must pass on105800 and then untouched106800 before acceptance. Full sources
and math retained; original smooth GN/public solver unchanged.

## 2026-10-22: relative intrinsic GN passes development; validate unchanged

Corrected105800 stage completes90/90 with30matched pairs; all8 frozen gates pass.
EDRgeo.898376 CI[.844245,.948903], paired median/p90time.650580/.790378,
RSS1.00524, worst1.07731x. Initial invalid launch preserved separately; duplicate
runner lambda forwarding fixed without prototype/production changes or tuning.
Now run unchanged106800 heldout; no acceptance/default promotion from dev alone.

## 2026-10-22: accept relative GN as an explicit measured-workload option

Unchanged106800 validation90/90 passes all8 gates: EDRgeo.923729
CI[.886622,.956495],time.653572/.993297,RSS1.00211,worst1.03642x; source sets
equal development. Accept explicit relative_gn option at10steps/smooth cap5 for
n500,m2,d5/20/50. Preserve original default. Port must reproduce prototype and
saved oldbaseline outputs on all60 dev/heldout datasets before completion.
No universal recovery/default-promotion claim; R still better in case medians.

## 2026-10-22: finish public relative GN port and direct R confirmation

grassman.solve(relative_gn=True,max_steps=10) ports the accepted algorithm in
99added/8removed lines; defaultFalse. All120 dev/heldout full-fit replays match
saved bases and every inner diagnostic exactly, including60 oldbaseline fits.
Current public/R confirmation on the SAME106800 data completes90/90 with all8
gates PASS: EDR.923729,time median/p90.649100/.944153,RSS1.00170,worst1.03642x.
This is a code/timing confirmation, not additional independent datasets.
209 focused tests pass; Ruff/format/Pyright clean. Current production SHA256
8804e23a16c4cf2de65d9c7a0fddc20b03fe6156d19bc3f8888bff2f607d782a.
RELATIVE_GN_REPORT.md and immutable raw snapshots retain limits and all failures.
Task done for measured n500,m2,d5/20/50; broader default promotion requires new
evidence. No convergence claim or claim of beating R MEANMAVE accuracy.

## 2026-10-22: user-requested combined Grassmann default and budget20

Active; supersedes the earlier decision to keep relative_gn=False as the
public default. User explicitly requested combining earlier joint time/EDR
gains below old promotion thresholds and then raising compute. Default solve
now enables relative intrinsic GN, direct/quadratic3 true-profile rank-one
Armijo and exact zero-U reduction, max_steps20 with no initial-smooth cap.
Same finite-sketch estimator, rank cutoff and live active-objective certificate;
exact constants restore full coefs/loss/rank counts/gradient normalization.
Earlier QR/SVD/gradient/GEMM/Jacobian optimizations were already present.
CG, scalar warmup, flat/full tangent/rank-k and per-center QR had no joint
full-fit win and are not silently combined. Legacy flags remain selectable.

Evidence: COMBINED_{MATH,PROTOCOL,REPORT,VERIFICATION}.md;24 exploratory fits
only on90800 before budget freeze. Fresh107800/108800 stages150+150 success,
30paired datasets/stage, same sources/settings, actual R MEANMAVE. Final20
validation versus original5: EDRgeo.925311 CI[.861556,.983259], time median/p90
.860129/1.189895, RSS1.00403; all8 historical reference gates pass. However
versus accepted relative10: EDR1.003418 CI[.986369,1.028379],time1.275697.
Extra budget20/combined10: EDR.996743 CI[.974926,1.023747],time1.306993.
Thus no established incremental recovery improvement; default20 follows the
explicit user request, not an invented quality win. R remains more accurate
on case medians. No outer convergence/full50-case generalization claim.

Public/control replay120/120 EXACT bases/diagnostics/stops;247 focused tests,
Ruff/format17 and Pyright14 pass. Whole suite has7 unrelated failures and3
collection errors reproduced with old solver; names/raw evidence retained.
Historical dynamic-clone scalar candidate and polish now pin old settings so
new public defaults cannot silently change those experiment meanings.
