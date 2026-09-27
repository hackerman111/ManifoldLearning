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
