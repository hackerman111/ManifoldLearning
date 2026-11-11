# Grassmann accuracy at bounded runtime

Task `grassman-accuracy-2026-10-18`; status **done (candidate rejected at R3)**.

Goal: improve Grassmann subspace recovery with little full-ADP runtime cost, using SVD drafts as hypotheses and the existing paired R MEANMAVE experiment as the benchmark.
Non-goals: alter local statistics, ridge, bandwidth, initialization, centers, or estimator; infer recovery from frozen objective; overwrite previous benchmark data.

Known evidence: production `grassman.solve` uses core GN with rank/workspace fallbacks and five warm-start steps; compact R-SDR fits predominantly use rank-one fallback. QR reorthogonalization, SVD profile, cached UY/UV, full core Jacobian and true-profile Armijo already exist. Several full/flat/block GN candidates failed earlier recovery/time gates. Prior main R-SDR plan is archived under `agent-notes/history/plans/grassman-full-rsdr-2026-10-18-interrupted.md`; its incomplete data are diagnostic only.

Hypotheses: (H1) rank guards may be triggered by objective-inactive neighborhoods, unnecessarily restricting direction choice; (H2) ill-conditioned local profiles or cached projection drift may impair direction accuracy; (H3) curvature-aware rank-one selection may improve progress at comparable cost. R0 must establish evidence before selecting a candidate.

Invariants: current finite-sketch profiled objective F(Y)=sum_j mass_j min_g(||I_j-U_j Y g||^2+tau||g||^2); orthonormal Y; tau=0 in benchmark; no hidden regularization; no dense ambient d*d matrix; identical data and ADP settings in paired fits; truth used only for evaluation. Change class determined per candidate: NUMERICAL for equivalent stable formulas, APPROXIMATE for direction/budget change. Isolate prototypes in experiments until frozen validation passes.

Read set: AGENTS.md; agent-notes/{STATE,WORKFLOW}.md; contracts/{numerics,research,engineering}.md; ADP/{solvers,multi-index,r-sdr-comparison}.md; SVD drafts selected by Luna; ADP/solver/{grassman,grassman_exp}.py; experiments/r_sdr/{design,worker,runner,protocol,report}.py; routed focused tests.

## Bounded work units

- [x] R0: Luna routes drafts and analyzes prior paired R-SDR evidence. Capture actual solver inputs on representative catalog fits; quantify inactive/rank-deficient neighborhoods, profile conditioning, cached-action error, gradient and stopping behavior. Save a compact diagnostic artifact and baseline source hashes.
- [x] R1: choose at most two candidates from R0; derive formula, finite-rank domain, complexity, and independent audit. Freeze development configuration and gates before running comparisons.
- [x] R2: isolated prototype; deterministic independent reference/finite-difference, rank-boundary, orthogonality, cached/live action checks; smallest relevant tests.
- [ ] R3: paired full fits through existing R-SDR data/ADP/R worker routes. Fresh development seeds 91800..91809; hold validation 92800..92809 untouched. Compare baseline/candidate/MEANMAVE, quality/time/RSS/failures and convergence separately. Select once; save hashes and all rows.
- [ ] R4: only a frozen R3 pass allows untouched validation with the same settings. Required: >=5% aggregate relative EDR improvement, no case-median EDR regression >0.005, median full-fit time ratio <=1.10, p90 <=1.25, no additional failures, bounded memory. Report pair counts and uncertainty. A pilot is exploratory and cannot support promotion.
- [ ] R5: integrate only a validated candidate, preserving a baseline route and API; focused checks, reproducible report, affected routing notes, decisions and final state. If a gate fails, preserve the experiment, leave production unchanged, and record the negative result.

Verification: independent small reference and derivative checks; focused Grassmann and R-SDR tests; one BLAS thread, float64; source/input fingerprints; per-case and paired recovery/time/RSS; scoped Ruff/Pyright and git diff --check. Runtime measures model.fit separately from startup; RSS is process peak.
Stop conditions: invalid derivation, unbounded allocation, mismatched paired hashes, source drift, failed frozen quality/runtime/failure gate, repeated rank-boundary numerical failure. Frozen candidate/cases/gates: experiments/grassman_accuracy_2026_10_18/PROTOCOL.md (local scalar fallback, ten steps; d-5,d-20,d-50); do not tune on held-out seeds.

R3 evidence: 90/90 successful fits, input hashes matched; EDR geometric ratio .9093, case median quality gates passed; median/p90 fit ratios1.368/1.729 failed. Validation92800..92809 was not run. Source and all outcomes are frozen in benchmark_outputs/r_sdr/grassman-accuracy-dev-2026-10-18/.
