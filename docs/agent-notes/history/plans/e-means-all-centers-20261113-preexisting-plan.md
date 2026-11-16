# e-means-inspired acceleration of the original anisotropic ADP

Task e-means-adp-20261113; status **done (negative research result)**. User requests a research attempt,
preservation of the anisotropic ADP, several distinct families, exact scoped
proofs, adversarial review, and honest reporting of any remaining gap.

**Goal:** Test whether mini-batch proposals, cosine EMA and full finalization
can improve EDR and/or time of public ADP-Grassmann combined20. Research spike;
no production promotion. Preserve existing dirty work and historical outputs.
**Non-goals:** Replace ADP with clustering/MAVE, alter weights/moments/bandwidth,
claim universal EDR recovery from profile descent, change defaults, commit.
**Evidence:** Paper arXiv:2611.09501v1 studies clustering MSE, not ADP/EDR.
Live ADP uses full-pool kernel moments and finite-sketch profiled least squares.
Existing large-fourteen background campaigns are separate; do not alter them.
**Hypotheses:** Compare EMA raw local moments; EMA anisotropic geometry;
stochastic frozen-profile optimization with full finalization; inactive-center
elimination. Main candidate is the third family, subject to mathematical audit.
**Change class:** APPROXIMATE optimizer of the same frozen profile. EMA of
weights/moments/outer geometry would be ESTIMATOR and is out of scope.
**Invariants:** Original public fit loop, centers/direction seed streams,
Epanechnikov K(u)=max(1-u^2,0) on the implemented squared-distance argument,
mass normalization, anisotropy/alpha/h schedule, initialization, rank handling,
outer best-step criterion, Grassmann objective and full final local refit.
No dense d*d state; float64, one BLAS thread; failures remain visible.

**Exact read set:** AGENTS/RTK; previous PLAN/STATE; agent-notes README,
contracts/{numerics,research,engineering}, WORKFLOW; ADP notes index-pipeline,
multi-index,adaptive-centers,r-sdr-comparison; index_fit.py:133-647,
statistic.py:30-254, grassman.py profile/solve/certificates; LSMR.HPAOResult;
r_sdr/{design,runner,worker}; tests/test_grassman.py; paper sections3 and H.

**Bounded units / acceptance evidence:**
- [x] R0: Baseline timing/stage profile and environment; four-family comparison
  with assumptions, operation/memory costs, counterexamples and exact gaps.
- [x] R1: Precise claims and proofs for the prototype; adversarial audit covers
  gauge invariance, truncation, rank boundaries and finite precision.
- [x] R2: Isolated module under benchmarks/experiments/e_means_adp; reference,
  invariance, degenerate and public-fit checks; no ADP source edits.
- [x] R3: Freeze protocol/source/input hashes before selection outcomes.
  Same-data paired original20, full-budget control, mini-batch and mini+EMA;
  record EDR, worst angles, fit/process time, RSS and solver certificates.
  Exact seeds/config/gates go in PROTOCOL.json before launch.
- [x] R4: Untouched validation only after a predeclared selection gate passes;
  otherwise close negative study with validation unopened and no retuning.
- [x] R5: Independently recompute saved metrics/pair completeness, write report,
  update routes/STATE/DECISIONS and mark plan done. Leave the experimental
  entry reproducible; a pilot does not authorize public/default promotion.

**Stop conditions:** Unsupported assumptions are not a proof. Stop a family
whose guarantee requires unchanged adaptive weights or an unproved EDR proxy.
A failed frozen gate closes that candidate. No tuning on held-out outcomes,
dropping failures, overwriting frozen artifacts or interference with other runs.
State resource bounds and theorem exclusions in the final report.

R0 evidence: baseline-profile.json records .558s fit, .293s solver (52.7%),
n800/d20/m3/J200/p12, RSS87.2MiB; single seed, diagnostic only. R1: MATH.md
compares four families, proves fixed-profile safeguard and PSD EMA identities,
states finite-precision domains/counterexample and exact remaining EDR gap.
Independent read-only reviewer accepted proof after damping/cost/indexing/
variance-assumption corrections. R2:49 focused new+Grassmann tests passed,
including independent dense EMA/refits, cutoff cases, RNG lifecycle, corrupt
proposal guards and frozen-worker/read-back tamper detection. Public ADP intact.

R3-R4 outcome:320/320 selection fits successful,80 pairs/arm, no missing rows.
Original mean EDR.408012; full6.453989; mini.439041; mini+EMA.437432.
Mini+EMA median time ratio.886412, mean EDR delta+.0294205, worst+.420027.
Both new candidates fail worst/per-case/bootstrap quality gates; full6 also
fails quality despite time ratio.662642. Independent audit passed5747 checks.
All1680 inner calls uncertified. Validation1132400..409 untouched; no retuning,
promotion or public-source edits. R5 only: report/independent claim review,
final checks, route/state/decision checkpoint.

R5 complete: REPORT/MATH/REVIEW/README and thematic route contain all scoped
claims, four-family costs/assumptions/gaps and the negative conclusion.
Independent final review found no blocking evidence defect; matrix-MSE wording
corrected to the proved variance identity. Audit5747 checks passes;1600 output
hashes unchanged; all68 public ADP source hashes identical. Final78 targeted
tests, Ruff, Pyright0 errors, formatting, links and diff checks pass. No claim
of a complete mathematical improvement solution; only the bounded research
attempt is complete. Validation remains unopened.

Prior plan/state archived under docs/agent-notes/history/{plans,state}/
e-means-20261113-preexisting-*.md.
