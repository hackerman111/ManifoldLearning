# ADP moment integration

Task `adp-moment-integration-20261030`; status **done**.
Goal: separately test residual initialization, fixed-objective moment proposals,
then local quadratic ADP on fresh paired data. No production/default changes.
Design: experiments-only adapter around current fit_index and solver/statistics
hooks; process-local initializer patch restored on exit. No new dependencies.
Classes: residual-start and WLS2 are ESTIMATOR; proposals are APPROXIMATE for a
fixed inner objective, but change the adaptive finite-budget procedure.
Known evidence: weak-moment-20261030 concerns a different global estimator and
cannot establish any of these integrations. Prior plan archived next to this one.

Invariants: X(n,d), index(m,d), iid synthetic samples, m=2,d=20,n=2000,float64;
truth used only in evaluation; identical centers, direction seeds and first
weights for matched-start arms. No new Y residual in the ADP objective.
Proposals may change P only when the original *same frozen* U,I,mass profile
strictly decreases by >64 eps max(1,F). No monotonicity across outer windows.
Full numerical error certificate and general ADP recovery remain non-goals.

Read set: weak_moments.py; weak_moment_study.py:1-170;
local_polynomial.py:1-260; index_fit.py:90-190,194-445;
grassman.py:96-155,405-455; logger.py:IndexProfiler;
contracts/{research,numerics}.md; WORKFLOW.md.

- [x] R1 write scoped objective-invariance/proposal proof, complexity, exact gaps;
  independent adversarial review before selection.
- [x] R2 create experiments/adp_moment_integration.py, integration_study.py and
  tests/test_adp_moment_integration.py. Tests first: dense profile reference,
  proposal acceptance/rejection/rank/orthogonality, fixed-source/seed contracts.
  Small preflight only on seed3020000; no quality-driven tuning.
- [x] R3 freeze config/source/data hashes, then 8 selection seeds3020100:3020108;
  cases quad_strong,quad_weak,cubic,quartic,correlated (last is diagonal Gaussian).
  Six arms: adp(20), residual_start(20), budget(20+20), proposal(20+trial+20),
  wls1 and wls2 (same residual start, local ridge=1e-4,max_steps40).
  Proposal tries residual T2 complementary eigenvector, each row rotation by
  +/-pi/8,+/-pi/4,pi/2; chooses best strict decrease; then second solver call.
  Gate independently per contrast residual_start/adp,proposal/budget,wls2/wls1:
  all fits valid, paired geometric EDR ratio upper seed-cluster bootstrap95%<1
  on quadratic cases; median fit-time ratio<=3,RSS<=1.5; worst-pair EDR
  increase<=.25. 10000 seed-cluster draws fixed RNG3020999. Stress descriptive.
- [x] R4 validate ONLY passing contrasts, 8 untouched seeds3021100:3021108,
  unchanged settings/gates. Failed-arm validation stays untouched. Each phase
  <=15min, fresh sequential workers timeout120sec and RSS<=1GiB after fit.
- [x] R5 audit saved completeness, pairing, source hashes, metrics, profile
  nonincrease; report all convergence/budgets/time/RSS/failures. Update routes,
  PLAN/STATE/DECISIONS. No promotion without both gates; no retuning on failures.

Check core new tests plus weak_moments/local_polynomial; Ruff/format/Pyright;
small reference before campaign; no unrequested broad unrelated fixes/commits.
Stop an individual contrast on failed gate; independent contrasts may continue.

R1 independently accepted; 38 focused tests passed, Ruff/Pyright clean.
Ruling: default first weights are Euclidean independent of start; corrected
interpretation before freeze. The first-step contrast isolates start numerically
up to reduction roundoff; later trajectories remain adaptive.

R3: 240/240 valid fits, saved audit passes (metrics<=7.8e-16). Residual-start
ratio .9425 upper1.0305 FAIL; proposal ratio1.0034 upper1.0646 FAIL. Their
validation arms closed untouched, no tuning. WLS2/WLS1 ratio .34365 upper.39180,
time1.357,RSS.9996,worst delta-.2718 PASS. Validate ONLY WLS1/WLS2 unchanged.

R4: 80/80 WLS1/WLS2 validation fits valid. Ratio .44798, percentile
[.40309,.49591],time1.3207,RSS.9998,worstdelta-.09369 PASS. Mean EDR.5623→.2607,
recovery0/16→12/16. Anisotropic stress1.3372→1.3334 remains poor; all-inner
stationarity only1/16 WLS2 quadratic fits. No held-out baseline ADP contrast.

R5 complete: independent FINAL_AUDIT.md accepted both phases; 320/320 fits.
Source hashes/snapshot/data/pair/gate audits pass; metric error<=7.8e-16,
orthogonality<=6.3e-15; direct independent initializer projector<=3.76e-14.
Audits did not rerun iterative fits or reconstruct unsaved per-step operators.
38 focused tests passed; all six new Python files Ruff/format/Pyright clean;
git diff --check clean. REPORT.md, paired_edr.png/pdf and reproduction artifacts
saved; routes updated. Prototype remains opt-in, no production/default/API edit.
Ruling: baseline is explicit fit_index+Grassmann20, not asserted identical to
public wrapper defaults. O(Jn) instrumentation weights are documented.
