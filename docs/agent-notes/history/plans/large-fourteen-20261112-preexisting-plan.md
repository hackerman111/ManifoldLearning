# Oscillatory frequency sweep: seven functions

Task oscillation-frequency-20261112; status **done**. User requests deeper
oscillatory analysis with seven frequency coefficients. Execute in current
checkout, preserve previous uncommitted work; no commit or solver/default edits.

**Goal:** Compare public ADP Grassmann and package MEANMAVE on seven fixed
f_a(z)=sin(a*z1)*cos(1.25*a*z2)+sin(1.5*a*z3), a=.5,1,2,3,4,6,8.
15 paired seeds0..14 per frequency, 210 fits; explain observed recovery limits.
**Non-goals:** Tuning, solver changes, sample-size/noise/rank sweeps, general
claims over all oscillatory links, or changing frozen earlier outputs.
**Evidence:** Prior high-frequency f8 failed recovery in both methods (EDR
means .9107/.9261). Previous seven-family campaign is complete and audited;
39 focused tests passed. Prior plan/state preserved under agent-notes/history.
**Hypotheses:** Increasing frequency loses local signal coherence, but low
frequency can also have weak population gradient directions. These are distinct
from optimizer stationarity; quantify them separately without claiming causality.
**Change class:** Explicit new experimental data-generating variants (ESTIMATOR);
shared-RNG extension changes only the new opt-in profile. No estimator changes.

**Invariants:** n2000,d20,m3, Gaussian X, signal centered/std(ddof0) normalized,
noise std.3. Identical B/X/noise for a given seed across all frequencies and
methods; data_key=family-oscillatory reproduces the previous f8 inputs. Float64,
one BLAS thread, cold sequential fits, limits300s/4096MiB. Methods exactly
ADP-Grassmann-combined20 and MAVE-package; oracle rank3, EDR<=.1 recovery.
No post-result changes to frequencies or settings. Failed fits stay visible.

**Exact read set:** AGENTS.md; prior PLAN/STATE; agent-notes README, WORKFLOW,
contracts/research.md; benchmarks/experiments/r_sdr/{design.py,__main__.py,
runner.py,protocol.py,README.md}; tests/test_r_sdr*.py; previous campaign
{manifest.json,REPORT.md,audit.py,make_report.py,completion.json} and selected meta.

**Bounded work units / acceptance evidence:**
- [x] Add opt-in frequency/data-key fields and oscillation-seven profile, plus
  exact population gradient moments in oscillation.py. Tests first: seven
  coefficients, cross-frequency shared inputs/noise, f8 legacy equivalence,
  finite-difference gradients and Gaussian quadrature reference; old bytes unchanged.
- [x] Document formula/protocol, run focused tests/lint, freeze and execute210
  fits with keep-inputs. Preserve all raw data and snapshots.
- [x] Independently audit completeness, hashes, regenerated inputs, bases and
  EDR; quantify direction alignment, population gradient eigenvalues/mean,
  signal coherence in fixed20-neighbor true-coordinate neighborhoods, phase
  differences, paired frequency changes and random-subspace reference.
- [x] Produce detailed Russian report and standalone plots; review, final
  verification, checkpoint plan/state/decisions and affected experiment routes.

**Stop conditions:** Degenerate generator, mismatched shared data, snapshot
mismatch, occupied output or missing dependency stops launch. Fit errors/timeouts
stay recorded; infrastructure interruption resumes unchanged. No frequency
selection based on observed EDR. Do not infer an ADP-specific failure mechanism
from population cancellation or oracle-neighbor diagnostics alone.

**Final evidence:** benchmarks/results/outputs/r_sdr/oscillation-frequency-20261112/
contains completion.json (210/210, zero errors), audit.json (105pairs, shared
inputs/noise,123 source hashes, exact regenerated inputs, max EDR error1.11e-15).
a8 prior payload hashes/projectors reproduce exactly. REPORT.md, population/
direction/local-resolution/paired-frequency CSVs and PNG/SVG figures are saved.
47 focused tests passed (verification.log); ruff/diff checks passed. Independent
source/mathematics review passed; final report aggregate overstatement was
corrected with explicit a2 per-seed exceptions and partial plane recovery.
Main evidence: recovery only MAVE a1 (15/15); a2 loses interaction directions;
a>=3 near random EDR. Oracle coherence at a2/a3 is still high, so density alone
does not explain the failure. ADP0/525 inner certificates, all max_steps.
No estimator/default changes, no tuning, no commit.
