# Lead audit

All numerical implementation was owned by the lead; GPT-6 Luna owned tests,
benchmark execution and an independent read-only code review.

- Scope: three new opt-in solver modules; existing grassman.py, SVD.py,
  statistic.py, adapters and model defaults unchanged. The normalized and
  legacy mass contracts are documented in README/PROOF.
- Independent reference checks cover CX and adjoints, Schur versus augmented
  lstsq/polar, finite-difference gradient/HVP, the note's -3 scalar HVP
  counterexample, metric whitening, indefinite trust steps, exact final
  ambient gradients, recycling, positive-ridge cases, API callbacks and
  rank/workspace fallbacks. Luna focused old+new Grassmann gate:73 passed.
- Independent review found stale Schur screening after a conditioning guard.
  Two-case RED reproduced zero accepted steps despite available descent;
  clearing core in fallback produced GREEN. No other substantiated finding.
- Source-level review checked the polar ridge metric, positive-only
  preconditioning, no d*d or (md)^2 matrices, cache lifetime within one frozen
  solve and mandatory full live final certificate. Newton is exact only in
  the chosen chart. Positive tau remains an explicit estimator setting.

## Raw evidence audit

Lead reopened JSONL and verified status/unique keys and per-input-group hashes:

| run directory | rows | unique | ok | hash-matched groups |
|---|---:|---:|---:|---:|
| frozen_candidates_tau0 | 45 | 45 | 45 | 9 |
| supplemental_local_calibration | 18 | 18 | 18 | 3 |
| weak_signal_calibration | 18 | 18 | 18 | 3 |
| cx_solver_parity | 12 | 12 | 12 | 6 |
| schur_kernel_distinct | 3 | 3 | 3 | separate kernel |

Every group above had exactly one set of input hashes. Each directory has
run_metadata.json. Failed harness attempts remain separate and are named in
REPORT. Primary and supplemental comparisons have distinct interpretation.

Audit corrections retained explicitly:

1. The first kernel reused identical K, caching Schur factors but recomputing
   the reference. Distinct-K timings now support ~4.5x only for small trials,
   with ~.76ms setup; the cache-biased earlier result remains preserved.
2. Old live_gradient_scaled normalized by the final objective rather than
   initial target energy. Certificates use raw gradient/target energy or the
   independently tested solver diagnostic. New rows record the correct scale.
3. Old weak_index is a mild anisotropic proxy. A separately frozen weak-signal
   calibration uses a planted second coefficient scaled .03; no retuning.
4. Whole solves are first-call timings after imports; operator/kernel trials
   are warmed. No unrun positive-tau stress is counted as benchmark evidence.
5. Final fallback fix only affects a smooth-profile/Schur-conditioning gap.
   Primary timed smooth rows had zero guard fallbacks; actual-stat guards
   came from nonsmooth profiles, where Schur was never constructed. Those
   pre-fix measurements are unaffected; raw source hashes are retained.

## Conclusions and verification limits

The weak-signal metric pilot is positive:3/3 certificates versus core2/3,
two jointly certified pairs give median speedup1.662x, projector median9e-6
versus9.22e-4. It is a three-seed local calibration, not held-out/full-fit
recovery evidence. Near-truth ordinary calibration favors existing core_gn.
Recycling reduces full operator actions without a measured wall-time gain;
Schur and CX benefits depend on dimension, repeated trials and sparse support.
Keep all candidates opt-in, with current production defaults intact.

Scoped solver/test Ruff and Pyright pass. Full pytest cannot collect three
catalog modules. Excluding those yields479 passed/7 failed/28 skipped; the
seven failures reproduce separately without new tests. Exact names/commands
and verbatim logs: test_results/full-suite-status.md. This is not a claim of a
green full repository suite. CUDA, positive-tau performance and held-out
statistical validation remain outside the completed benchmark evidence.
