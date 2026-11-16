# Independent adversarial review, 2026-11-13

Read-only reviewer examined MATH.md, candidate.py, study.py, audit.py, relevant
public solver/fit ranges and tests. No production implementation was delegated.

Verified exact domains: gauge identity; low-rank projector EMA recurrence and
top-m projector approximation; input-relative full-value safeguard; full-rank
least-squares residual perturbation bound; explicit profile-descent/EDR
counterexample. These do not establish EDR/original20 superiority.

Corrections incorporated before frozen outcomes:

- J/b scaling changes absolute smooth-GN damping relative to the data and can
  change stopping; it is an explicit optimizer choice, not exact trajectory parity.
- Intrinsic-rank SVD cost included;16MiB is a GN allocation-estimate fallback
  threshold, not a process/RSS/LAPACK bound.
- EMA indexing fixed; endpoint schedule adaptation and deterministic weights
  in the conditional variance identity stated explicitly.
- Tests added around rank cutoff, two-call RNG replay, false minibatch
  certificates and polishing rollback. Fresh instance required per fit.
- Integrity audit now binds stage manifest to frozen protocol, independently
  recomputes gates/bootstrap, and rejects empty-manifest/rows, gate tampering
  and corrupted bases. All input hashes precede outcomes. Validation also
  requires independently audited passing selection.
- Completion separates execution_complete, selection_eligible,
  validation_passed/failed and validated_success; no success from launch alone.
- Public numerical rank-cutoff evaluator distinguished from an ideal exact
  minimum; no interval certificate for real-arithmetic descent is claimed.

Reviewer independently obtained49 focused passes and checked synthetic
two-case/ten-seed gate agreement. Final pre-freeze review found no further
blocking prototype/runner defect. Scientific outcomes were not seen by the
reviewer during design. Final outcome/claim review is recorded after the run.

## Final outcome/claim audit

Independent reviewer reran the frozen-source audit:5747 checks pass,320 valid
fits, no integrity failures. Independently checked all4 means/recovery/paired
timings, raw process medians,0/1680 certified calls,0/420 applicability for
each mini arm, EMA198/last222 starts, zero polishing rejections, exactly80
selection inputs, absent validation and negative completion. All68 public
ADP hashes match, with no Git source changes.

One wording correction was applied: independent unbiased errors give the
exact matrix-MSE identity, not unconditional reduction when variances differ.
MATH now also gives the explicit variances100/1 counterexample. Final verdict:
no blocking evidence defect; scoped theorem and remaining EDR gap correctly
stated. The negative conclusion was retained without retuning.
