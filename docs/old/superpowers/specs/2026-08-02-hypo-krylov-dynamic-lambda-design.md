# Hybrid Krylov ADP with dynamic lambda

## Goal

Add an isolated ADP variant under `hypo/Krylov` that preserves the data,
statistics, bandwidth, direction, slope, and outer-iteration logic from
`hypo/single_index/ADP_single_index.py`, but replaces its fixed-lambda LSMR
beta update with Hybrid Golub--Kahan regularization and projected GCV from
`tex/hybrid_krylov_lambda.tex`.

The existing `hypo/single_index` implementation remains unchanged.

## Structure

- `hypo/Krylov/ADP_single_index.py` subclasses the working single-index class.
- `hypo/Krylov/tester.py` keeps the same synthetic benchmark and CLI surface as
  `hypo/single_index/tester.py`.
- `hypo/tester.py` discovers the new tester automatically as `Krylov`.

Subclassing avoids copying the existing 360-line ADP pipeline. The subclass
overrides only fit-time telemetry and the alternating beta solver.

## Solver

For fixed statistics and slopes, define matrix-free operations

\[
Av = \operatorname{vec}\{c_j U_jv\}_{j=1}^J,\qquad
A^\top z = \sum_j c_jU_j^\top z_j.
\]

Starting from the current normalized direction \(\beta_0\), form
\(r_0=b-A\beta_0\). A Golub--Kahan process builds \(V_k\) and the lower
bidiagonal \(B_k\) without forming \(A\), \(A^\top A\), or an augmented ridge
matrix. The process restarts whenever the slopes change; only \(\beta_0\) and
the previously selected lambda are retained.

At every fifth Krylov step, and at the final step, the solver:

1. computes the SVD of the small \((k+1)\times k\) matrix \(B_k\);
2. evaluates weighted projected GCV on 21 logarithmically spaced lambda
   values, using the full moment-row count \(m=JP\), equivalently
   \(\omega_k=(k+1)/m\);
3. uses the spectral interval
   \([10^{-8}\theta_1^2,10^2\theta_1^2]\), enlarged to include
   \([\lambda_{\mathrm{prev}}/100,100\lambda_{\mathrm{prev}}]\);
4. expands a boundary side by one decade once when the grid minimum lies
   there;
5. refines an interior grid minimum with bounded Brent optimization in
   \(\log\lambda\);
6. solves the projected ridge problem by the same small SVD.

The continuation value is frozen for all checkpoints of one Golub--Kahan
process. It is updated only after that process returns a valid candidate, so a
boundary minimum cannot enlarge its own next checkpoint interval.

The update is

\[
\beta_{\mathrm{new}} =
\operatorname{normalize}(\beta_0+V_ky_{\lambda,k}),
\]

with its sign aligned to \(\beta_0\).

The maximum projection length is \(\min(d,100)\). The process stops early
after two consecutive checks satisfying all three TeX criteria:

- change in \(\log_{10}\lambda\) below \(0.1\);
- projective direction distance below \(10^{-3}\);
- relative projected-GCV change below \(10^{-2}\).

Early stabilization additionally requires an interior lambda minimum.

## Edge cases and telemetry

- A zero residual keeps \(\beta_0\).
- A finite Golub--Kahan breakdown returns the last valid projected solution.
- Nonfinite inputs, factors, projected solutions, or a missing valid candidate
  raise `RuntimeError`; they are not reported as convergence.
- Each outer trace row records solver status, Krylov iterations, selected
  lambda, projected GCV, boundary status, sign-invariant beta change,
  cross-fitted moment loss, and whether the candidate was accepted.
- Timing uses a `krylov` field instead of the inherited temporary `lsmr` field.
- The selected lambda is exposed as `lambda_`.

The fixed `lambda_penalty` value inherited for constructor compatibility is not
used by this variant.

The reproducible upper-bound bias of ordinary projected GCV activates the
weighted criterion described in the TeX. For the strong local initialization,
candidate updates are checked by a deterministic two-fold observation split:
local slopes are estimated on one fold and moment residuals are evaluated on
the other, then the roles are swapped. The initial moment system remains a
fixed validation target and a candidate is applied only when this loss does
not increase. Random initialization uses weighted GCV without this early veto
because sparse cross-fitted moments otherwise prevent the exploratory steps
needed to leave a poor initial direction.

Krylov recycling and selection by the known synthetic truth remain excluded.

## Verification

The tester uses the same generated single-index data, arguments, cosine quality
threshold, timing checks, and weight diagnostics as the baseline tester. It
also checks:

- finite positive selected lambdas and GCV values;
- valid solver statuses and Krylov iteration counts;
- normalized finite output;
- a deterministic small projected-ridge calculation against a direct dense
  solve.
- weighted-GCV equivalence to the full-row denominator;
- continuation remaining fixed across checkpoints of one Krylov process.

Technical solver validity and the cosine quality threshold are reported
separately.
