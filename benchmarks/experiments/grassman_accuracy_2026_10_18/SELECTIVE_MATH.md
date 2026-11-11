# Per-center QR/SVD without relaxing rank guards

The profile objective is a sum of independent local least-squares minima.
The production QR sufficient condition is local, although its fallback is
whole-batch. Reuse exactly that condition per center; compute safe centers
with the original QR path and unsafe centers with the original SVD path.
Coefficients and residuals are reassembled in original order; sum the same
weighted residual/ridge expression in that order. No center is discarded.
The2026-10-22 revision reuses QR factors already computed for the guard.
Safe coordinates use Q.T times the same augmented target and the same
triangular back substitution, avoiding a second safe-center QR pass.
An all-safe state retains its actual triangular R for normal_solve. An
all-unsafe state delegates directly to the original SVD profile, since the
production QR guard would also fail. Floating-point reductions can differ.

When the unsafe subset is nonsmooth, the original whole batch is also
nonsmooth: safe QR centers satisfy the stronger conditioning guard. Thus
the global GN guard remains false and the original SVD normal_solve has
all-zero inverse_squared. Identity right factors with zero inverse_squared
give the same zero action in the combined state. If the unsafe subset is
smooth, use the original whole-batch profile and all its SVD factors. This
preserves the GN normal_solve contract without a new mixed factorization.
For m>2 or p<m retain the original path. Ridge augmentation and cutoffs are
identical to production, including its max(p,m) cutoff convention.

This is NUMERICAL: QR and SVD differ by roundoff and do not produce bitwise
identical optimization paths. Independent augmented-lstsq, gradient, ranks,
smoothness and normal_solve comparisons are in test_grassman_selective.py,
including actual frozen outer calls. Additional masks and copies are
O(J*p*m), with no ambient d*d allocation. An extra initial profile selects
the nonsmooth10/smooth5 budget in the historical scalar wrapper and is included
in its measured fit time. The conjugate_qr wrapper chooses the profile inside
the common loop and adds no separate budget-selection profile.

The accurate scalar curve uses the guarded per-center AngleProfile from
candidate.py/MATH.md. Original directions, value-based Armijo, GN and live
certificates remain. Increasing steps to ten is APPROXIMATE and requires
paired end-to-end recovery evidence; a smaller frozen loss is insufficient.
