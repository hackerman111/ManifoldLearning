# Conjugate direction with per-center QR/SVD profiles

This candidate keeps the conjugate10 optimizer, its gradient, rank resets,
geodesic, actual-slope Armijo test and five-step initial-smooth cap. Only the
local profile implementation changes to `selective.profile`, now reusing
the guard's QR factors instead of computing safe-center QR a second time.
Class NUMERICAL on top of the APPROXIMATE conjugate optimizer, no estimator
change. SELECTIVE_MATH.md supplies the existing per-center derivation.

Each local coefficient still minimizes
`||I_j-M_j g||^2 + tau||g||^2`, with M_j=U_jY. Safe centers satisfy the
production reorthogonalized-QR conditioning/absolute guard; other centers
use the original SVD/lstsq implementation and identical rank cutoffs.
Reassembly preserves center order, masses and the full residual/ridge sum.
Every trial and the final live UY refit use this same selected profile.

For a mixed batch, if the difficult subset is smooth, the helper delegates
the entire profile to production. Otherwise the original whole-batch SVD
would also be nonsmooth; production zeroes every normal_solve factor in
this case. Identity right factors and zero inverse_squared in the combined
profile preserve that action. No new full-rank derivative certificate is
introduced. All-safe batches retain the computed triangular R; all-unsafe
batches call production SVD directly, and m>2 uses the production profile.

QR and SVD can differ by floating-point reduction/conditioning error.
Therefore neither bitwise optimizer parity nor unchanged adaptive EDR is
claimed. Compare direct augmented-lstsq profiles, ranks, gradients, normal
actions, small full solves and frozen full fits; retain original applicability
and convergence flags. Existing selective stress/reference tests remain
part of acceptance. No Gram normal equations or dense ambient d*d matrix.

Temporary arrays scale as O(J*p*m), matching the existing selective helper;
conjugate history remains O(d*m+J). There is no cache shared between solves.
No gradient contraction is changed: profiling placed its total cost below1%.

Only exploratory90800 is available for the unprofiled prototype comparison.
Fresh105800 development and106800 independent validation use the same frozen
quality/time/RSS/outlier gates only if the pilot justifies proceeding.
Old103800 evidence cannot validate this numerical variant;104800 stays unused.
