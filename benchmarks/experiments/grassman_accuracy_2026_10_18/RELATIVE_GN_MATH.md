# Relative damping for intrinsic GN proposals

Class: APPROXIMATE optimizer; the estimator and all-center objective stay fixed.
Scope: only the existing eligible regular-center intrinsic GN proposal, tau=0.
Smooth GN, numerical rank cutoffs and final certificates retain their semantics.

Let J have shape (regular_centers*p,q*m) and r be its weighted profile residual.
Set mu=lambda_prox*||J||_F^2/(q*m), then solve the existing augmented system
min_k ||J k+r||^2+mu||k||^2. No normal matrix is formed. For positive response
and mass rescaling I'=cI and mass'=a mass, coefficients/residual scale by c,
J'=c sqrt(a) J, r'=c sqrt(a) r and mu'=a c^2 mu. The entire small quadratic
scales by a c^2, so its minimizer is unchanged in exact arithmetic. With mu>0
it is unique. The norm cap and polar retraction consequently give the same
subspace; the full gradient/slope/objective scale by a c^2, preserving Armijo.
The full-gradient SVD may change basis in repeated singular spaces; the
isotropic damping is invariant to orthogonal coordinate changes there too.

This is conditional proposal invariance, not a global convergence statement.
Branch history/stopping can change when max(1,sum mass*I^2) crosses its unit
floor, or under finite precision/rank decisions. Numerical intrinsic rank
classification is approximate. Omitted centers are exactly locally constant
only when rank(U_j Y)=rank(U_j)<m holds in exact arithmetic. The original
all-center gradient, trial profile, rank guards and live final profile remain
mandatory; their values cannot be replaced by the regular-center model.

Zero/nonfinite mu is not a positive damping parameter: skip that GN model and
use the existing spectral fallback. Do not add a statistical ridge or clamp a
tiny denominator. This can sacrifice scale invariance on under/overflow.
Positive mu needs no dense d*d matrix; the existing workspace check bounds
the rows-by-(q*m) augmented solve. Scaling adds only a Jacobian scalar reduction.

Checks: direct augmented least squares; positive response/mass scaling with
fixed stopping branch; full refit coefficients/objective/gradient; m=1/2/3;
smooth/default/ridge/workspace behavior; zero/nonfinite damping fallback.
Source: backtracking._intrinsic_model and core_gn branch, grassman._core_jacobian,
INTRINSIC_RANK.md. Algebra independently audited by Luna before implementation.
No assertion that relative damping improves full ADP EDR follows from this proof.
