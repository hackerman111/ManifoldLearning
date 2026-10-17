# Formula audit and implementation boundary

The local file `SVD/adaptive_centers_anisotropic_adp_edr_report.tex` is the
source of the proposals, not independently verified experimental evidence.
Its v2 kernel agrees with the live code: `K(q)=max(1-q*q,0)` and q is a
**squared** localized distance. This implementation retains that convention.

## P0: spectral floor (ESTIMATOR)

Raw canonical spectrum has maximum 1; initialization uses all ones. The
experimental localization uses `L=max(raw_spectrum,lambda_floor)` in both
weights and full-pool alpha selection. Raw published eigenvalues, canonical
rotation, direction generation and solver regularization retain their meaning.
Density multiplier is exactly 1. The optional report rule max(floor,alpha²)
is not implemented; the constant floor is the isolated hypothesis.

For row-orthonormal P and delta=X_i-center_j write z=P delta and
r=max(||delta||²-||z||²,0). Orthogonal geometry uses
`q=(sum L_r*z_r² + alpha²*r)/h²`; full geometry instead uses
`q=(sum L_r*z_r² + alpha²*||delta||²)/h²`. The first iteration retains the
existing orthogonal geometry. With a positive floor and q<1,
`||P delta|| <= h/sqrt(floor)`. A Taylor remainder bound using this radius
requires the current subspace to equal the true one; otherwise subspace
misalignment and the orthogonal radius must also be included. A smaller
radius can reduce curvature bias and increase variance. Recovery improvement
is a statistical hypothesis, not an algebraic consequence of the floor.

## P1: projected two-stage search (EXACT modulo roundoff)

Both geometries have a nonnegative residual contribution, so q<1 implies
`p=sum L_r*z_r² < h²`. A KD-tree on `(X-X[0])P.T sqrt(L)` returns necessary
candidates before reading full distances for those pairs. The query radius
is conservatively expanded for floating point error; p and residual are
recomputed in the original coordinate-wise reduction order. Final K(q)
removes extra candidates. Zero eigenvalues need no artificial floor: the
filter remains necessary but can return all pairs.

Candidate CSR stores principal/residual components at alpha=0. The existing
`_compact_alpha` bisection then uses these candidates, targets `J0*N_loc`, and
prunes only pairs that are zero throughout the remaining feasible interval.
The next step reuses the geometry constructed in the update stage: basis,
effective spectrum, h, training rows and pool are unchanged in between.
Roundoff and reduction-order differences are covered by direct tests; this
does not imply bitwise identical adaptive trajectories in all ill-conditioned
problems. No approximate nearest-neighbor search is used.

The existing initialization still allocates/caches D=(J0,n). New persistent
state is O(E0+n*m), transient weights O(E); row scratch is bounded by n*m.
No J*n*d or d*d localization array is introduced. KD-tree overhead and dense
support can make this path slower; timings include construction and alpha.

## P2: projected moments and prefilter

For every pool center and the exact current weights, compute weighted means,
`Sigma_Z=sum a_i (Z_i-Zbar)(Z_i-Zbar).T`,
`g_Z=sum a_i (Z_i-Zbar)(Y_i-Ybar)` and effective sample size. These are exactly
`P Sigma_X P.T` and `P c_X`. Their cost is O(E*m²+J0*m³), storage O(J0*m²),
and they precede full selected-center U assembly. A surrogate slope uses an
explicit `score_ridge`, separate from the production solver/local ridge.

Feasible means n_eff>=m+1 and smallest covariance eigenvalue greater than
eps*max(m,neighbor_count)*largest eigenvalue. This is a heuristic feasibility
filter, not proof that the full finite sketch identifies all EDR directions.
If too few feasible centers meet the budget/minimum, retain the complete pool
and expose `insufficient_feasible_full_pool`; zero full-pool mass is an error.
`selection=all, projected_statistics=True` measures pure overhead. `score`
keeps largest squared information-vector norm. Filtering is ESTIMATOR.

Information vectors follow the report confidence form, with explicit frozen
choices `kappa0=.01`, `varsigma=std(Y)`, N_min=N_loc and scale sqrt(J0/J_feas).
Zero response scale produces zero vectors. Full direction streams are still
generated on the entire pool to preserve the common sketch across variants;
the selection saves full local U and solver work, not direction generation.

## P3: LogDet plus mass coverage (ESTIMATOR hypothesis)

The report writes a ratio of uncentered log determinants. For eps<1 its
denominator can be negative or zero, and F(empty) is not zero. The code uses

`L(S)=logdet(I+sum_{j in S} u_j*u_j.T/eps)`

and `F(S)=L(S)/L(feasible)+nu*R_mass(S)` when L(feasible)>0, with zero
information contribution otherwise. This is an explicit correction. The
function is nondecreasing, not strictly increasing when gains vanish.

For fixed u the determinant lemma gives marginal
`log(1+u.T M_S^-1 u)` with M_S=eps*I+sum u*u.T. Cholesky triangular solves
evaluate it without an inverse. M_A<=M_B for A subset B gives decreasing
information gains. Coverage uses fixed pool observation masses and positive
support, so coverage gains also decrease. Lazy greedy is therefore applicable.
The fixed-cardinality approximation statement concerns this surrogate only;
the dynamic 96.5% information/95% active-mass stopping rule is a heuristic
for budget choice, not a minimum-budget or recovery theorem.

The minimum center count is `min(J0,max(4m,2d))`, explicitly capped by J0.
The report's `max(4m,min(J0,2d))` can exceed J0 in small pools. A fixed budget
below the implemented minimum or above J0 is rejected. Coverage diagnostics
distinguish full-pool mass coverage and feasible-active mass coverage; dynamic
thresholds use the latter, as in the report.

Optional quadrature uses `M_j=sum_i w_ji*Wpool_i/Wselected_i`. Summing over
selected j gives exactly `sum_{covered i} Wpool_i`. No normalization/clipping
of gamma is applied. Alpha always uses the full pool, so no quadrature mass
approximation enters bisection. The scalar identity does not preserve each
local normal operator or guarantee recovery. Canonicalization/solver consume
the explicitly modified masses only in this quadrature variant.

## Integration and verification

All variants use the same public full-fit path, initialization, candidate pool,
training split, and per-center sketch stream. A separate selection RNG cannot
consume initialization/direction randomness. Step error is computed on the
complete candidate pool for every arm, so shrinking J cannot mechanically win
best-step selection by scoring fewer centers. Selected coefficients map back
through `effective_parameters_["coefficient_center_indices"]` and per-step
`selected_pool_indices`. CPU/new-estimator/built-in Epanechnikov are checked
before initialization; unsupported GPU/kernel/single/legacy use is rejected.

Tests retain the original dense weights/alpha/statistics as an independent
implementation path; additionally direct weighted covariance, direct kernel
boundary values, exhaustive greedy marginal choices, and PoU mass sums are
checked. No default or solver change is supported by this derivation alone.
