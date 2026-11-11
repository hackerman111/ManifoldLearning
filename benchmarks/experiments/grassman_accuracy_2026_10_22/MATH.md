# Isolated accuracy candidates

All candidates preserve F(Y)=sum_j mass_j min_g (||I_j-U_jYg||²+tau||g||²)
and row-basis output Y.T. Reference is the exact live HEAD4b06b98 source.
Tests: tests/test_grassman_accuracy22.py. No new statistical assumptions.

## N: orthogonal-form profile derivative (NUMERICAL)

For a smooth full-column-rank augmented design A=[M;sqrt(tau) Id]=QR,
target b=[I;0], g=A^+b, residual r=b-Ag, and perturbation E=[WK;0],
differentiating A.T r=tau-adjusted zero and r=b-Ag gives

    dr = -(Id-Q Q.T) E g - Q solve(R.T,E.T r).

This is the same full residual Jacobian as the reference, including both
variable-projection terms. Neither squared singular inverses nor a normal
matrix are formed. Q is obtained from an extra thin QR; no d*d matrix.
J,p,m,q shapes give O(J*(p+m)*q*m) scratch. Candidate multiplies the
existing rows*q*m estimate by4 (4->16) before any Jacobian allocation to
cover QR/design/direct/cross/solve/product temporaries. This conservative
guard may trigger the existing spectral fallback earlier than baseline.
The production guard is unchanged; no derivative at a rank boundary claimed.

Important negative result: a six-row nearly collinear design cond~2e7,
80-digit central-difference reference (h1e-27) yields relative Jacobian error
3.8081e-9 for original versus3.8360e-9 for this formula. One mixed-precision
profile refinement gave3.4942e-9 and was discarded. The expected QR/SVD
factor/coefficients error O(eps*cond) remains. Thus removing inverse-squared
notation does NOT itself establish accuracy improvement. Test records that
conditioning bound and FD agreement, not a fabricated precision win.

## S: relative damping for every GN proposal (APPROXIMATE)

Use mu=lambda_prox*||J||F²/(q*m) in both smooth and intrinsic models.
The existing variant normalizes only intrinsic GN. For positive response
scaling I->cI and weights mass->a mass, J->c sqrt(a) J and r->c sqrt(a)r.
The augmented LS quadratic scales by a*c², so K is unchanged in exact
arithmetic. This statement is conditional on identical rank/stopping branches
and finite positive mu; the old scale floor and all fallbacks remain.
Smooth absolute damping has no such invariance. Work added: one scalar
Jacobian reduction. Estimator, cap, full-profile Armijo and certificate fixed.

## P: SPD diagonal plane (APPROXIMATE)

Let G=(Id-YY.T) grad F and

    D_ii = sum_j mass_j ||g_j||² sum_p U_jpi²,
    delta = .05 mean(D), Z=(Id-YY.T) (D+delta Id)^-1 G.

This is a diagonal surrogate to frozen-g curvature, not exact profiled
Hessian or Newton. If delta>0 and G!=0, <G,Z>=<G,(D+delta Id)^-1 G>>0.
Choose the core ambient plane by thin SVD(Z) instead of SVD(G), then reuse
the original full Jacobian, true reduced gradient V.T G and acceptance.
After rank truncation the actual slope is checked as before; the positivity
above is not a convergence proof or an unconditional guarantee for truncation.
No center is removed from gradient/certificate. A nonfinite/nonpositive
scale falls back to original G. Coordinate diagonal is not rotation invariant
under a general ambient change of axes; compare correlated cases explicitly.
It is invariant to orthogonal basis changes inside Y, up to SVD ties.
The .05 floor is fixed before exploratory runs, not fitted to validation.

Compute D in center chunks using sum_p U², no copy of entire U and no d*d.
Extra work O(J*p*d) per inner step, scratch <=4MiB plus one center; base
cached actions and solver workspace unchanged. Test tiny chunk against an
independent dense contraction; actual full-fit time/RSS decide utility.

The existing SVD drafts justify profile/core/angle algebra, not an EDR gain.
None of N/S/P guarantees statistical improvement; paired full ADP and actual
R MEANMAVE results are required. No production modification at this stage.

## Q: one extra length proposal for an accepted polar GN step (APPROXIMATE)

The first24/24 exploration fits gave no winner for N/S/P. Q is a revised,
separate hypothesis, not an attempt to combine those rejected variants.
It uses the baseline damping and plane. For fixed capped K, define
Y(t)=polar(Y+t VK). Its derivative at0 is VK, so f'(0)=<G,VK>=s<0.
An already accepted baseline trial at t0=2^-backtrack gives a quadratic
interpolation with stationary length

    t_star = -s*t0² / (2*(f(t0)-f(0)-s*t0)).

Only positive finite denominator is used. Clamp t_star into
[t0/8,min(2*t0,tan(max_angle)/||K||2)], skip relative changes<=1e-3,
evaluate one additional full profile, retain it only if its value is less
than the accepted baseline and satisfies its own Armijo bound.
Otherwise retain the original accepted pair unchanged. No recursive search,
new damping choice or interpolation-only acceptance. The norm cap still
holds. More than one GN step can change future trajectories, so the one-step
objective dominance is not a guarantee of full-run objective/EDR improvement.

Extra work is at most one local profile/refit per accepted GN step; no new U
application or d-dependent search. One extra profile state is transient.
Tests include baseline one-step dominance over5 deterministic problems with
strict gain in at least one, ridge/nonridge true-profile and final-gradient
agreement, and the earlier independently checked profile kernels.

## R: positive local ridge (ESTIMATOR)

Existing tau>0 profile and exact derivative are already tested independently.
Explore only fixed tau1e-6,1e-4,1e-2 with20steps; no new math implementation.
All20/20 pilot fits succeeded but every ridge worsened d50 EDR8..16%; reject.
The regularizer cannot be described as a numerical cancellation fix.

## K: bounded full-tangent Krylov plane (APPROXIMATE)

Past full-GN30 failed its inner certificates and adaptive models were never
used because of rank guards. Instead reuse ProfileJacobian with exactly the
eligible regular-center model already used by intrinsic core GN. Two LSMR
iterations approximate min_Z ||J_full Z-r||²+mu||Z||² solely to choose the
ambient core plane. Positive sign is intentional: its plane approximates
the preconditioned gradient; ordinary core GN still chooses the signed step.

Set mu=lambda_prox*||J_full G||²/||G||² (positive finite only), a
response/mass-scale-equivariant directional curvature. Record LSMR stop,
iterations, residual/normal-residual estimates; capped iteration does not
certify a full Newton solve. Reject nonfinite or nonascent Z back to G.
Horizontalize Z, thin SVD gives a plane of at most m columns. Continue with
the original q*m core GN, original all-center G and true-profile Armijo.
Final stationarity uses the original all-center live operator.

No new statistical criterion or rank cutoff. For intrinsic proposals we pad
the regular profile with zero coefficients/residuals and zero mass, safe
identity factors in omitted rows, instead of copying U[mask] (potentially
gigabytes). The resulting full-size operator has exactly zero omitted rows;
its nonzero rows and adjoint are those of the regular-center Jacobian. Test
compares both actions with a separately constructed subset operator and
asserts sharing the original U buffer. No full J or dense d*d matrix.
Extra cost: one Rayleigh J action and2LSMR iterations; extra storage
O(J*(p+m)*m+d*m), plus existing4MiB adjoint chunks.10/20budgets are separate
frozen arms. This model is a proposal and does not change rank applicability.
