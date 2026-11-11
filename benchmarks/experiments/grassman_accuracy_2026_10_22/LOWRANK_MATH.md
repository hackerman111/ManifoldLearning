# Joint correction proposals (2026-10-22)

These are APPROXIMATE optimizer variants for the same finite-sketch profile,
not estimator changes. Shapes: Y=(d,m), U=(J,p,d), g=(J,m), r=(J,p),
mass=(J); Y has orthonormal columns. All calculations use float64.
Production grassman.py is unchanged. Positive local ridge is guarded out.

For one iteration freeze the current minimum-norm local coefficients g and
write A(Delta)_j=U_j Delta g_j and r_j=I_j-U_j Y g_j. The proposal objective is

    q(Delta)=sum_j mass_j ||r_j-A(Delta)_j||² + mu ||Delta||_F²,
    Y.T Delta=0, rank(Delta)<=k.

This is the additive fixed-g correction problem from SVD_solver.tex680-774,
with a horizontal constraint and subsequent reprofiling. It differs from
grassman.block_rank_k (joint angles) and from SVD.py's matrix-rank target.
The correction rank is the rank BEFORE retraction: rank(Q-Y) after polar
normalization need not be k. k is at most min(m,d-m) for the initial plane;
an enriched plane can have up to 2k vectors, but Delta still has m columns.

For horizontal Delta, C=I+Delta.T Delta is positive definite and
Q=(Y+Delta) C^(-1/2) is orthonormal. Choosing local coefficients
g'_j=C^(1/2)g_j gives Q g'_j=(Y+Delta)g_j. Consequently, in exact arithmetic
with no local ridge,

    F(Q) <= sum mass ||r-A(Delta)||² <= q(Delta).

Thus q(Delta)<=q(0) implies F(Q)<=F(Y). This is a majorization consequence,
not a recovery guarantee. The code additionally requires actual all-center
profile Armijo decrease, caps tangent spectral norm at tan(max_angle), and
recomputes the final profile/gradient through live U. Floating-point rank
cutoffs can invalidate exact identities, which is why this extra test matters.

SVD block: obtain k horizontal left singular vectors V of the full gradient,
then solve for ALL k*m entries of C in Delta=V C at once. The weighted design
is D[j,p,(a,b)]=sqrt(mass_j)*(U_j V)[p,a]*g_j[b]. An augmented least-squares
solve [D;sqrt(mu)I] avoids squaring its condition number. This is a conditional
optimum in a chosen plane, not the global optimal rank-k correction. Joint
coefficient refit is broader than refitting only k singular scales.

LoRA-like: initialize with that block and alternate its two factors. SVD of
Delta gives an orthonormal-row right factor R=(k,m). For fixed R, solve
for L=(d,k) using the matrix-free operator sqrt(mass) U horizontal(L) R g.
Because R R.T=I, ||L R||_F²=||L||_F². The identity ridge belongs to this
product gauge; separate penalties on arbitrary factors would change q.
LSMR uses an explicit augmented ridge block centered at ZERO Delta; its x0
is only an initial iterate. Limit: two factor rounds, three LSMR iterations
each. Stop code/iteration/normar are retained; cap exhaustion is not convergence.
Orthonormalize the new left space and jointly refit C again. Accept a factor
round only if its original q does not increase; retain the prior block otherwise.
This mitigates greedy path dependence but does not establish global optimality.

Independent idea: after a coarse block compute the full frozen-g normal
residual R=A*(sqrt(mass)r-A Delta)-mu Delta in weighted operator notation,
remove its old-plane component, and augment V with up to k new vectors.
Resolve the same q on that enlarged space with the SAME mu. Since the old
space remains feasible, the exact coarse optimum cannot worsen q. The code
still checks it. This differs from the previously rejected profiled-Jacobian
two-iteration Krylov direction: no derivative of local coefficients is needed,
so rank-deficient local profiles are allowed for proposals.

mu=.05*sum_j mass_j ||g_j||² ||U_j V||_F²/(k*m), fixed within the iteration.
Response and mass unit changes scale design/rhs/ridge together, leaving Delta
unchanged. This is optimizer damping, not local coefficient regularization.

The literal old SVD greedy control calls correction mode with gradient pair
selection and no dense direct normal solve. Its public rank constraint is
rank<m, so its maximum rank is m-1 (unlike the new k=m arm); this control is
not an equal-rank full-k ablation. A single rank-one pair is still a true
rank-one additive correction after horizontal projection.

Storage: matrix-free actions use O(J*p*k+d*k) working arrays, no ambient
Jacobian/Hessian or d*d matrix; adjoint data blocks are about 4MiB. Coarse LS
estimates four design-sized work arrays and rejects estimates above16MiB.
Original U storage is shared. The new prototypes are CPU-only; production
NumPy/CuPy compatibility is preserved by leaving production untouched.
Standalone low-rank helpers require C-contiguous U and reject strided input
instead of silently copying the entire statistics tensor. The routed solver
already provides contiguous U before calling them.

Additional own proposal P: let D_j=[M_j;sqrt(tau)I] and let E_j be the
perturbation [U_j V K;0] of that local design. With profiled residual z_j,

    J_full K = -(I-D D^+) E g - D (D.T D)^(-1) E.T z
             = J_perp K + J_parallel K.

The two terms are orthogonal in residual space. At local LS normality,
D.T z=0, so J_perp.T z=J_full.T z and
J_full.T J_full-J_perp.T J_perp=J_parallel.T J_parallel is positive
semidefinite. Keeping only J_perp in the small GN core retains its gradient
while discarding a potentially conservative curvature contribution. This is
an APPROXIMATE Jacobian model, not a numerically equivalent derivative.
It adds no full-U actions and retains original damping, rank/intrinsic guards,
true-profile acceptance, step caps, and final certificate. Independent augmented
QR, gradient, PSD, unsafe-rank and final-certificate checks are in
tests/test_grassman_projected22.py. It need not improve recovery or convergence.

Independent checks: tests/test_grassman_lowrank22.py, dense physical forward
and adjoint, direct coarse ridge optimum, full-horizontal reference recovery,
rank/horizontal constraints, unit invariance, LoRA cap diagnostics, enrichment
objective, workspace guard and all-center final certificate. Full-fit EDR/time
and R evidence belong in LOWRANK_REPORT.md, separately from these identities.
