# Algebra, overlap, and limits

The target is exactly F(Y)=sum_j N_j min_g(||I_j-U_j Y g||^2+tau||g||^2),
Y in R^(d x m), Y.T Y=I. Local coefficients are refitted on every trial and
on the final live U action. Its smooth-region envelope gradient is
G=-2 horizontal_Y(sum_j N_j U_j.T residual_j g_j.T).

Let G=V diag(sigma) A.T be its thin horizontal SVD, q<=min(m,d-m).
For angles theta the exact Grassmann exponential in these principal planes is

    Y(theta)=Y + ((Y A_q)(cos(theta)-1)+V_q sin(theta)) A_q.T.
    UY(theta)=M + ((M A_q)(cos(theta)-1)+W sin(theta)) A_q.T,
    M=UY, W=UV_q.

Columns of V_q are horizontal and orthonormal; columns of A_q are orthonormal.
Thus Y(theta).T Y(theta)=I, and maximal principal rotation is bounded by
max|theta_i|<pi/2. No ambient d*d projector or rotation is needed. QR cleanup
can change V's gauge: take a small SVD of V.T G before using principal planes.
Shapes: M(J,P,m), W(J,P,q), A.T(q,m), G(d,m). Projection costs O(JPd(m+q))
per fresh gradient frame; each angle trial uses only O(JPm q) cached actions
plus local refit. Augmented joint angle Jacobian has O(J(P+m)q) storage;
augmented least squares uses QR/SVD, no explicit normal inverse.

Existing spectral implements theta_i=-t sigma_i/sigma_1 and Armijo; existing
flat_gn implements interacting profiled q-angle GN with exact exponential.
Existing core_gn optimizes a larger q*m rectangular core with polar retraction.
Neither block formula nor angle Jacobian is new. The experiment adds explicit
rank policy, short joint budget, bounded cost pilot, and tests the rank law.

## Conditional exponential bound (not an unconditional theorem)

In a smooth constant-rank region suppose the pullback is L-smooth along the
tested geodesic, step eta=1/L respects the cap, and the profile satisfies a
local PL inequality ||G||^2 >= 2 mu (F-F*) for the same minimum F*. Then

    F(Exp_Y(-eta G_k)) <= F(Y)-||G_k||^2/(2L),
    ||G_k||^2=sum_(i<=k) sigma_i^2 >= (k/r)||G||^2,
    (F_new-F*)/(F_old-F*) <= 1-(mu/L)k/r <= exp(-(mu/L)k/r).

These are additional hypotheses: no global PL or uniform curvature bound is
established for the arbitrary finite-sketch profiled objective. Near local
rank changes even smoothness fails; near nonoptimal stationary points the
gradient can vanish with a positive gap to a better solution. Armijo can take
a much smaller step than1/L. Optimized GN is tested empirically; its accepted
step is not guaranteed to beat a separate spectral trial. This conditional
derivation therefore cannot prove the user's boxed law for this solver.

An explicit smooth counterexample rules out an unconditional c>0 claim.
Take J=1,d=4,m=2,U=I4,I=(1,0,0,0),Y=(e3,e4),N=1,tau=0. The local matrix
UY has full column rank, g=0, F=1 and G=0. At Y*=(e1,e2), F*=0.
Any gradient-selected block remains stationary: the ratio is1, whereas
exp(-ck/r)<1 for every c>0,k>0. The code's stationary flag is only a
first-order certificate, not global optimality. This counterexample is
covered in tests/test_grassman_rank_reference.py; it does not preclude a
useful empirical law on the planted benchmark subset.

Replacing F* by a numerical F_ref changes the ratios and may invalidate a
bound; retain negative/near-zero gaps separately and publish reference full
gradients/start spread. Fit c on training only and test on untouched tasks.
EDR truth is a separate target: lower finite-sketch F does not logically imply
smaller projector error under noise or adaptive outer statistics.
