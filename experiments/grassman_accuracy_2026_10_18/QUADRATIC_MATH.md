# Quadratic angle proposals with true-profile acceptance

For the same rank-one geodesic f(t)=F(Y(t)), use f(0), slope s=v^T G a
and one true profiled value f(t). The interpolating quadratic is
q(x)=f(0)+s*x+(h/2)*x², h=2*(f(t)-f(0)-s*t)/t². If h is positive and
finite its minimum is x=-s/h; clamp it to [maxangle*2^-19,maxangle].
Evaluate up to three such proposals, beginning with maxangle=.5, and choose
the lowest ACTUAL value satisfying f(t)<=f(0)+1e-4*s*t. Stop proposing on
nonpositive/nonfinite curvature or a repeated angle. If no proposal passes,
use the original20-halving true-profile Armijo search. The model is solely
a proposal mechanism; neither q nor its curvature replaces the objective,
the GN model or a certificate. Every accepted value is a full QR/SVD refit.

At smooth fixed-rank points s is the directional derivative. At rank
boundaries s is a heuristic model slope and no differentiability, Hessian,
or global convergence theorem is asserted; actual acceptance decreases
the value since the signed gradient plane gives s<=0. Original local rank,
GN/workspace, polar/spectral and final live certificate guards are unchanged.
The rotation preserves orthogonality in exact arithmetic as in ARMIJO_PROTOCOL.

Budget is fixed before selection: ten steps for initial nonsmooth profiles,
five for smooth, three quadratic proposals per rank-one line search. Same
tau0 estimator and statistics. This is APPROXIMATE. Costs are up to three
initial true refits plus fallback if all fail; no scalar Schur subtraction.
Only current/best trial states are retained (O(J*p*m)); no d*d allocation.
