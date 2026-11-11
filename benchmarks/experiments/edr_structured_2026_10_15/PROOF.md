# Mathematical contract, 2026-10-15

Source: `SVD/edr_research_note.tex`, equations (Schur, dg, HVP, CX,
metricstep). These are algebraic consequences to check, not speed guarantees.
The implemented objective is the existing frozen finite sketch:

    Phi(Y) = sum_j mass_j min_g (||I_j-U_j Y g||² + tau ||g||²), Y.T Y=I.

`lambda_prox` damps a numerical step; it never enters Phi. All pairwise
comparisons fix tau. Positive tau changes the estimator relative to tau=0.

## Schur, including rectangular cores

Let Y.T V=0, V.T V=I_q and A.T A=I_a; complete A by A_perp. Let
C=UY A_perp, D=UY A, W=UV and K have shape (q,a). Polar(Y+V K A.T)
is equivalent to an unnormalized design with coefficient penalty
tau*(I+K.T K) on the active a coordinates and tau*I on the others.
Eliminating inactive coefficients gives Pi=I-C(C.T C+tau I)^-1 C.T.
Do not form a p*p projector: project [D,W,I;0,0,0] orthogonally against
the augmented inactive design [C;sqrt(tau)I]. Its residual Gram gives:

    k=I.T Pi I, alpha=D.T Pi I, beta=W.T Pi I,
    E=D.T Pi D, F=D.T Pi W, H=W.T Pi W, T=H+tau I_q,
    S=E+F K+K.T F.T+K.T T K+tau I_a, h=alpha+K.T beta,
    g=solve(S,h), rho=beta-(F.T+T K)g,
    phi=sum mass*(k-h.T g), grad=-2 sum mass*rho g.T.

For perturbation L, B=F.T+T K:

    dg=solve(S,L.T rho-B.T L g),
    drho=-T L g-B dg,
    HVP=-2 sum mass*(drho g.T+rho dg.T).

The quadratic form is 2 sum mass*((L g).T T (L g)-dg.T S dg).
It can be negative. Steihaug truncated CG must terminate on negative curvature
at the trust boundary; positive block preconditioning never changes HVP.
The block preconditioner retains the SPD part 2 sum mass*T*g_l² separately
for each active coefficient l and adds positive damping. It approximates
curvature only for CG, not the objective. No q*a square Hessian is stored.

At tau=0 derivatives require well-conditioned full-rank local designs;
otherwise the solver uses value-checked metric/spectral descent and reports
rank fallback, never adds an unrequested ridge. Final acceptance uses the
original QR/SVD local profile to protect against Schur subtractive cancellation.

## CX

For normalized row weights a, the existing statistics compute centered Q,
H=(Q-Q a)*a, s=sum H. C=(H-s*a)*gamma, gamma=1 or local mass.
Therefore U=C X and I=C y. The builder retains CSR C and globally centered X
and y; this changes reduction order in float64, not the exact estimator.
Rows have zero sum algebraically. Centering avoids amplifying their roundoff
sum by a large common offset. Operator actions use C(XV) and X.T(C.T P).
CSR storage is O(p*nnz(weights)+n*d), not necessarily less than dense U.
Construction cost and both representations' storage are measured explicitly.

## Metric and recycling

R=sum mass*g*g.T with a positive spectral floor is a search metric only.
For G horizontal, the rank-q metric steepest direction is
Z=-[G R^-1/2]_q R^-1/2 / ||[G R^-1/2]_q||_F.
Both right-whitening factors are retained; refactor the entire Z before
constructing orthonormal V. This is not an exact SVD solution for general U.

A cached shell R_shell spans current Y and the newly proposed V. Subsequent
bounded steps compute the exact restricted gradient from U R_shell and use
that same cached operator for trials. A small restricted gradient forces a
full refresh; it cannot certify full stationarity. Every return recomputes
UY and the full ambient gradient and records action counts. Shells are local
to one frozen solver call; no cache crosses an outer weights/sketch update.

Newton here is exact only inside the chosen q*a chart, not a full Grassmann
Newton method and not a quadratic-convergence guarantee. All candidates stay
opt-in regardless of exploratory benchmark results.
