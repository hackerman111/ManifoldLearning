# Metric descent: what this experiment can distinguish

## Existing optimizer

For orthonormal Y (d,m), frozen U (J,P,d), I (J,P), mass (J,), the objective
is Phi(Y)=sum_j mass_j min_g ||I_j-U_j Y g||^2, with solver tau=0. Let G be
its full horizontal gradient, g_j the current minimum-norm local coefficients,
R=sum_j mass_j g_j g_j.T, and R_f its positive spectral flooring at
epsilon*lambda_max(R). The implemented proposal is

    Z = -[G R_f^(-1/2)]_q R_f^(-1/2)
        / ||[G R_f^(-1/2)]_q||_F.

For nonzero G this minimizes <G,Z> among rank<=q matrices with
||Z R_f^(1/2)||_F<=1. Substitute W=Z R_f^(1/2) and use the truncated SVD
linear-functional optimum. R acts on coefficient coordinates (m,m); it is
not an ambient feature covariance. Orthogonal coefficient changes Q give
G'=GQ, R'=Q.T R Q, Z'=ZQ when the truncation has a nondegenerate cutoff.

One heuristic curvature motivation is the fixed-coefficient linearization:
its quadratic term is sum mass_j*||U_j E g_j||^2. Only under an additional
approximation U_j.T U_j ~= c*I on the relevant tangent directions does this
become c*trace(E R E.T). The actual local operators differ and the profiled
coefficients also respond to E. R therefore omits local feature geometry and
VARPRO curvature; it is a cheap search metric, not the exact profiled Hessian.

The solver selects V from the full Z, orthogonalizes V against Y, recomputes
the metric proposal for V.T G, caps its spectral norm at tan(.5), and uses
polar retraction and true-profile Armijo acceptance. Both plane selection
and step length matter. q is an upper bound and can drop with numerical rank.
Matrix work remains thin d*m and local J*P*m plus small m*m factors.

For m=q=2 and full-column-rank G, multiplication by the invertible right
metric does not change col(G). Thus metric2 and isotropic2 choose the same
two-dimensional left search span in exact arithmetic; their difference is
the direction inside that span and its scale. Rank1 is the actual additional
truncation, where weighting can also change the selected left plane. Calling
rank2 a stronger low-rank restriction than the original m=2 gradient would
be misleading. Numerical rank guards can still reduce either requested rank.

When metric_floor=1, R_f=lambda_max(R)*I and

    Z_iso = -[G]_q / (sqrt(lambda_max(R))*||[G]_q||_F).

The bound ||Z_iso||2 >= 1/sqrt(q*lambda_max(R)) does not imply that the cap
activates for q=2. Coefficient scale is a confound unless clipping cancels
it. The separate matched control preserves default proposal spectral length
on each metric call while replacing its orientation by truncated -G; this
does not enforce identical trajectories or accepted Armijo lengths.

The normalization also makes the proposal depend on a common weight scale.
At tau=0, replacing all masses by c*mass preserves local coefficients and
the minimizer of Phi, while G'=c*G and R_f'=c*R_f. Substitution gives

    Z' = Z/sqrt(c).

The certificate scale is unchanged relative to G when sum mass*I^2>=1, but
the uncapped optimizer trajectory need not be unchanged. The present line
search starts at this proposal and only halves it; it does not expand it.
Thus common-weight scaling is a step-calibration issue even without a metric
floor change. This identity is independently checked with c=100 on all nine
saved inputs in audit_proposals.py and results_repaired/proposal_audit.json.
All nine have actual q=2, an inactive default floor, and inactive metric and
isotropic angle caps. Their raw proposal spectral norms are about .0053-.0148,
well below tan(.5)~.5463. These are sampled initial proposals, not telemetry
for all later iterations or accepted steps.

Normalizing R by a scalar such as total mass would preserve its anisotropy
while changing this proposal scale. It is an untested optimizer variant;
these results do not identify a good scalar or establish that normalization
would improve recovery. The matched control removes one per-call magnitude
confound, but differs in norm constraint and subsequent trajectory from the
original metric method.

## Hard Spokoiny has no intrinsically weak population index here

This is a mathematical property of the exact existing generator, not a
measurement of its fitted local coefficients. Let x,y be independent
Uniform[-1,1], u=(x+y)/sqrt(2), v=(x-y)/sqrt(2). Then

    f=(u^3+v)(u-v^3)=u^4-u^3*v^3+u*v-v^4,
    f_u=4*u^3-3*u^2*v^3+v,
    f_v=u-3*u^3*v^2-4*v^3.

The rotated-square distribution is invariant under independent sign flips
of u,v and under their exchange. Odd moments in either coordinate vanish.
Writing M_rs=E[u^r v^s], direct multiplication gives

    E[f_u^2]=16*M60+9*M46+M02-6*M24,
    E[f_v^2]=M20+9*M64+16*M06-6*M42,
    E[f_u*f_v]=4*M40-12*M62+12*M26-4*M04.

Symmetry makes the two diagonal expressions equal and the cross term zero.
Exact rational integration gives c=5557/1155, so

    E[grad_(u,v) f * grad_(u,v) f.T] = (5557/1155)*I_2.

The rational calculation expands each monomial in x,y, uses E[x^k]=0 for
odd k and 1/(k+1) for even k, and includes 2^(-(r+s)/2). A separate Luna
audit checks this result independently. Thus the previous artificial weak
coefficient pilot is not this benchmark's population geometry. Empirical
profiled R can still be anisotropic through finite samples, noise, current
subspace, localization and rank loss. Measure those effects instead of
assuming that a second true population index is weak.

## Interpretation gates

The certificate always uses the full ambient frozen gradient and the shared
scale max(1,sum mass*I^2). Local rank loss at tau=0 restricts differentiability;
small gradients there are not smooth stationarity certificates. A lower
frozen Phi need not give a closer truth subspace or improve adaptive outer
weights. A full fit stopping at h_min has completed its schedule, without
establishing optimizer or statistical convergence. All arms fix the estimator;
this study does not compare against the older fixed-g SVD estimator.
