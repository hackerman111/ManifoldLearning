# e-means ideas inside the original anisotropic ADP

Research spike, 2026-11-13. This document states scoped results, not a proof
of better EDR or faster convergence of adaptive ADP. Reference:
[Bessa et al., arXiv:2611.09501v1](https://arxiv.org/html/2611.09501v1),
sections 3.1-3.3 and appendix H. Their target is clustering MSE. We borrow
mini-batch proposals, a cosine EMA and full finalization; we do not reproduce
e-means or transfer its empirical scaling laws to ADP.

## Fixed target and dimensions

At one original ADP scale the public code supplies fixed
`U in R^(J,p,d)`, `I in R^(J,p)`, positive `w in R^J` and orthonormal
row basis `B in R^(m,d)`, with `1 <= m < d` and `p > m` in the full fit.
The unchanged Grassmann target is

    F(B) = sum_j w_j min_g (||I_j - U_j B^T g||^2 + tau ||g||^2).

The study pins tau=0, original combined20 settings and float64. Initializer
ridge is a different parameter. Kernel, local moments, sketch law, center pool,
bandwidth/alpha search and best-step selection remain in the public ADP loop.
Changing B changes later adaptive weights through the original formulas.
Thus identical formulas do not imply identical realized outer trajectories.

The implementation shares the ORIGINAL numerical rank cutoff/evaluator.
An exact minimum over all g agrees with the ideal SVD calculation only when
every nonzero singular direction is retained. Truncating a small but nonzero
direction can differ from that ideal minimum; section4 explicitly limits the
real-arithmetic descent claim. The unconditional implementation safeguard is
about its recomputed numerical value, not an interval-certified ideal F.

## Four genuinely different families

### A. EMA of observation-level sufficient statistics (ESTIMATOR)

For fixed local weights, a covariance requires weighted count N, sums Sx,Sy,
and second sums Sxx,Sxy. An EMA of mini-batch sums equals a weighted sum with
coefficient `a_t = alpha_t product_(s>t)(1-alpha_s)` on batch t. Centering by
the EMA count is therefore exactly the covariance under reweighted weights
`w_i a_batch(i)`, not generally the original covariance. This follows by
induction on the affine recurrence, followed by the weighted-mean identity.
Only equal batch coefficients, or an independent final accumulation of ALL
original weights, gives the original moments. If Phi or localization changes,
old projected statistics cannot be used as statistics of the new scale.

Costs: storing Sxx for every center is O(J d^2), versus original U O(J p d).
For d=1000,J=300,float64 this alone is 2.4 GB. Computing sketched statistics
instead can retain O(J p d) storage only with fixed projections. A full exact
final pass still costs the original moment construction. **Result:** exact
reweighting identity. **Gap:** no reason its bias improves EDR, and no saved
exact final pass. Reject for this preservation-focused prototype.

### B. EMA of outer anisotropy / coefficient scatter (ESTIMATOR)

A convex average of PSD structural matrices is PSD and invariant under basis
changes if implemented in ambient coordinates. It need not represent the
current bandwidth's gradient distribution. Even in d=2,m=1, averaging a past
projector e2e2^T with the current true e1e1^T returns e2 when the past weight
exceeds one half, and is nonunique at one half. Thus smoothing can worsen EDR
even when the current proposal is perfect.

Under the ADDITIONAL assumptions of independent, mean-zero matrix errors with
finite second moments about one fixed target and deterministic normalized
weights a_t, these weights satisfy
`E||sum a_t E_t||_F^2 = sum a_t^2 E||E_t||_F^2`: expand the square and use
zero cross expectations. Actual ADP uses the same data, Y-dependent geometry,
different bandwidths and possibly rank boundaries, so those assumptions are
not established. A spectral gap is additionally needed to control a subspace.
Even under those assumptions, improvement over the latest estimate requires
variance control: weights .5/.5 and independent scalar variances100 and1
give variance25.25, larger than1. The identity is not a reduction theorem.
Exact untruncated low-rank factors cost O(d m T) storage, and their thin SVD
costs O(d r min(d,r)), r <= m(T+1). Truncation at each scale changes the EMA.
**Result:** PSD/gauge identity and the explicit conditional variance identity.
**Gap:** adaptive bias/dependence and a population EDR gap. Reject as primary.

### C. Mini-batch optimization of the fixed profile (APPROXIMATE, selected)

Use the ORIGINAL Grassmann solve for each proposal, with one step on each of
four random center batches, for two fresh-shuffle epochs. Rescale batch mass
by J/b for a batch of size b. This is an explicit optimizer choice: smooth
GN uses absolute lambda_prox damping, so this rescales its damping relative
to the data by b/J; stopping also uses max(1,sum w I^2). It is not a claim
that scaled and unscaled bounded solvers follow the same path. Maintain an
optional EMA of the projectors of
the eight proposals. EMA is used only as an extra final starting point.
Evaluate the input, last proposal and (if eligible) EMA on ALL original
centers. Pick the smallest full-profile value; run the original solver for
four full steps; independently reevaluate the full operator and roll back if
the computed full objective increased. All final coefficients/diagnostics
are recomputed on ALL centers. No minibatch certificate is returned as a
certificate of the full problem.

Controls: original20; original6 (same nominal six full-equivalent step budget);
mini without EMA; mini with EMA. Four calls per epoch are capped by J on tiny
inputs. Batch schedules/EMA state reset at every outer solve.

**Result:** unchanged full target, gauge-correct EMA and full-objective
nonincrease relative to the INPUT in exact arithmetic, proved below.
**Gap:** no improvement theorem relative to original20's output, no EDR
guarantee, no global convergence result, and no guaranteed wall-time saving.
The expensive original statistics are still constructed in full.

### D. Revive/drop uninformative centers (EXACT only for exact constants)

If U_j=0 then its profiled contribution is w_j||I_j||^2, independent of B;
g=0 is the minimum-norm/ridge solution. Removing it and restoring the constant
preserves the problem. Current grassman.solve(constant_centers=True) already
does this. Small norm alone is insufficient: for tau=0, U_j=epsilon A_j with
epsilon>0 has exactly the same range/projection loss as A_j, using g/epsilon.
Dropping it can remove an O(1) contribution for arbitrarily small epsilon.
Reviving/moving it changes kernel neighborhoods and is ESTIMATOR.
Costs: screening is O(J p d); removing D exact-zero centers saves repeated
actions O(D p d m) per full projection, but not upstream moment construction.
**Result:** exact constant elimination. **Gap:** already implemented; no new
asymptotic or unconditional EDR benefit. Do not add a near-zero heuristic.

## Exact claims for C and their proofs

### 1. Target and batch identity

For orthogonal R in R^(m,m), replace B by RB and g by Rg. Predictions and
||g|| are unchanged, proving F(RB)=F(B), including rank-deficient systems.
For a uniformly random subset S of cardinality b and a FIXED B independent
of S, linearity gives `E[(J/b) sum_(j in S) w_j f_j(B)] = F(B)` because
`Pr(j in S)=b/J`. This does not imply unbiased adaptive gradients: later
batches of a random permutation and later B values are dependent. We use
this identity only to describe the scaling, not as an SGD convergence proof.

### 2. Low-rank EMA and well-definedness

Let B_t have orthonormal rows, N_0=1e-10, L_0=sqrt(N_0) B_0^T. For
`alpha_t = .5[.1 + .9(1+cos(pi t/(M-1)))/2]`, t=0,...,M-1 (M=1 uses .5),
set gamma_t=1-alpha_t. This endpoint-adjusted schedule deliberately differs
from the paper's t/M schedule and fixes the uncalibrated peak at .5.
For updates k=1,...,M define

    L_k = [sqrt(gamma_(k-1)) L_(k-1), sqrt(alpha_(k-1)) B_k^T],
    N_k = gamma_(k-1) N_(k-1) + alpha_(k-1).

Block multiplication proves
`L_k L_k^T = gamma_(k-1) L_(k-1)L_(k-1)^T + alpha_(k-1) B_k^T B_k`.
Induction proves PSD, trace=m N_t, N_t>0 and the exact untruncated EMA
identity. No ambient d*d projector is materialized. Orthogonal changes of
any B_t preserve L_t L_t^T. A top-m left singular subspace minimizes
`||Q - L_t L_t^T/N_t||_F` over rank-m orthogonal projectors Q: expand the
square into constants minus `2 tr(Q C)`; the spectral theorem bounds this
trace by the sum of the top m eigenvalues, attained by their eigenspace.
This PROJECTOR approximation is not an EDR statement.

Uniqueness requires lambda_m>lambda_(m+1). The code rejects the EMA proposal
when this gap is <=64 eps max(1,lambda_1), rather than choosing an arbitrary
basis at a tie. SVD is on a d-by-r factor with r<=m(M+1), without truncating
its history. Degenerate/empty/invalid inputs are rejected or use the defined
non-EMA proposals; no zero vector is normalized.

### 3. Full finalization safeguard

The input B0 is among the candidates, so their exact full-profile minimizer
Bc obeys F(Bc)<=F(B0). For ANY returned polished proposal Bp, the final guard
chooses Bp if F(Bp)<=F(Bc), and Bc otherwise. Thus the returned Br satisfies
F(Br)<=F(B0), irrespective of minibatch success or line search failure.
The final minimum-norm coefficients are those of Br at all J centers. This
proof uses actual full values, not a minibatch proxy or cached action.
It does NOT compare Br with the output of original20.

### 4. Finite precision: exact domain and remaining limitation

Input/output orthogonality and finite values are checked. SVD/lstsq use the
original numerical rank cutoff. Let A=[U_j B^T;sqrt(tau)Id], b=[I_j;0].
For full column rank sigma_min(A)>=s>0 and perturbations ||E||=a<s,
||e||=h, put r=||b||. Least-squares solutions obey
`||x||<=r/s`, `||x'||<=(r+h)/(s-a)`. Evaluating each problem at the other's
solution bounds the change of its optimal residual NORM by

    D = h + a(r+h)/(s-a).

Both residual norms are at most r and r+h, respectively; hence the change
of squared residual is at most `(2r+h)D`. Sum these bounds with positive w_j
and add the error of the final scalar reduction to bound F's perturbation.
This is an explicit conditioning bound, not a measured roundoff certificate.
It blows up as s approaches a. At unregularized rank boundaries no such
uniform bound exists, and the numerical cutoff itself can change the target.

The prototype uses float64 reevaluations and a zero-tolerance comparison of
COMPUTED values, not interval arithmetic/backward-error bounds per fit.
Accordingly claim 3 is an exact-arithmetic theorem; tests of its float64
realization are numerical evidence only. No rigorous real-arithmetic descent
or EDR certificate is claimed for rank-boundary adaptive fits. Stationarity
uses the full live operator and is labeled inapplicable at a rank boundary.

### 5. Objective descent cannot stand in for EDR

Let d=p=2,m=J=1,w=1,tau=0, U=diag(1,2), I=(1,1), truth=e1. B0=e1^T has
F(B0)=1 and EDR=0. B1=(2,1)/sqrt(5) has F(B1)=0 and EDR=1/sqrt(5).
Both local designs have full column rank. Thus even EXACT full-profile
improvement can strictly worsen EDR. Neither a fixed-scale proof, lower
training loss nor a stationarity certificate solves adaptive recovery.

## Explicit resource accounting (no unconditional speed claim)

For dense U, a full action U B^T costs about 2 J p d m scalar FLOPs.
Local thin SVD costs O(J p m^2) for p>=m. A core GN model of width q*m
requires O(J(p+m) q^2 m^2 + (q*m)^3) work for its dense least squares,
with q<=min(m,d-m); operator actions and line-search evaluations add work.
These expressions describe the original subsolve; batches replace J by b.
Eligible rank-deficient centers also require an intrinsic SVD of U_j, costing
O(J_bad p d min(p,d)) per fresh rank cache. Subcalls recreate those caches,
so this work can repeat even though the total minibatch center count is2J.
With two complete sweeps of one-step batches and four full final steps the
nominal STEP work is six full-center equivalents, versus twenty for the
control. This is not a 20/6 runtime theorem: each subsolve repeats initial/
final profiles, gradient evaluation, intrinsic SVD and validation; line
search counts and trajectories also differ. Full-value guards add at most
five full profile evaluations (input,last,EMA,polished,final refit).
Record actual calls/accepted steps/evaluations and measured full-fit time.

Persistent original input U remains 8 J p d bytes; no RAM-saving claim.
One batch copies O(b p d). The original16 MiB GN allocation-estimate threshold
triggers a fallback; it does not bound process RSS, caches or LAPACK memory.
EMA storage is O(d r), r<=9m; thin SVD costs O(d r min(d,r)). Its numerical
Gram matrix is never formed, avoiding squared conditioning of L. Economy
SVD vectors may be square on tiny d<=r, still bounded by O(d r).
At d1000,m3,J300,p12, U is28.8 MB; the EMA factor is at most216 KB. All
figures exclude the unchanged full-pool distances and public fit workspace.

## Adversarial acceptance

Tests must compare EMA with a dense projector recurrence; prove gauge/sign
invariance away from ties; exercise ties, zero/rank-deficient operators,
unequal masses, ridge and ill-conditioning; independently refit final
coefficients and check objective/gradient semantics. End-to-end identity of
the baseline with the public API must hold. Frozen same-data/seed experiments
must retain failures, audit saved bases and use quality AND timing gates.
Failure closes a candidate without touching validation or changing defaults.
