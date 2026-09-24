# H7: guarded reduced residual Gauss–Newton

Status: derivation and separate algebra audit completed before implementation.
Isolated APPROXIMATE optimizer; objective/mass/cutoff unchanged, CPU float64.
The domain, spectral guards, discarded-value/gradient-defect identities, and
stationary correspondence are those proved in
`../multi_solver_search_2026-09-24/proofs/reduced_lbfgs.md`. In particular a
rank-deficient local system is allowed; a full-column-rank-only proof is invalid.

## 1. Residual and its exact derivative on the fixed-rank spectral domain

For one center, `A=UBᵀ=Q diag(s) Vᵀ`, Q has m columns, `alpha=QᵀI`,
`q0=I-Q alpha`, retained set R={i:s_i>tau*s_0}, tau=eps max(P,m).
The reduced residual is `r=I-Q_R alpha_R`, weighted residual `r_w=sqrt(w)r`.
Its squared norm/2 equals the implemented local-refit objective. Right
orthogonal rotations of B preserve s, retained projector, r and f.

For a row-basis variation Z let `E=UZᵀ`, `T=EV`, and `c=V_R(alpha_R/s_R)`.
Differentiation of the retained spectral projector yields `dr=-D`, where

```
D = (I-QQᵀ) E c
    + sum(i in R) q_i (q0ᵀ T_i)/s_i
    + sum(i in R, k not in R, k<m)
      (s_i q_kᵀ T_i + s_k q_iᵀ T_k)/(s_i²-s_k²)
      * (alpha_i q_k + alpha_k q_i).
```

Derivation: for `M=AAᵀ`, cross-cluster projector differential is
`dP=sum(i in R,k outside R) (q_k q_iᵀ+q_i q_kᵀ)
* q_kᵀ(dA Aᵀ+A dAᵀ)q_i/(s_i²-s_k²)`.
Multiplying by I gives the finite k<m sum above. For the orthogonal complement
of Q, s_k=0; summing its first term gives `(I-QQᵀ)Ec`, and its other term gives
`sum q_i(q0ᵀEv_i)/s_i`. Thus the formula covers positive discarded singular
values as well as structural rank loss. For R empty require U=0, giving D=0.
Repeated singular values inside the retained cluster are harmless; only
retained/discarded gaps appear in denominators. Gradient is Jᵀr_w.

For full retained rank this is the usual exact variable-projection residual
derivative, including the residual-dependent second term. Dropping that term
would be another approximation, not implemented here.

## 2. Adjoint and geometry

J maps horizontal Z (`ZBᵀ=0`) to `-sqrt(w)D`; for arbitrary input compose the
formula with `Pi_B Z=Z-(ZBᵀ)B`. For the adjoint start with `y=-sqrt(w)v` and
`a=Qᵀy`. In rotated E coordinates, the coefficient matrix W is

```
W_i = (y-Qa) alpha_i/s_i + q0 a_i/s_i   (i retained; zero otherwise)
for each retained i / discarded k:
  h = (alpha_i a_k + alpha_k a_i)/(s_i²-s_k²)
  W_i += h s_i q_k
  W_k += h s_k q_i
```

Consequently `J* v=Pi_B sum_j (W_j V_jᵀ)ᵀ U_j`. Every step follows directly
by pairing the three terms for D with y; the sqrt mass appears exactly once.
This proves the adjoint identity without forming a JP×md matrix. Test both
the identity and residual finite differences, including genuinely positive
discarded singular values with small value defect.

Use the existing polar retraction `R_B(Z)=[(B+Z)(B+Z)ᵀ]^-1/2(B+Z)`. For
horizontal Z the small m×m factor is `I+ZZᵀ`, positive definite; first
derivative at zero is Z. No d×d projector is formed.

## 3. Direction, acceptance and certificate

At each step solve `min_Z ||r_w+JZ||² + lambda||Z||²`, lambda>=lambda_prox>0,
using matrix-free LSMR with an explicit augmented operator. With
`g=J*r_w`, the equation is `(J*J+lambda I)Z=-g`. Verify the original-coordinate
residual `e=-g-(J*J+lambda I)Z`, `||e||<=0.1lambda||Z||`. An inaccurate solve
may be refined twice at the same lambda using the explicit zero-centered
augmented problem; all iterations share max(50,5md) budget. If it still fails,
increase lambda by4 and retry, at most12 trials, then fail explicitly.

The projected operator makes g horizontal. Exact uniqueness implies Z is
horizontal; project numerical Z and recalculate its certificate. Since
`-gᵀZ>=||JZ||²+0.9lambda||Z||²`, every nonzero certified direction descends.
Start Armijo with `t=min(1,0.25sqrt(m)/||Z||)` and backtrack by1/2, at most30
trials, retaining the same rank pattern and defect guards at every endpoint.
Accept iff `f(R_B(tZ))<=f(B)+1e-4 t gᵀZ`; no fake zero step. The linear
certificate belongs to the computed direction Z, explicitly separate from
the scaled accepted displacement tZ. Reject nonfinite, non-descent directions.

Predicted reduction for the accepted scaled direction uses the data model
`-t gᵀZ - t²||JZ||²/2`; if the actual/model ratio>0.75 halve lambda down to
lambda_prox, if<0.25 double it. This changes the numerical direction only;
lambda never enters the statistical objective. lambda_prox floor prevents
vanishing damping from creating an unnecessary precision problem.

Reuse the original independent stationarity definitions and two consecutive
actual accepted steps with small relative loss change, subspace distance,
global/local gradients and orthogonality. Exactly zero gradient at entry is
reported as stationary_initial without fabricating two steps. No global
minimum or empirical recovery follows from this certificate.

## 4. Conditional convergence, truncation, finite arithmetic

Assume finite data, positive mass, iterates confined to a compact subset of
the fixed-rank spectral domain with a uniform spectral gap; exact derivative;
uniformly bounded J; positive lambda floor and finite upper bound; successful
linear solves with the given forcing bound. Writing M=J*J+lambda I gives
`||g||<=(||M||+0.1lambda)||Z||` and
`-gᵀZ>=0.9lambda||Z||²`. Conversely `||Z||<=||g||/(0.9lambda)`.
Thus direction angle and size satisfy uniform gradient-related bounds.
Smooth polar retraction and Lipschitz gradient on this compact set give the
descent lemma. Armijo and the radius cap have a uniform positive lower bound
on accepted t (g bounded). Summable decrease therefore implies ||g||->0.
Accumulation points are stationary for the truncated objective. Exact joint
stationary correspondence holds only when no positive singular value is
discarded; otherwise retain the old explicit defect bounds.

These are conditional conclusions. Finite caps, loss of gap, huge retained
condition number, or floating-point error can yield an explicit failure or
nonconvergence. Rank/defect guards are runtime checks, not a proof of global
compactness. No fallback estimator is used.

For full-rank centers, batched SVD and residual construction are algebraically
the old reference; defects vanish. Only rank-deficient centers need the old
audited reference to check exact/truncated value/gradient defects. Scaling a
center's normalized gradient defect back to absolute units, summing their
norms, then dividing by the whole problem scale is a conservative triangle
bound because full-rank centers have zero defect. It can reject more inputs
than the exact summed-vector check, but cannot weaken the guard. Keep the
absolute retained-singular guards and original tau.

Memory beyond input U: O(JPm+Jm²+Jmd+JP+md), with fixed m and constant Krylov
vectors; operations per Jacobian pair O(JPdm+JPm²). No d×d, (md)² or JP×md
design. Batched low-dimensional arrays depend on J/P/m, not quadratically d.

## 5. Separate audit and counterexample checklist

|Statement|Algebra recheck|Runtime/reference evidence required|
|---|---|---|
|Same f_tau|r=(I-P_R)I; orthogonal projector has same SVD cutoff.|Audited scalar evaluator vs batched value/coefficients/ranks.|
|Full dr|Differentiate both off-diagonal pieces of dP; omitted q0 term would fail FD.|FD several h; nonzero residual; full/structural/truncated rank.|
|Adjoint|Pair each finite sum and complement term, then E=UZᵀ.|Random inner product test; dense Jacobian from basis vectors.|
|Mass/sign|r_w=sqrtw r, derivative -sqrtw D, gradient J*r_w.|Objective directional FD, rotated-basis invariance.|
|Ridge/forcing|Explicit penalty, eta0.1, final horizontal projection.|Dense augmented reference and original residual.|
|Accepted displacement|tZ may differ from certified Z; certificate names them distinctly.|Real two-step certificate, monotone loss, backtrack exhaustion.|
|Rank domain|Only cross-cluster gaps, rank0 iff U=0.|Ill-scaled/rank-change rejection, tiny discarded component.|
|Defect guards|Center absolute scales restored; triangle bound before global normalization.|Compare old normalized defects on mixed-rank problems.|

Counterexamples considered: omitting residual term changes J even at full
rank; positive discarded singular value can create large objective defect;
crossing cutoff destroys smoothness; an exact stationary entry cannot count
as two accepted steps. Each is tied to a check above. The old observed local
minima may still trap GN; the frozen objective gate decides that empirically.
