# H6: certificate-directed precision for the existing ridge subproblem

Classification: explicit APPROXIMATE option `hybrid_inner_rtol=0.01`; existing
default remains unchanged. This is a numerical solve of the same ridge target,
not statistical regularization. H6 fixes `dense_max_unknowns=dense_max_bytes=0`
to avoid the measured setup cost; both are existing settings.

Let `A: R^(md)->R^(JP)` be the mass-weighted joint design, `b` the current
weighted residual, lambda>0, `H=AᵀA+lambda I`, `g=Aᵀb`, and
`q(x)=1/2||Ax-b||²+lambda/2||x||²`. A has exactly the existing actions.
Let `D=diag(AᵀA)`, `S=diag(D+lambda)^(-1/2)`. Since lambda>0, S is invertible.
Minimizing `||[A;sqrt(lambda)I] S z-[b;0]||²` and setting x=Sz is exactly q.
The explicit penalty is essential: a nonzero x0 with implicit damp must not
shift the center of the penalty. Existing HYBRID already uses the correct
explicit augmented operator, with an adjoint S(Aᵀv+sqrt(lambda)w).

Define `e=g-Hx`. Runtime certificate is `||e||<=eta lambda||x||`, eta=0.01.
Since H>=lambda I, the unique exact solution x* satisfies
`||x-x*||<=||e||/lambda<=eta||x||`. This is independent of LSMR's own stop code.
With q's gradient at zero -g,
`gᵀx=xᵀHx+xᵀe >= ||Ax||²+(1-eta)lambda||x||²`, and the unregularized frozen-C
data loss changes by `-1/2||Ax||²-lambda||x||²-xᵀe<0` for x!=0.
Gauge preserves prediction exactly; exact local minimization can only decrease
loss further. Current truncated refit/cutoff is unchanged, so the actual
post-refit loss is still checked by the old acceptance test. No equivalence
to exact elimination is claimed when a positive singular value is discarded.

Choose the initial inner tolerance using a conservative size estimate:
`L=||g||/(sum(D)+lambda)<=||x*||`. A recycled x supplies the other valid bound
`||x||-||e||/lambda<=||x*||`. Use the larger nonnegative bound. For the scaled
augmented matrix K, `||K||_F=sqrt(md)` and `||S^-1||=sqrt(max(D)+lambda)`.
For zero start the optimal augmented residual is <=||b||, so
`atol≈0.1 eta lambda L/(sqrt(max(D)+lambda)*sqrt(md)*||b||)` is a conservative
starting heuristic. LSMR norm estimates, warm starts, compatible-system and
finite-precision stops mean this heuristic is **not a certificate**. Clamp it
to [eps,1e-3]; permit values below the former1e-10 floor. Always recompute e.

If the original certificate fails, tighten atol proportionally to
`0.1 eta/observed_ratio` and resume the *same* explicitly augmented problem with
x0=x/S, at most twice and within the original total Krylov iteration budget.
Each refinement minimizes the identical q, with the same zero-centered penalty.
Failure to reach the certificate remains a failed trial for existing HPAO;
no successful fallback or artificial zero step is introduced. In particular,
no assertion is made that every ill-conditioned input can be certified.

Conditional nonlinear statement: on a compact rank-stable region with exact
local minimization, bounded A and coefficients, accepted lambda bounded above
and below by positive constants, finite successful inner solves and exact
arithmetic, descent implies summable ||x||² and hence x->0. Then
`g=Hx+e->0`; local gradients vanish by refit and horizontal B-gradient tends
to zero. Numerical rank changes/finite budget invalidate these assumptions;
the old two-step common certificate and failure flags remain authoritative.
This is stationary convergence under assumptions, not global basin/recovery.

Extra storage: existing O(Jd+JP+md) workspace plus a constant number of Krylov
vectors; no new dimension-dependent dense matrix. At most two refinement calls
share the original iteration cap; all iterations and refinements are reported.
lambda=0 follows the old non-relative path; no division by zero. Zero g gives
the original zero solution/certificate. Nonfinite data/result remains an error.

## Separate algebra and implementation audit (before patch)

|Claim|Recheck / assumptions|Guard / test|
|---|---|---|
|Objective unchanged|Substitute x=Sz in both data and penalty, S invertible forlambda>0.|Dense augmented reference with extreme scales/masses/ridge; nonzero recycled start.|
|Adjoint|Inner product expands to zᵀS(Aᵀv+sqrt(lambda)w).|Existing operator/adjoint checks plus correction reference.|
|Error/descent|H inverse norm<=1/lambda; sign e=g-Hx checked explicitly.|Original-coordinate residual<=eta; no changed acceptance.|
|Tolerance is heuristic|Frobenius bound holds; warm start can invalidate residual bound.|Always certificate; refinement test forces an early inaccurate solve.|
|Refinement invariant|Explicit [A;sqrtlambda I], damp absent, x0 only starting vector.|Test against dense solution and zero-centered ridge.|
|Finite termination/status|At most2 refinements, cumulative iterations<=maxiter.|Iteration exhaustion must stay uncertified; no fake convergence.|
|Nonlinear theorem conditional|Rank-stable compactness and lambda bounds are assumptions, not runtime theorem.|Common evaluator; capped tasks recorded as nonconverged.|

Counterexample addressed: a large component of b orthogonal to range(A)
permits LSMR backward-error stop while lambda||x|| is tiny; R1 demonstrates it.
Unchanged default path must retain original calls/tolerances/certificates.
