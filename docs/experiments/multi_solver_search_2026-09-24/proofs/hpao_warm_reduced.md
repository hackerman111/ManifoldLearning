# One-step HPAO warm start followed by guarded reduced L-BFGS

Status: S4 proof completed before implementation for the **second and last** candidate. The subsequent [six-task experiment](../s4_final.md) rejected H5 empirically; the conditional mathematical claims below remain bounded by their stated assumptions. The proposal fixes exactly one HPAO AO step; no search over warm-start length. It composes the unchanged current HPAO correction with the [guarded reduced method](reduced_lbfgs.md), on the same U/I/mass/B and same SVD cutoff. This is an `APPROXIMATE` solver variant for the existing objective, not an estimator change.

## Claim and domain

Assume float64 CPU inputs with `U=(J,P,d)`, `I=(J,P)`, positive finite mass, full-row-rank initial `B=(m,d)`, `1≤m≤min(P,d)`, `lambda_prox>0`, finite tolerances and trust radius. For one HPAO AO step use the current weighted operator and local refit; then start the guarded reduced L-BFGS at its returned orthonormal `B₁`. The reduced phase requires the same rank/cutoff/error domain and conditional compact-sublevel assumptions as `reduced_lbfgs.md`. If HPAO cannot accept one certified step or B₁ fails that domain, fail explicitly. The two phases never replace U, I, mass, cutoff, kernel, or objective.

## Phase-one correction and descent

At `B₀`, production `_local_refit` computes `C₀=C^τ(B₀)`. For fixed C₀, `_linear_operator` implements the weighted design `Aδ` and its adjoint; `_global_correction` approximately solves

```text
min_δ ||Aδ-r||² + λ||δ||²,
r_j = sqrt(mass_j) [I_j-U_j B₀ᵀc₀j],  λ=lambda_prox>0.
```

Its normal residual `e=Aᵀr-(AᵀA+λI)δ` is recomputed in original coordinates and accepted only if `||e||/(λ||δ||)≤θ`; because `AᵀA+λI ⪰ λI`, the exact ridge minimizer δ* obeys `||δ-δ*||≤||e||/λ≤θ||δ||`. This controls the correction error but is **not** a guarantee of decrease of the reduced objective. HPAO separately rejects a step exceeding its trust radius, orthonormalizes the raw basis, transforms C so predictions are unchanged, refits local C at the new basis, and accepts only if `fτ(B₁)≤fτ(B₀)+64eps max(1,fτ(B₀))`. Therefore the accepted phase-one step preserves the implemented objective up to its documented roundoff allowance. On the selected experiments `λ=0.05`; `λ=0` is outside this composite proof and must be rejected for this variant.

This is a one-step statement. It does not assert convergence of HPAO after one AO step or identify a global basin. The returned phase-one `converged=False` is expected because HPAO itself requires two certified accepted steps; no fictitious certificate is created.

## Phase-two correctness and stationarity

Check `fτ(B₁)` independently against HPAO's final loss using the common evaluator before the second phase. Reorientation/QR of B₁ leaves the Grassmann subspace and fτ unchanged by basis invariance. The guarded reduced method then uses exactly (1)–(4), horizontal projection, polar retraction, projected transport, safeguarded L-BFGS direction, Armijo, rank/error guards, and the same common certificate as proved in `reduced_lbfgs.md`. Its conditional stationarity theorem applies from any admissible starting B₁, independent of how B₁ was obtained. With a finite 320-step budget or 60 s timeout, nonconvergence/timeout remains explicit.

Only the **reduced phase** counts toward the required two consecutive certified accepted steps; the HPAO warm step is recorded separately and does not satisfy one of those two. Final coefficients are the second phase's local minimum-norm SVD coefficients, and the result uses the original `HPAOResult` row-basis convention `(m,d)`. The final external evaluator recomputes objective, old normalized global/local gradients, orthogonality, rank and aligned step. The HPAO linear certificate remains attached to phase one; reduced has no linear subproblem. A `converged=True` composite output requires the reduced phase's full common certificate plus successful warm-step and domain checks.

## Objective and rank qualification

The staged objective is `fτ` throughout. Positive but discarded singular values can make `fτ` differ from the exact joint minimum; the value identity (1), explicit horizontal gradient defect, guard thresholds and counterexample in `reduced_lbfgs.md` apply unchanged at B₀, B₁, and reduced accepted endpoints. For exact structural rank with no positive discarded component, stationarity correspondence to the exact joint problem holds. Otherwise only the bounded approximate statement is warranted. A rank transition, zero/negative mass, near-singular retained spectrum, or failed defect guard is an explicit failure, never a hidden fallback.

## Cost, memory, and convergence limit

Total work is one HPAO correction plus reduced evaluations. The first phase adds its usual matrix-free `A/A*` actions and bounded `(J,P,m)` local SVD; the second adds `O(JPdm+JPm²)` per evaluation and history `O(5md)`. The phases are sequential and form no `d×d`, `(md)²`, `(J,n,d)` or full `(JP,md)` design. Peak extra memory is the maximum of the phases plus small retained outputs, not a sum of large workspaces. The existing HPAO code may allocate a `(J,P,d)` input; this is the shared frozen input, not new candidate memory.

The reduced convergence proof establishes first-order stationarity only under its rank-stable compact-domain assumptions; the warm step does not strengthen that theorem. Neither phase proves a better local minimum, global optimum, or ADP recovery. The S4 comparison must reject this candidate if it reaches a worse objective or costs too much despite passing a stationarity certificate.

Potential counterexamples: an HPAO warm step can move outside the rank-stable domain, in which case this candidate fails; a different stationary basin can still be reached after the warm step; a tiny discarded singular value with a large discarded response component invalidates exact-minimum equivalence; and the one-step correction itself can cost more than the saved reduced iterations. These are selection questions with the six-task budget in `hypotheses.md`, not theorem consequences.
