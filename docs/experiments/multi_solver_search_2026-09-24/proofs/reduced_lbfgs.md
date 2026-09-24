# Reduced multi-index objective: proof for a guarded Riemannian L-BFGS variant

Status: mathematical derivation for S2, **not** an implementation or a claim that the solver succeeds on all ADP inputs. The theorem below is conditional on an explicitly checked rank-stable domain. The exact joint least-squares problem and the current numerical `rcond=None` objective are distinguished throughout.

## 1. Problem and domain

For fixed real `U_j∈R^(P×d)`, `I_j∈R^P`, positive finite `w_j`, and `B∈R^(m×d)` with `BBᵀ=I_m`, let `A_j(B)=U_j Bᵀ`. Define

```text
F(B,C) = (1/2) Σ_j w_j ||I_j - A_j(B)c_j||².
```

The production local refit (`ADP/solver/LSMR.py:429–489`) uses an SVD/lstsq relative cutoff `τ=eps(float64) max(P,m)`. Let `A_j=Q_j diag(σ_j) V_jᵀ` be a thin SVD with decreasing `σ_jk`, and `r_j=#{k:σ_jk>τσ_j1}` (`r_j=0` if `A_j=0`). The implemented minimum-norm coefficient is `c_j^τ=Σ_{i≤r_j}(q_jiᵀ I_j/σ_ji)v_ji`; its prediction is `P_j^τ I_j`, `P_j^τ=Q_jr Q_jrᵀ`. Thus the **implemented** reduced objective is exactly

```text
f_τ(B) = F(B,C^τ(B))
       = (1/2) Σ_j w_j (||I_j||² - ||Q_jrᵀ I_j||²).
```

The equality uses orthogonal projection and holds even when a positive singular value is truncated. It preserves the existing mass, cross terms, data, and cutoff; `lambda_prox` is absent from `F` and only belongs to the old correction algorithm. A new finite-iteration optimizer for `f_τ` is classified `APPROXIMATE`; changing the cutoff or using ridge local refits would be `ESTIMATOR` and is outside this plan.

The proof domain `D` is an open subset of the row-Stiefel manifold where, for each j, the retained rank `r_j` is constant, every retained singular value is positive, and the squared-spectrum gap `σ_j,r²-σ_j,r+1²` is positive (`σ_j,m+1=0`). If `r_j=0`, require `U_j=0`; then its contribution is constant. The rank pattern and a strict gap to the numerical cutoff are checked at the initial point and every proposed accepted point. The convergence theorem additionally assumes the iterates lie in a compact sublevel subset of `D` with uniform positive gap; a runtime check at finitely many points cannot prove that global condition. If a guard or line search fails, return an explicit failure, never a convergence claim.

## 2. Exact elimination and stationary correspondence

In exact arithmetic with the Moore–Penrose pseudoinverse, `c_j^+=(A_j)^+ I_j` minimizes each local quadratic because `A_jᵀ(I_j-A_j c_j^+)=0`; the minimum value is `f_+(B)=F(B,C^+(B))`. Since all `w_j>0`, local minimization separates. For any orthogonal `R∈O(m)`, `A_j(RB)=A_j(B)Rᵀ` and `c_j^+(RB)=R c_j^+(B)`, so predictions, `F`, and `f_+` are basis-invariant. The same conclusion holds for `c_j^τ` because right rotation preserves singular values and the retained left spectral projector.

On a neighborhood where each `rank(A_j)` is constant, the pseudoinverse and residual are differentiable. Differentiating `F(B,C^+(B))` gives the envelope formula

```text
∇_B f_+(B) = -Σ_j w_j c_j^+ (U_jᵀ r_j)ᵀ,
r_j = I_j - A_j c_j^+,
```

because `A_jᵀr_j=0` kills the derivative through `c_j^+`. If `A_j` is rank-deficient, every minimizing coefficient is `c_j^++v_j` with `A_jv_j=0`. Along any smooth curve in the constant-rank neighborhood there is a differentiable null vector `v_j(t)` through `v_j`, so differentiation of `A_j(t)v_j(t)=0` gives `(dA_j)v_j∈range(A_j)`. Hence `r_jᵀ(dA_j)v_j=0`: the B-gradient is independent of which local minimizer is chosen. A pair `(B,C)` with C minimizing each local quadratic is first-order stationary for the joint constrained problem exactly when B is stationary for `f_+` on Grassmann. This is a **stationarity** statement, not global optimality or recovery.

If a positive singular value is discarded, `c_j^τ` need not minimize the exact local quadratic, so this exact correspondence does **not** apply. In exact arithmetic, writing `K_j={k:σ_jk>0}` and `α_jk=q_jkᵀ I_j`, the value defect is the exact nonnegative identity

```text
f_τ(B)-f_+(B) = (1/2) Σ_j w_j Σ_{k∈K_j, k>r_j} α_jk².       (1)
```

Where the positive-rank pattern is also constant, the gradient defect is `∇f_τ-∇f_+`, both given explicitly in §3; its norm is an a posteriori error bound by equality, and horizontal projection cannot enlarge it. The prototype must check value defect `≤1e-10 max(1,f_τ)` and normalized horizontal gradient defect `≤1e-8` at accepted endpoints. If tiny positive singular values make the exact comparison numerically unreliable or nonfinite, the guard fails. Within the subdomain where discarded singular values are exactly zero, `(1)` and the gradient defect vanish, and the exact stationarity correspondence above holds. Outside it, only the guarded approximate stationary claim is valid: a common gradient score `≤tol` implies the exact reduced gradient score is at most `tol+1e-8` with the same positive normalization scale. There is no uniform bound from `σ_discarded` alone: `α_discarded` may be large even when `σ_discarded` is tiny. This is the rank-truncation counterexample the guard addresses.

## 3. Derivative of the *truncated* objective

For one center suppress j, put `M=AAᵀ`, `λ_i=σ_i²`, `α_i=q_iᵀI`, and `P_r=Σ_{i≤r}q_iq_iᵀ`. Let `q_k` for `k>m` complete an orthonormal basis with `λ_k=0`. Differentiating `Mq_i=λ_iq_i` and projecting on `q_k` gives, for `i≤r<k`, `q_kᵀdq_i = q_kᵀ(dM)q_i/(λ_i-λ_k)`. Terms within the retained cluster cancel in `dP_r`; therefore

```text
df_j = -w_j Σ_{i≤r,k>r} α_i α_k
               q_kᵀ[(dA)Aᵀ + A(dA)ᵀ]q_i /(λ_i-λ_k).        (2)
```

This requires only a gap **between** retained and discarded clusters; repeated singular values within one cluster are harmless. Equation (2) also proves differentiability of `f_τ` on `D`. With `q_0=(I-Q_mQ_mᵀ)I`, its Euclidean gradient with respect to `A` is

```text
G_A = -w [ Σ_{i≤r} (α_i/λ_i) q_0 (Aᵀq_i)ᵀ
         +Σ_{i≤r,k=r+1..m} α_i α_k/(λ_i-λ_k)
             { q_k(Aᵀq_i)ᵀ + q_i(Aᵀq_k)ᵀ } ].               (3)
```

When `r=m`, (3) reduces to `-w(I-P_m I)(c^+)ᵀ`, the envelope gradient. When `A=0` because `U=0`, set `G_A=0`. Because `dA_j=U_j(dB)ᵀ`, the ambient row-basis gradient is

```text
G_B = Σ_j G_Ajᵀ U_j                  (m×d).                  (4)
```

All `w_j` appear exactly once in (3), not inside `c_j`. The local gradient of `F` at `C^τ` is `-w_j A_jᵀ(I_j-A_jc_j^τ)`; it is zero in the untruncated case and otherwise checked by the common certificate. No `lambda_prox` enters (2)–(4). A finite-difference check along horizontal tangent vectors at several step sizes is required before using this formula in timing.

## 4. Geometry, updates, and certificate

For row-orthonormal B, the horizontal tangent space is `H_B={Z∈R^(m×d):ZBᵀ=0}`. The orthogonal projector is `Π_B(Z)=Z-(ZBᵀ)B`, so the Grassmann gradient is `g=Π_B(G_B)`; dimensions are `m×d` throughout. Basis invariance gives the same critical subspace under `B↦RB`. For horizontal η, use the polar retraction

```text
R_B(η) = [(B+η)(B+η)ᵀ]^(-1/2)(B+η).
```

The inverse square root is only `m×m`. Since `ηBᵀ=0`, its argument is `I_m+ηηᵀ`, positive definite; `R_B(0)=B` and `DR_B(0)[η]=η`. Transport a history vector to the new horizontal space by `T_{B→B'}Z=Π_B'(Z)`. Orthogonal projection is norm-nonincreasing and smooth. History length is bounded by five `(m,d)` pairs. A two-loop L-BFGS direction is accepted only if it is finite, horizontal, `⟨g,p⟩≤-c||g||²`, and `||p||≤C||g||` for fixed positive `c,C`; otherwise set `p=-g`. Curvature pairs with nonpositive or too-small `⟨s,y⟩` are skipped. These safeguards make the convergence argument independent of unproved positive definiteness of transported L-BFGS history.

For a trial `t=1,1/2,1/4,...`, first check the rank/cutoff and error domain, then accept only if `f_τ(R_B(tp))≤f_τ(B)+10^-4 t⟨g,p⟩` and the objective is finite. A cap on backtracking causes explicit failure; it never creates a zero accepted step. Actual diagnostics also require nonincreasing objective up to the existing `64 eps max(1,F)` rounding allowance. The common evaluator outside the candidate recomputes `f_τ`, normalized old Stiefel/global gradient, local gradient, orthogonality, and rank; this avoids self-certification by (3).

To report `converged=True`, require **two consecutive accepted** steps each satisfying: relative objective change `<tol`, projector-aligned subspace step `<tol`, normalized common global/local gradients `<tol`, and orthogonality `<tol`. The linear residual certificate applies to HPAO only; this method has no linear correction. For L-BFGS, report Armijo/rank/error guards, objective history, gradient history, accepted/rejected evaluations, and reason for noncertification. A stationary point of `f_τ` by itself is insufficient to pass the common certificate when truncation defects remain.

## 5. Conditional convergence and its limit

Assume exact arithmetic; finite J,P,d; positive masses; a compact sublevel set contained strictly in `D` with uniform retained/discarded spectral gap; `f_τ` bounded below; exact (3); and infinitely many accepted iterations before a finite-budget stop. On this set `f_τ` has a Lipschitz derivative along the smooth polar retraction. The direction safeguard gives `-⟨g,p⟩≥c||g||²` and `||p||≤C||g||`. A retraction descent lemma gives

```text
f_τ(R_B(tp)) ≤ f_τ(B)+t⟨g,p⟩+(L/2)t²||p||²
```

for sufficiently small t, uniformly on the compact set. Therefore Armijo accepts a step bounded below by a positive constant depending on `L,c,C` and the backtracking factor. Each iteration decreases `f_τ` by at least a positive constant times `||g||²`; summing and using `f_τ≥0` proves `Σ||g_k||²<∞`, hence `||g_k||→0`. Every accumulation point in the domain is first-order stationary for `f_τ`. If all positive singular values are retained, the §2 equivalence makes it stationary for the exact joint problem. With guarded truncation, only the stated error-bounded approximate claim follows. This proof does **not** establish global minimization, a rate, full ADP recovery, or eventual satisfaction of the stricter two-step common certificate.

Finite precision, a fixed iteration budget, or a finite backtracking cap can break the theorem's premises. The implementation must expose that status and must not call it convergence. If the rank gap is lost, gradients/transport fail a finite check, objective increases, the error guard fails, or the line search exhausts its budget, reject the candidate explicitly. Zero/negative/nonfinite mass remains invalid as in `LSMR._validate_inputs`; zero U is a constant local term; nearly singular retained systems are rejected by the cutoff-gap guard. No hidden estimator change or fallback is permitted.

## 6. Work and memory; attempted counterexamples

One evaluation forms `A_j=U_jBᵀ` in `O(JPdm)`, thin local SVDs in `O(JPm²)`, and (3)–(4) in `O(JPdm+JPm²)`. Besides the existing input `U=(J,P,d)`, storage is `O(JPm+Jm²+hmd+md)` with fixed history `h≤5` and centerwise gradient scratch; no `d×d`, `(md)²`, `(J,n,d)`, or `(JP,md)` allocation. The HPAO reference and this candidate consume identical frozen U/I/mass/B and use the same result orientation `(m,d)`.

Two failure constructions constrain the claim: (i) a rank-one local design with an arbitrarily small but positive second singular value and `I` aligned with its second left singular vector has an arbitrarily large **relative** truncation value defect despite the small singular value; (ii) a path crossing the cutoff changes `r_j` and can make the derivative discontinuous. Formula (1), the gradient-defect guard, and the rank-domain guard explicitly address these. S1 found 1–21 structurally rank-deficient centers per task, so a proof assuming `rank(A_j)=m` for all j is inapplicable. On the 12 initial frozen points the observed discarded-energy identity is below `1.6e-31` in absolute value, and discarded/threshold singular ratios stay below 0.065; these are diagnostic observations, not assumptions of the theorem or evidence for all future iterates.

The implementation mapping and independent recheck of every critical equality/assumption are in `audit_reduced_lbfgs.md`. S3 is prohibited until that audit has been completed and any uncovered obligation resolved.
