# Combined solver: invariants and scope

User request 2026-10-22 authorizes combining earlier joint time/EDR gains and
raising the compute budget, including effects below the old promotion gate.

The finite-sketch objective remains
F(Y)=sum_j mass_j min_g (||I_j-U_j Y g||²+tau||g||²), Y^T Y=I.
Statistics, numerical rank threshold and tau are unchanged.

1. **EXACT reduction.** If every entry of U_j is exactly zero, its minimum
   is mass_j||I_j||², attained by g_j=0 (minimum norm when tau=0).
   Its directional derivatives vanish for every Y. Solve the remaining
   centers, restore zero coefficients, constant loss/history and rank counts.
   Set reduced tol to tol*max(1,sum_all mass*I²)/max(1,sum_active mass*I²)
   so the absolute gradient stopping threshold remains unchanged. Exact
   constants do not invalidate smoothness of the reduced objective; final
   stationarity applicability describes that reduced objective. All-zero
   inputs retain the existing conservative rank-boundary diagnostic.
   No near-zero operator or positive-mass center is otherwise removed.
   Rounding/reduction order and normalized-vs-absolute damping selection can
   change the optimizer trajectory, even though its objective is identical.
2. **Relative GN.** Preserve the independently validated eligible-intrinsic
   classifier and mu=lambda_prox*||J||F²/(q*m). The regular-center model is
   only a proposal. Full active profile, gradient, Armijo and live refit
   remain authoritative. See RELATIVE_GN_MATH.md and INTRINSIC_RANK.md.
3. **Quadratic angle proposals.** At the true rank-one fallback, fit
   curvature c=2*(F(t)-F(0)-t*F'(0))/t² and propose -F'(0)/c in the existing
   bounded angle interval, at most three trials. Select only an actual
   refitted Armijo-accepted trial; retain halving fallback. This is an
   APPROXIMATE search, not a quadratic objective assumption or rank proof.
   See QUADRATIC_MATH.md; smooth GN and workspace fallbacks stay intact.
4. **Budget.** Compare combined10/smooth5 before increased budgets with the
   smooth cap removed. More optimization need not improve EDR after outer
   adaptive localization. Budget effect must be reported separately.

Inputs U:(J,P,d), I:(J,P), Y:(d,m). GN scratch is still bounded by the
existing 16MiB guard. Constant reduction copies only active input statistics
O(JPd), plus O(Jm) coefficients and a boolean center mask. It creates no
ambient d*d matrix. Peak process RSS is measured in paired full fits.

Independent checks: tests/test_grassman_combined.py compares direct rank-one
geometry, augmented local least squares, full loss, gradient scale and the
audited intrinsic prototype at m=1,2,3. No global convergence/recovery claim.
