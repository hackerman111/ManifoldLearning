# Implement truncated-SVD multi-index training

task_id: svd-multi-index-2026-09-28
status: done
change_class: APPROXIMATE (rank constraint), explicit solver variant

Goal: implement the fixed-statistics rank-r algorithm of `SVD.tex` and make it
usable through the current `ADP_multi_index` custom solver interface.

Non-goals: change the default LSMR/HYBRID objective or solver, claim the greedy
algorithm is a global rank-r optimum, or infer missing m-r EDR directions from
a rank-r matrix alone.

Known evidence: `SVD.tex` defines F(B)=sum_j mass_j ||I_j-U_j B.T g_j||² +
lambda||B-P||² with fixed g_j and rank(B)<=r<m. Live multi fit requires a
full m-row orthonormal basis and local coefficients; existing LSMR uses a
correction penalty rather than this F.

Invariants: float64; positive finite mass; bounded O(J p d + m d + J p r)
working memory (no d-by-d Hessian); augmented matrix-free least squares for
the v step; conditional singular-value solve; full-rank public m-basis with
explicit prior-space completion; objective and rank diagnostics; no truth in fit.

Work units completed:
- Implemented rank-r fixed-g solve, compact SVD, exact conditional scale
  refit, stop rule, and diagnostics. Four focused tests passed.
- Exposed the solver via the custom-solver hook and completed missing basis
  directions from the prior; public fit test passed.
- Updated routing and decision notes; Ruff and `git diff --check` passed.

Smoke `(J,p,d,m,r)=(48,12,40,3,2)` reduced fixed-g F from 678.515 to 582.462
in 0.0183 s; process peak RSS was 70,296 KiB including Python startup. This is
not comparative speed or recovery evidence.
