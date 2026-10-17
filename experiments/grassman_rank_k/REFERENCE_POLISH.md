# Separate reference-quality supplement

The original frozen CG reference has1000 iterations/start and is preserved.
Some endpoints miss its tol1e-9 certificate. Before fitting the rank law, add
one bounded independent reference pass on these cases only, after all primary
timed workers finish. This changes neither a tested solver nor its budgets.

For an orthonormal chart center Q0, set C=Z-Q0(Q0.T Z), B=Q0+C.
Then B.T B=I+C.T C >= I: the chart basis always has full column rank, without
constructing an ambient projector. At tau=0 the profiled residual depends
only on col(B), so F(B)=F(QR(B)). Its Euclidean envelope derivative is
-2 sum_j N_j U_j.T r_j g_j.T. The derivative in the fixed chart is its
horizontal projection relative to Q0. This follows by the chain rule for the
linear map Z->C; no QR derivative is needed. Full-rank local systems are
required for this smooth derivative, and final Riemannian gradients are
checked independently at QR(B). Positive-ridge objectives are not invariant
under general basis changes and are outside this supplementary routine.

Run SciPy L-BFGS-B with analytic chart gradient, max5000 iterations/start,
ftol1e-15, gtol=.01*1e-9*profile_scale, maxcor20/maxls50. Then apply the
original independent Riemannian CG for at most1000 more steps/start and
evaluate the full-gradient certificate. Starts are the same Y0, truth and
fixed random frame as before; each retains its objective/gradient/stop.
Timeout180s per three-start case; retain failures and nonconvergence.

The minimum observed F is still only a numerical upper reference. Record
both original and supplementary results; do not discard an unfavourable
rank-law prediction. Compare original fitting with the improved references,
and separately report the cohort with a passed full-gradient certificate.
F>=0 also gives a useful absolute reference-error bracket when F is tiny.
Finite-difference/gauge/full-column-rank tests must pass before this stage.
