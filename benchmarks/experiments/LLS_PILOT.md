# Frozen LLS recovery pilot, 2026-10-23

Question: does replacing production sketch moments by compressed local WLS
improve the EDR of the returned basis? This is an ESTIMATOR experiment.
Production ADP code, outer loop, step selection and solver remain unchanged.

Arms: base, phid (random square directions), identity (Phi=I), lls (user QR).
All use production combined20 Grassmann defaults pinned in the script.
Six user-specified n=500 cases, ten paired seeds 630000..630009.
Untouched confirmation seeds 640000..640009, base and lls only.
No a/h_min/N_loc/tensor/last sweeps and no truth-based selection of steps.

Gate fixed before selection: every fit valid; paired geometric LLS/base EDR
ratio <=0.95; upper 95% bootstrap CI <1. Bootstrap resamples seed blocks,
preserving all six cases in each block. Report per-case regressions as well.
Validation runs only if selection passes. Validation uses the same gate.
These are pilot intervals on a fixed synthetic catalog, not universal claims.
Failures remain in planned recovery denominators. Inner convergence, outer
stop, EDR<=0.1, fit/process time and RSS are separate fields.
One worker at a time, one BLAS/OpenMP thread, timeout120s, sampled RSS4096MiB.
The source archive and hashes, inputs, bases, full traces and errors are saved.

## Proof and limits

For fixed local weights w, mass=sum(w), a=w/mass, let
A=sqrt(a)(X-mu), b=sqrt(a)(y-ybar), and reduced QR A=QR.
Padded U=R and I=Q.T b give
||b-ABg||^2 = ||I-UBg||^2 + ||(Id-QQ.T)b||^2.
The last term is independent of B,g. This holds for tall/wide and
rank-deficient A because Q has orthonormal columns spanning range(A).
The solver's external mass recovers the original unnormalized weights.
At fixed B, coefficient elimination is the same minimum-norm LS problem;
the objective is invariant under B->BO, g->O.Tg for orthogonal O.
Numerical rank decisions and stopping thresholds can still affect a finite fit.
LLS changes the criterion from squared covariance moments to prediction RSS.
It does not prove statistical consistency or adaptive full-fit recovery.
Weights evolve differently after arms diverge. Absolute objective values across
arms and outer steps are not comparable; QR discards a weight-dependent constant.
With legacy normalized=False the supplied user convention scales I/U by mass;
the pilot uses only estimator=new, normalized=True.

R2 checks: covariance/cross-moment identities, constant RSS difference for
arbitrary coefficients including deficient and wide designs, full identity
moments, invalid weight rejection, and exact base replay of public fit.
An independent audit precedes selection. Full-fit output is the quality gate;
no claim that this isolates optimizer quality is made.
