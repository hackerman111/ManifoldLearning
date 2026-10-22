# Constant-image centers and a possible GN proposal

Read-only Luna diagnostic on frozen001..014 (J300 each):43centers have
rank(M)<m, and every one also has numerical rank(U)<m with rank(M)=rank(U).
Projected-only rank losses:0. Counts per call:10,8,6,3,2,3,4,1,1,2,2,0,0,1.
Twenty have exactly zero U/M spectra. Among the other23, first discarded
sigma/cutoff is.00654-.1497 for U and.0180-.3097 for M. Seven projected
spectra lie within10x of cutoff; retained leading singulars are far above it.
These are numerical-rank observations, not exact algebraic-rank proofs.

At tau0, if exact rank(U)=rank(UY)=r<m, image(UY)=image(U). Whenever this
rank persists in a neighborhood, the least-squares projection loss is locally
constant, although minimum-norm coefficients vary. Its residual Jacobian can
be set to zero; the profiled GN model for other regular centers remains valid.
The exact-zero U case is already a special case of this argument.

For finite-cutoff ranks the omitted singular tail is nonzero, and the
implemented numerical projector can move or change rank. A model that omits
those centers is therefore only an APPROXIMATE direction proposal, unless
stronger exact/perturbation bounds are established. It must never claim a
full-rank derivative certificate. Positive ridge also invalidates the
constant-profile argument. An optional proposal may use regular-center GN,
check its slope against the existing full gradient and accept only by the
unchanged FULL true-profile Armijo test, retaining original rank-boundary
certification semantics. This would change the proposal, not the estimator.

Current intrinsic.py pilot, separate from rejected quadratic candidate: identify
numerically intrinsic centers from U on the few rank-deficient rows; require
all remaining centers pass original smooth guards; build GN on regular rows,
refit every trial on all rows, otherwise retain original value-based fallback. Compare
against independent fixed-rank references and actual full fits. No such
proposal was implemented or used in the completed97800 quadratic run. The
new proposal is isolated; pilot compares5/10 and freezes10nonsmooth/5smooth. No production
default is changed. `_intrinsic_model` lives in backtracking.py, with default
intrinsic_gn=False preserving old Armijo/quadratic paths. Rank(U) is cached
only inside one frozen solve; U SVD is chunked at4MiB input scratch (or a
single center if larger). Every regular-center profile must pass the existing
smoothness guard. Accidental projected loss, guard failure or no regular rows
retain the old rank fallback. Full-gradient slope, damping, polar/Armijo,
workspace/spectral fallback, final loss/ranks and original certificate remain.

No discarded-tail error bound is claimed. The local constant-image theorem
explains the model at exact rank; the numerical classifier and finite steps
define an APPROXIMATE optimization proposal, never an estimator change or
proof of full-rank smoothness. Near-cutoff centers are specifically retained
in all true trial/final profiles. Omitted constant residuals add only an offset
to the idealized GN least-squares model, so do not affect its minimizer.
