# Preserve the baseline, then refine

The preceding intrinsic and one-step-warmup candidates can leave a poor
initial region within ten GN steps; both failed independent gates. Retain
the entire original five-step solve before proposing a refinement. This
preserves its first-phase trajectory exactly, including original bounded
scalar searches, damping, rank/workspace fallbacks and final live refit.

An additional small budget uses the same APPROXIMATE intrinsic GN model
with all-center true-profile Armijo acceptance. No original objective or
estimator changes. The exact constant-image argument and numerical-rank
caveats remain INTRINSIC_RANK.md. There is no theorem connecting lower
finite-sketch objective to improved statistical recovery.

Skip refinement at a certified baseline, positive local ridge, or an
initial nonsmooth state for which the intrinsic model is unavailable.
At a smooth final baseline state the ordinary full-rank GN model applies.
The final index, coefficients and certificate all come from the same final
live profile. Both phase histories and their boundary gap are retained:
renormalizing and refitting the baseline basis can shift values by roundoff.
The complete baseline five steps are not replaced or selected retrospectively.

Pilot90800 compares extra2/3/5 steps, then freeze one total budget before
fresh103800 development and104800 validation. Previous consumed data cannot
validate this candidate. Same quality/time/memory gates and2xoutlier bound.
