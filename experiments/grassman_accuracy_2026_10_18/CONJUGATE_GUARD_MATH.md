# PR+ direction at a rank guard

The pure spectral pilot was cheap at ten steps but improved EDR by only
2.25% on90800; fifteen steps improved5.85% with24.4% median extra time.
This new APPROXIMATE proposal reuses gradient history to accelerate that
spectral path. Smooth core GN and its initial five-step cap are unchanged.

For current orthonormal Y, projection transport is T(H)=H-Y(Y^T H).
At an unchanged local-rank pattern use Polak-Ribiere-plus
beta=max(0,<G,G-T(G_old)>/||G_old||F^2), D=-G+beta*T(D_old).
Reset to D=-G at a rank-pattern change, missing/invalid history, or unless
<G,D> < -1e-3*||G||F*||D||F. This margin guards an actual descent proposal.
The small SVD of -D defines the joint geodesic with maximum angle t. Its
true-gradient slope is <G,D_q>/sigma1(D_q), NOT minus a squared singular
norm of the search vector. Cache only the actual rank-q tangent direction.
History costs O(dm+J); transport and inner products require no new U pass.

All trials refit ALL centers and must pass the original true-profile Armijo
test. Original rank cutoff, final live gradient/refit, convergence and
applicability flags remain. At numerical rank boundaries neither the old
gradient nor this direction is a derivative certificate. Projection
transport/PR+ with this bounded Armijo-only search is a heuristic; no
global convergence or recovery theorem is claimed.

Independent tests cover transport/beta/reset/descent, actual slope and
full direct least squares. Pilot90800 compares5/10/15 before freezing the
budget and fresh103800 development/104800 validation. Previous consumed
data are not reused as independent evidence; same fixed gates including
2x individual EDR outlier limit,10% median and25% p90 time limits.
