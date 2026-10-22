# Joint spectral gradient direction at a rank guard

The existing rank-one fallback retains only the leading singular pair of
the horizontal gradient G. Instead use all available nonzero pairs at a
nonsmooth profile (or the explicit user rank cap), keeping smooth core GN
and its energy selection unchanged. No local full-rank residual Jacobian
or numerical intrinsic-rank classifier is used for the guarded proposal.

Write G_q=V S A^T after horizontalization. The existing spectral geodesic
rotates each plane (V_i,Y A_i) by -t s_i/s_1. Its derivative at zero is
-G_q/s_1, and its smooth-stratum objective slope is
-sum_i s_i^2/s_1 < 0. The maximum angle is at most max_angle. The small SVD
after QR recovers actual coordinates, so QR signs do not change the step.
All matrix operations are thin, with cached UY/UV and no dense d*d matrix.

At numerical rank boundaries the existing gradient is a proposal, not a
derivative certificate. Each trial refits ALL centers with the original
minimum-norm profile; full true values decide Armijo acceptance. Final
live refit, ranks, gradient and applicability/convergence flags are unchanged.
Positive local ridge remains the same explicit estimator option.

This reuses the old spectral step conditionally, not a novel spectral
algorithm. Standalone spectral has mixed historical fullfit evidence in
grassman_rank_k/REPORT.md. Neither those runs nor lower profile values prove
recovery benefits for the new hybrid and budget. Pilot90800 compares5/10/15
steps, initial smooth cap5, before fresh103800/104800 paired evaluation.
