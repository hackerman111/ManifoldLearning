# Local scalar profiles and accuracy budget

Classification: EXACT/NUMERICAL scalar evaluation, APPROXIMATE increased
optimizer budget. The estimator and the full-rank GN guard are unchanged.

For orthonormal Y and a unit horizontal v, unit a, let
Y(t)=Y+[(cos(t)-1)Ya+sin(t)v]a^T. Thus Y(t)^TY(t)=I and
M_j(t)=M_j+[(cos(t)-1)M_ja+sin(t)U_jv]a^T.

Choose A spanning a's orthogonal complement. Eliminate the constant columns
C_j=M_j A using their augmented QR/SVD, including sqrt(tau) A when tau>0.
Write the remaining projected data columns as z_j, u_j, w_j. With
k=||z||², b=z^Tu, c=z^Tw, alpha=||u||², delta=u^Tw, eta=||w||²,
the reduced minimum is

    f_j(t) = k_j - (b_j cos(t)+c_j sin(t))² /
             (alpha_j cos²(t)+2 delta_j cos(t)sin(t)+eta_j sin²(t)+tau).

This is the existing production formula. Its constant-column rank and
denominator guards apply independently to each center. The sum is separable:

    F(t) = sum_{j in safe(t)} mass_j f_j(t)
           + profile(M_unsafe(t), I_unsafe, mass_unsafe, tau).

Evaluating unsafe centers by the existing direct QR/SVD preserves the same
finite-sketch objective; it does not discard or regularize a center. A center
with U_j=0 contributes the constant mass_j ||I_j||². A rank boundary in any
other center is still evaluated by the original value-based fallback. The
full GN Jacobian is still forbidden whenever its original smoothness guard
fails. No differentiability across changing ranks is asserted.

The scalar expression can cancel near zero residual. A further finite-rank
guard is required: with M=[e_0,1e-15 e_1], a=e_1, I=e_1, tau=0, the
unprotected formula gives zero at t=0 while lstsq(rcond=None) drops the
second column and gives loss one. This defect also exists in the old scalar
path when its global guard is safe. The candidate uses the conservative bound

    S = ||C_aug||_F² + (|cos(t)| ||u_raw|| + |sin(t)| ||w_raw||)² + tau.

Only centers with both Schur denominator and sigma_min(C_aug)² exceeding
sqrt(eps)*S use scalar elimination; all m-1 constant columns must be retained.
The triangle inequality avoids cancellation in S. At m=1 the empty constant
block imposes no constraint. For the bounded small dimensions in this study,
these guards keep the full design away from the direct numerical rank cutoff;
they are a conservative numerical policy, not a theorem for arbitrary p/m.

Every accepted step is checked with the full true QR/SVD
profile and original Armijo slope. Final UY, gradient and orthogonality are
recomputed by the original solver. Cached/live objective discrepancies and
independent augmented-lstsq references bound finite-precision error in tests.

Cost: the Gram setup remains O(J p m²); each scalar evaluation uses O(J)
arithmetic plus direct profiles on J_unsafe centers, rather than all J.
Storage is O(J) additional masks/scalars, and the original O(J p m) profile
caches. There is no ambient d*d allocation. Increasing max_steps is a changed
approximation budget, so it is an explicit experiment variant. Better inner
objective or more steps does not establish better adaptive EDR recovery.

Sources: SVD/glm.tex:621-638 (directions and angle profile proposals),
ADP/solver/grassman.py:213-268 (implemented scalar profile),
diagnostics.json and frozen/ (actual R-SDR input diagnostics).

## Direction search pilot

`directions.py` also tried each retained gradient SVD pair independently,
using the same scalar minimizer and full true-profile Armijo test, then chose
the lowest accepted profile. This includes the original rank-one plane up to
roundoff, so its best one-step objective cannot be worse under successful
identical searches. There is no corresponding multi-step/EDR guarantee.
On exploratory seed 90800 it cost 1.08–1.49 times baseline at five steps;
eight steps cost more. It was rejected before frozen selection. The selected
candidate retains the original direction and uses ten steps, enabled by
local scalar fallback savings. See pilot.json and pilot_directions.json;
single-seed pilot recovery is not validation evidence.

Independent audit by GPT-6 Luna checked ridge augmentation, rank domains,
separability and cache scale; its cutoff/cancellation findings led to the
strengthened guard above and independent reference tests.
