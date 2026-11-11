# Direct Armijo: proof and frozen accuracy protocol

Independent hypothesis after LOCAL10_REPORT.md (accuracy pass/time fail) and
CONSTANT_REPORT.md (time pass/accuracy fail). No consumed development data
is reused for candidate choice. Exploratory seed90800 is a pilot only.

Candidate ADP-Grassmann-armijo: original profile/gradient/SVD planes,
GN and spectral guards; initial smooth profile min(requested_steps,5),
initial nonsmooth profile requested_steps=10. On rank-one fallback choose
original a and signed horizontal v, then test t=.5*2^-k, k=0..19, on the
full true profiled objective until the original Armijo1e-4 test passes.
This removes bounded scalar minimization and its Schur profile setup.
No changes to estimator, tau0, data, statistics or outer settings.

With ||a||=||v||=1, Y^Tv=0, Y(t)=Y+[(cos(t)-1)Ya+sin(t)v]a^T stays
orthonormal. Cached M(t)=UY(t) follows by linearity, and final UY is
recomputed. At a differentiable fixed-rank point, F'(0)=v^TG a<0 by the
chosen sign; hence sufficiently small steps satisfy Armijo under the usual
local smoothness assumption. At a rank boundary only actual value-based
acceptance is claimed; no differentiability/global-convergence claim.
Every accepted step decreases the full QR/SVD profile, up to floating-point
evaluation; a failed20-trial search returns its failure status. Full local
rank/conditioning guards remain unchanged for GN. Finite-budget changes are
APPROXIMATE and remain explicit opt-in until independent paired validation.

Memory: same O(J p (m+q)) caches and bounded core Jacobian as baseline;
no dense d*d matrix. Two large operator passes per rank-one iteration;
additional steps can cost time even if line searches are cheaper. Only
whole-fit measurements support the runtime claim. The true profile avoids
the earlier Schur tiny-column rank-cutoff defect on rank-one trials.

Frozen comparison: same original five-step ADP-Grassmann and real R
MAVE::MEANMAVE (MAVE-package), cases d-5,d-20,d-50, n500,m2, ten paired
datasets each. Development95800..95809; heldout96800..96809 untouched until
development passes. One BLAS/OMP thread, float64; unchanged adp_settings and
method seeds; 300s/4096MiB per process. Raw rows, metadata, source snapshots,
environment and input hashes are retained by experiments.r_sdr.

R3/R4 gates unchanged: complete30paired successful baseline/candidate fits,
no additional failures; geometric mean paired EDR ratio<=.95; no case
median paired EDR delta>.005; median/p90 paired fit-time ratios<=1.10/1.25;
median RSS ratio<=1.10; finite bases and orthogonality<=1e-10. Report stops
and certificates separately, and paired bootstrap uncertainty without
discarding failures. On failed development, close candidate and keep
validation96800..96809 untouched. No tuning on selection failures.

    UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.r_sdr --profile full --cases 'd-5,d-20,d-50' --runs 10 --seed 95800 --methods ADP-Grassmann,ADP-Grassmann-armijo,MAVE-package --grassman-steps 5 --threads 1 --keep-inputs --output benchmarks/results/outputs/r_sdr/grassman-armijo-dev-2026-10-18
