# Current state

Task grassman-lowrank-2026-10-22: DONE as bounded requested experiments;
requested stable accuracy improvement NOT achieved. Production grassman.py
and SVD.py unchanged, no API/default/estimator change. No commit/push.

User requested own error-reduction variants, LoRA-like factors and joint
rank-k SVD correction; Luna routine work. Implemented isolated APPROXIMATE
proposals in experiments/grassman_accuracy_2026_10_22/lowrank.py:
SVD block joint k*m fit, LoRA two-factor ALS/LSMR, normal-residual plane
enrichment. All use fixed-g surrogate only for proposals, horizontal Delta,
polar/reprofile/original Armijo and all-center live final certificate.
Full-k here min(m,d-m); rank applies BEFORE retraction. Literal SVD.py greedy
control has rank<m restriction. Helpers reject strided U to avoid full copies.
Additional own projected-only profile Jacobian preserves J.T r, removes PSD
curvature; NOT an exact derivative. LOWRANK_MATH.md gives shapes/proofs/limits.

Evidence:385/385 fits status ok,77 unique datasets (7 pilot90800,70 dev),
actual R MEANMAVE. Raw benchmark_outputs/r_sdr/grassman-lowrank22-
{pilot,projected-pilot,development}/; frozen source/input/method-seed checks
passed. LOWRANK_REPORT.md + lowrank_analyze.py + paired_analysis.json.
Pilot LoRA_k10 EDR/base.990,time1.017 (single-seed only, not independent gain);
rank1 worse, full-k20 slower/worse; enriched10 near-neutral/faster, d50 worse.
P10 pilot .9859/.723 selected and frozen before new seeds.

Development111800..809: baseline20,budget10,P10,R on same7cases,280fits.
P10 EDR/base1.00065, bootstrap95[.98131,1.02227], fit median/p90 .737/.885;
budget10 .99895,[.97675,1.02411], .742/.941. P10/budget10 EDR1.00169,
CI[.98859,1.01678], median time.990. Both fail frozen quality and case/individual
regression gates. No stable accuracy or direction benefit. All ADP outer stops
h_min, orthogonality <1.4e-14; no convergence guarantee. No production promotion.
Validation112800..809 remains UNTOUCHED; do not retune to it after failed gate.

Verification:214 focused tests pass incl Grassmann, SVD core/gradient/metric,
independent dense operator/adjoint/coarse ridge, rank/unit invariance, polar
majorization, zero/rank-deficient centers, capped-LSMR diagnostics, strided-U
rejection; augmented-QR/gradient/PSD projected checks; analysis gates/failures/
finite/zero recovery/budget ablation. Ruff and Pyright clean; diff --check clean.
Reference and production bytewise equal SHA3a0807d0d31afded0705890a2478cdd5
39de1f1f553831dcf593861ddb3acda6. New prototypes CPU only; CuPy untouched.
Affected routes: ADP/solvers.md and ADP/multi-index.md. PLAN preserves gates,
DECISIONS preserves frozen selection/failure. Prior nine-arm screening and
incomplete GPU heavy-quality task remain archived (see earlier report/plan).

This bounded protocol is complete. Future continuation needs a new
mechanism/protocol and fresh independent validation; do not claim
pilot improvements or internal loss reductions as recovery improvements.
