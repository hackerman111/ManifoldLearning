# Constant-center candidate: proof and frozen protocol

Separate hypothesis from rejected LOCAL10_REPORT.md. No previous development
seed is reused. Frozen before development93800..93809; validation94800..94809.
Candidate ADP-Grassmann-constant; baseline ADP-Grassmann; R MAVE-package.
Cases d-5,d-20,d-50, n500,m2, ten paired datasets each; one BLAS thread,
float64, unchanged ADP config, seed scheme, five-step cap and tau0. Existing
R-SDR runner saves all source/input hashes, environment, process RSS and rows.

For U_j=0 exactly, min_g ||I_j-U_jYg||²+tau||g||²=||I_j||², with minnorm
g=0 for all tau>=0. This term has zero gradient and is smooth regardless
of the local design rank. Separate active centers from constant centers,
solve the original problem on active rows, restore zero coefficients in
original order and add C=sum_inactive mass_j||I_j||² to all objective/history
values (C/2 to loss values). At positive tau, augmented constant rows have
full local rank; at tau0 they have rank0. Nonzero low-rank rows are never
discarded and retain every original GN guard.

Keep the original stopping scale S=max(1,sum_all mass_j||I_j||²): pass
tol_active=tol*S/S_active to the existing solver, then report ||G||/S.
This preserves the original gradient threshold. An all-zero operator keeps
the original path. The objective reduction is EXACT; the changed finite GN
trajectory is APPROXIMATE and remains explicit opt-in. Source arrays are
filtered once, costing an extra O(J p d) copy plus O(J p d) Boolean scan;
at J300,p12,d1000 this is at most28.8MB plus3.6MB temporary (decimal MB),
with no ambient d*d matrix. No universal speed or EDR theorem is implied.

Independent GPT-6 Luna audit checked constant objective/gradient, scale and
coefficient restoration, and identified the positive-ridge rank diagnostic
fix. Independent direct-lstsq tests are tests/test_grassman_constant_centers.py.
Exploratory seed90800 is retained in pilot_constant.json, not validation.

Same R3/R4 gates as prior protocol: 30/30 successful pairs, no extra failures;
geometric mean paired EDR ratio<=.95; no case median paired EDR delta>.005;
median/p90 paired model.fit time ratios<=1.10/1.25; median RSS ratio<=1.10;
orthogonality<=1e-10 and finite outputs; all stopping/rank data retained.
Any failure closes the candidate, preserving validation94800..94809 untouched.

    UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.r_sdr --profile full --cases 'd-5,d-20,d-50' --runs 10 --seed 93800 --methods ADP-Grassmann,ADP-Grassmann-constant,MAVE-package --grassman-steps 5 --threads 1 --keep-inputs --output benchmark_outputs/r_sdr/grassman-constant-dev-2026-10-18
