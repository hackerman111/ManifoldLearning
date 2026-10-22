# Frozen intrinsic GN direction experiment

Before any99800 evaluation: choose intrinsic.py wrapper using
intrinsic_gn=True, max_steps10, smooth_max_steps5, quadratic_trials0,
tol1e-6, local_ridge0. At rank guard, a numerical intrinsic-rank classifier
can propose GN on only regular centers; ALL centers remain in gradient,
true-profile trial acceptance and final live certification. Other cases
use direct value-based Armijo, smooth/workspace/GN guards remain original.
Rank-boundary certificates remain inapplicable. This is an APPROXIMATE
direction, not a new estimator or proof of exact numerical rank.

The90800-only5/10pilot preceded freeze:10 improves allthree case EDR over5,
with time ratios.50/.58/1.04 vs baseline5. No dimension-specific tuning.
Proof, numerical-cutoff caveats and data diagnosis: INTRINSIC_RANK.md.
Independent tests compare exact constant-image residual/gradient with the
regular subset, full live profile/coefficients/ranks/orthogonality, accidental
projected-rank fallback, sensitive regular guard, ridge bypass and4MiBchunks.

Same R-SDR paired protocol: d5,d20,d50,n500,m2, ten seeds each; original
ADP-Grassmann5, explicit ADP-Grassmann-intrinsic, real R MEANMAVE under
MAVE-package; same X/y/truth/method seeds, one thread,300s/4096MiB limits,
immutable source/environment/input manifests and logs.
Development99800..99809, heldout100800..100809 only after pass.

Gates unchanged: complete30pairs/no failures; EDR geometric ratio<=.95,
case median delta<=.005; paired median/p90fit ratios<=1.10/1.25; median
RSS<=1.10; finite bases/orthogonality<=1e-10. Report paired uncertainty,
individual regressions, actual R medians, stops and certificates separately.
Failed development closes candidate and keeps heldout untouched. No tuning
on development or validation failures. Production defaults unchanged.

    UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.r_sdr --profile full --cases d-5,d-20,d-50 --runs 10 --seed 99800 --methods ADP-Grassmann,ADP-Grassmann-intrinsic,MAVE-package --grassman-steps 5 --threads 1 --keep-inputs --output benchmark_outputs/r_sdr/grassman-intrinsic-dev-2026-10-18
