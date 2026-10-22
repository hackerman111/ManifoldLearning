# Frozen quadratic angle selection/validation

Before any97800 data evaluation: choose quadratic.py, quadratic_trials3,
max_steps10, smooth_max_steps5, tol1e-6, local_ridge0. The exploratory90800
full-smooth10 alternative worsened d50 EDR and is not selected. No further
tuning on development results. Defaults and estimator remain unchanged.

Same paired R-SDR settings as ARMIJO_PROTOCOL.md: d-5,d-20,d-50,n500,m2,
ten datasets each; original ADP-Grassmann5, explicit ADP-Grassmann-quadratic,
real R MAVE-package=MEANMAVE; identical X/y/truth and method seeds, one thread,
300s/4096MiB process limits, immutable source/environment/input manifests.
Development97800..97809, untouched validation98800..98809 only after pass.

Gates unchanged: complete30paired successes/no failures, EDR geometric
candidate/base ratio<=.95, each case median paired delta<=.005; paired
median/p90 model.fit ratios<=1.10/1.25; median RSS<=1.10; finite bases and
orthogonality<=1e-10. Report individual outliers, bootstrap uncertainty,
outer/inner stop reasons and certificates separately. Failed development
closes this candidate; validation stays untouched. No global recovery claim.

Independent reference tests reconstruct every quadratic proposal with
explicit U@Y and per-center augmented lstsq, check chosen minimum among
actually accepted proposals and live final profile, gauge/orthogonality and
rank guards. Default zero-trial mode matches old frozen Armijo source
exactly on two actual frozen inputs. Mathematical limits: QUADRATIC_MATH.md.

    UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.r_sdr --profile full --cases d-5,d-20,d-50 --runs 10 --seed 97800 --methods ADP-Grassmann,ADP-Grassmann-quadratic,MAVE-package --grassman-steps 5 --threads 1 --keep-inputs --output benchmark_outputs/r_sdr/grassman-quadratic-dev-2026-10-18
