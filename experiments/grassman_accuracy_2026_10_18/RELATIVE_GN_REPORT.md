# Grassmann relative GN: accepted explicit option

Historical accepted10 evidence. Since the user's later integration request,
public defaults are combined20; see COMBINED_REPORT.md. The explicit settings
below still reproduce this earlier option.

## Use

```python
from ADP.core.ADP_Solver import ADP_solver
from ADP.solver.grassman import solve

solver = ADP_solver(solve, relative_gn=True, max_steps=10,
                    quadratic_trials=0, constant_centers=False, smooth_max_steps=5)
```

Use the same explicit settings for direct arrays. The measured configuration uses
lambda_prox=.05,tol1e-6,local_ridge=0,initial-smooth cap5,one BLAS thread.
ADP_solver receives damping from ADP_Config.lambda_penalty; do not pass
lambda_prox/smooth_max_steps as adapter settings. R method: ADP-Grassmann-relative.

## Mechanism and correctness

APPROXIMATE optimizer of the same all-center unpenalized finite-sketch profile.
At eligible numerical intrinsic rank loss, use regular-center GN and
mu=lambda_prox*||J||F^2/(q*m). Positive response/mass scaling uniformly
rescales the model, preserving the conditional step. Math/finite-precision
limits: RELATIVE_GN_MATH.md and INTRINSIC_RANK.md. Full-gradient slope, all-center
true Armijo, original rank thresholds, live coefficients/gradient/certificates
remain mandatory. Ineligible model -> direct rank-one Armijo; zero/nonfinite
damping or workspace failure -> spectral fallback. Smooth GN is unchanged.
No dense d*d array; rank SVD input scratch4MiB per chunk or one larger center.

## Frozen independent evidence

Each stage:30 paired datasets,90 successful fits, n500,m2,d5/d20/d50,
original5/relative10/actual R MAVE::MEANMAVE. All8 prespecified gates pass.

| Stage | EDR geometric ratio [95% interval] | Paired time median/p90 | RSS ratio | Max individual EDR ratio |
|---|---:|---:|---:|---:|
| Development105800 | 0.898376 [0.844245,0.948903] | 0.650580/0.790378 | 1.005236 | 1.077314 |
| Untouched validation106800 | 0.923729 [0.886622,0.956495] | 0.653572/0.993297 | 1.002106 | 1.036418 |
| Public confirmation106800 | 0.923729 [0.886622,0.956495] | 0.649100/0.944153 | 1.001704 | 1.036418 |

Public confirmation repeats the consumed validation data: no extra independent
samples. Development and untouched validation used identical source sets.
The current public port reproduced all120 saved full fits exactly (60default,
60relative): basis entries, every inner diagnostic and outer stop match.
Artifact: relative_gn_public_replay.json. Current public timing is measured
separately in the last row; no CPU-heavy checks ran concurrently.

## Current public code versus R

Case medians, EDR / model.fit seconds; lower is better. R is actual installed
MAVE-package with MEANMAVE, not the separate ordinary R MAVE implementation.

| Case | Original Grassmann5 | Relative GN10 | R MEANMAVE |
|---|---:|---:|---:|
| d-5 | 0.049694 / 0.3181 | 0.049903 / 0.1850 | 0.031150 / 0.1510 |
| d-20 | 0.110213 / 0.6181 | 0.100482 / 0.3907 | 0.075923 / 1.0805 |
| d-50 | 0.200713 / 0.7337 | 0.171312 / 0.6692 | 0.141289 / 4.5245 |

Quality benefit is concentrated at d20/d50; d5 is nearly unchanged.
R remains more accurate in these case medians. The overall observed EDR
reduction is7.63%, paired median time reduction35.09% in public confirmation.
The validation95% interval includes gains smaller than5%; the fixed gate uses
the point estimate. All ADP outer stops are h_min, zero certified inner calls:
this establishes bounded-budget recovery/time improvement, not convergence
or a universal guarantee for other n,m,distributions or parameters.

## Verification and artifacts

209 focused numerical/solver/R-SDR tests pass, including actual worker/model.fit
and R; Ruff/format/Pyright clean. Dense finite-difference augmented-LS references,
m1/2/3 scaling, invalid damping/workspace/rank/default/port checks are included.
Public-vs-prototype small cases and120 full-fit replays have zero output error.
Current grassman.py SHA256: `8804e23a16c4cf2de65d9c7a0fddc20b03fe6156d19bc3f8888bff2f607d782a`.
Golden baseline SHA256:6d5aa753da9d9c8532545fa7519e63ba479c0f4082a8a6d52358cd4c66af72a0.

Raw run folders (each retains sources, inputs, environment/config fingerprints,
journals, failures and analysis_summary.json):
- benchmark_outputs/r_sdr/grassman-relative-dev-repaired-2026-10-22/
- benchmark_outputs/r_sdr/grassman-relative-heldout-2026-10-22/
- benchmark_outputs/r_sdr/grassman-relative-public-confirm-2026-10-22/

Initial invalid development launch retained separately as grassman-relative-dev-2026-10-22:
63rows/21candidate errors before optimization, duplicate lambda_prox forwarding.
Repair changed only runner forwarding; the candidate algorithm/budgets/gates
stayed frozen. This invalid launch is not pooled into successful evidence.
Luna audited the algebra and prior routine work, then hit its usage limit;
remaining tests/analysis used deterministic scripts under the lead agent.
