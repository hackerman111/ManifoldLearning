# Manifold recovery: M1 baseline

Frozen protocol: `PLAN.md`. Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/adp-uv-cache \
  uv run --no-sync python -m experiments.manifold_recovery_probe \
  --seeds 0:10 \
  --out docs/experiments/manifold_recovery_2026-09-23/baseline_selection
```

`baseline_selection/{manifest.json,runs.json,centers.csv}` records the code hash,
configuration, RNG streams, every center and every phase. Seed 0 reproduces
the previous `runs.csv` quality `0.48095653044126213` exactly. The metric
recomputed from center sines equals `experiments.runner._local_subspace_metrics`
to `1e-12` on successful fits. No baseline estimator code was changed.

| Symptom | Phase | Evidence on selection seeds 0–9 |
|---|---|---|
| Poor local quality precedes ADP updates | Pilot / initialization | Pilot-gradient RMS sine across 240 centers is 0.266; graph-SVD initialization is 0.479. The latter already exceeds the 0.2 full-center threshold on all ten seeds (per-seed 0.405–0.553). |
| Near-origin radial direction is hardest | Pilot / initialization | Radius quartiles: pilot RMS 0.431, 0.229, 0.164, 0.135; initialized RMS 0.658, 0.551, 0.375, 0.202, from smallest to largest radii. All centers remain in the primary metric. |
| One-factor update cannot rescue initial bias | Sync / scale | Four successful fits end at 0.481, 0.437, 0.510, 0.570. Their initial errors are 0.471, 0.427, 0.553, 0.507. No successful fit reaches 0.2. |
| Global average mass hides empty local statistics | Scale / slope | Function mass minimum and effective sample size minimum are both 1; their 10th percentiles are 4.42 and 5.79 while median values are 18.0 and 25.5. Six fits fail with rank-0 slope, usually after a source center has only its self observation. |
| Linear solver is not the observed failure | Successful B solves | Certified relative residuals in recorded traces are below 1e-6; the exceptions originate in slope rank checks before the B solve. |

The local-linear design has median condition 1.33 (maximum 8.40) where it
passes rank checks. The manifold graph has 1–21 neighbors per target (median
10); its mass target of six holds only *on average*. These observations support
testing pilot and localization bias and a per-center mass rule. They do not
establish a mathematical code error.

`convergence_pass` in the previous experiment runner is currently the result
of a successful `fit`, because the CLI writes `converged=True` unconditionally.
The manifold procedure has a scheduled terminal scale, not an outer fixed-point
certificate. In this study, `completion_pass` means `h_min` or the explicitly
chosen feasible mass boundary was reached with certified inner solves. We do
not call that outer convergence; a `convergence_pass` claim would require an
explicit fixed-point or stationarity test. Failed fits count against recovery.
