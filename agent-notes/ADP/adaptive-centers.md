# Experimental adaptive centers P0-P3

Explicit CPU multi-index/new-estimator options, disabled by default. Built-in
Epanechnikov only; no solver/default change. Formula audit and numerical
boundaries: [MATH.md](../../experiments/adaptive_centers/MATH.md).
Frozen experiments, raw data and interpretation:
[PROTOCOL.md](../../experiments/adaptive_centers/PROTOCOL.md),
[REPORT.md](../../experiments/adaptive_centers/REPORT.md).

```python
from ADP import AdaptiveCentersConfig, ADP_Config, ADP_multi_index

experiment = AdaptiveCentersConfig(
    lambda_floor=0.25,       # P0: ESTIMATOR, floor only localization
    projection_search=True, # P1: exact projected candidates and full-pool alpha
)
model = ADP_multi_index(2, ADP_Config(experimental_centers=experiment)).fit(X, y)
```

P2 standalone overhead/diagnostics: `projected_statistics=True`, selection all.
P2 prefilter: `selection="score", budget=...`; equal-J control `"random"`.
P3: `selection="logdet"` or `"logdet_coverage"`, fixed budget or `budget=None`
for dynamic 96.5% information/95% feasible-active-mass threshold. Fixed budgets
must be in `[min(J0,max(4m,2d)), J0]`. With insufficient feasible centers,
the complete pool is used and the fallback is exposed. Feasibility is a
heuristic, not an EDR identifiability guarantee.

Optional `quadrature=True` uses unnormalized PoU masses. It conserves covered
scalar mass, not the full operator. It is a separate ESTIMATOR ablation.
`score_ridge=1e-6` regularizes only the projected surrogate; solver ridge is
independent. `logdet_ridge=1`, `coverage_weight=1` are explicit defaults.

Raw canonical eigenvalues and full-pool direction RNG remain unchanged.
Weights, alpha and optional solver metric context use the floored spectrum.
Initialization and alpha use the complete pool. Error for best/last step uses
the same full-pool local predictions for all variants. Full selected-center
I/U use the shared statistics implementation and exact current weights.
Direction generation still allocates the full pool sketch. Initialization
still stores D=(J0,n); additional projected candidate geometry stores O(E0).

Trace `experimental_centers` contains selected_pool_indices, selected_centers,
mass_coverage, active_mass_coverage (greedy), fallback, support/candidate edges,
effective localization spectrum, quadrature and separate weight/projected/
selection/assembly times. Update-stage geometry construction is in profile
`update`, not the diagnostic weight timer. Returned selected coefficients map
to original observation rows via `effective_parameters_["coefficient_center_indices"]`.
GPU, single-index, legacy and custom kernels are explicitly unsupported.

P1 preserves same-state support/weights and alpha; full adaptive trajectories
can diverge after reduction-order roundoff enters the capped iterative solver.
The replay in experiments/adaptive_centers/verification/trajectory_audit.json
reproduces one final EDR difference of -.01036 despite I/U differences <1.4e-15.
Diagnostic eta is unstable on singleton supports and is unused by the solver.

## Measured outcome (2026-10-17)

Luna:120 paired development fits on4 new cases,n500,m2; all completed at h_min,
no converged inner certificates. No predeclared global quality/speed gate
passed; selection=null, heldout92900..4 untouched. P0 floor25 median paired
EDRdiff0,time.994baseline; case-dependent additive30 median .4190->.2486.
P1/floor25 time1.682; P2/P1 time1.179 without quality change. P3 LogDet+coverage
EDRdiff+.03968,time2.487baseline, but -.03773 EDR vs equal-J random at1.338time.
No process RAM saving in these full fits. J shrinks500->60/80/100.
Separate24-worker localization benchmark:n10000,d100,J500,h.4,cache D common,
P1/baseline time.493(weights),.414(alpha+weights), exact emitted weights;
n500 slower. CSR-output timing/memory only, not end-to-end or native streamed
baseline RAM. See REPORT and supplement/REPORT; do not promote defaults.

Final27 feature tests/86 shared tests pass; unrelated collection/manifold
failures remain. After timings only a seed=None compatibility patch changed
selection RNG to existing root entropy; frozen integer streams identical.
Original snapshots immutable, explicit overlay at experiments/adaptive_centers/
verification/compatibility_overlay.json; lead_audit.py verifies it. Original
harness freeze checks intentionally reject the patched tree.

## Source locators

| Source | Locator |
|---|---|
| opt-in API/validation | `ADP/core/experimental_centers.py:10-63` |
| config hook | `ADP/core/ADP_Config.py:51-119` |
| exact projected geometry/alpha | `ADP/engine/common/experimental_geometry.py:16-103` |
| projected moments/score vectors | `ADP/engine/common/experimental_centers.py:15-90` |
| coverage/PoU/lazy LogDet | `ADP/engine/common/experimental_centers.py:93-189` |
| scope/preselection/statistics | `ADP/engine/common/experimental_step.py:25-168` |
| shared integration | `ADP/engine/common/index_fit.py:318-647` |
| CSR/dense statistics dispatch | `ADP/engine/common/statistic.py:30-124` |
| common row-centered moment formulas | `ADP/engine/common/statistic.py:191-237` |
| independent direct/reference checks | `tests/test_adaptive_centers.py` |
