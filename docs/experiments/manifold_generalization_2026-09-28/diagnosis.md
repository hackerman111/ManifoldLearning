# Why the manifold generalization run fails

This diagnosis uses the frozen `main_stop/` rows and the read-only paired
counterfactual in [diagnosis.py](diagnosis.py). It addresses geometric recovery,
independently of linear solver status. Production ADP code and the frozen
`main/` and `main_stop/` outputs were not changed.

## Flat control: the third direction is lost during scale refinement

For broad support, `m=3,c=0`, all 10 completed `main_stop/` fits fail the
center/query threshold although the oracle query error is zero. Across the 10
frozen rows, initial center RMS has median 0.172 and final center RMS has
median 0.582. Hence neither nearest-chart discretization nor fit exceptions
explain this control.

The production kernel uses

`distance² = sum_k lambda_k (p_k · delta)² + alpha² ||delta_perp||²`,

with `lambda_k` equal to the recovered singular spectrum divided by its
largest value (`ADP/engine/manifol_engine/{weights,optimisation}.py`). Weak
signal directions therefore get a wider kernel. The next statistics are
computed with that kernel and generate the next spectrum. In seed 81000,
the median third `lambda` after the three sync steps is 0.425; subsequent
scale steps give 0.119, 0.030, 0.012, 0.005, while center RMS grows from
0.159 to 0.421 by scale 4 and 0.608 at the final scale. This is positive
feedback between spectral weighting and localization. `alpha` controls only
the perpendicular term and cannot restore localization along the weak row
direction.

An isolated `UnitySpectrum` subclass replaces returned relative eigenvalues
with ones, leaving projector updates, data, random streams, solver and
thresholds intact. In the 10 paired flat `m=3` runs, baseline recovery is
0/10; `UnitySpectrum` recovery is 10/10. Its maximum center sine is at most
0.182 (baseline minimum over these rows is 0.540). For seed 81000, after
scale 2 both variants have `h=2.92116`; baseline center RMS is 0.243 and
`UnitySpectrum` is 0.068. The variants then stop at different scales (5–6
versus 3), as expected because changing the kernel changes mass feasibility.
This counterfactual identifies the feedback mechanism; it is an **ESTIMATOR**
change, not a validated replacement for the production rule.

## Curved controls: wide graph mixes distinct tangent spaces

The broad initialization graph targets mean mass 30 among only 40 centers.
Keeping the same graph and center weights but substituting exact noiseless
gradients at its source centers gives the following median center RMS over
seeds 81000–81004. The fitted-gradient column uses the production local
linear gradients; exact-gradient column removes that estimation error.

| m | curvature | fitted gradients | exact gradients |
|---:|---:|---:|---:|
| 1 | 0.35 | 0.330 | 0.268 |
| 2 | 0.35 | 0.321 | 0.271 |
| 3 | 0.35 | 0.336 | 0.279 |
| 1 | 0.8 | 0.502 | 0.588 |
| 2 | 0.8 | 0.563 | 0.612 |
| 3 | 0.8 | 0.661 | 0.649 |

Thus even exact gradients do not make this broad graph's weighted-gradient
SVD recover the local row space. It pools gradients from distinct tangent
spaces. Narrower support lowers this bias but the frozen local mode contains
rank failures and less stable local fits. The flat spectral counterfactual
does not solve this curved initialization problem.

The combined query criterion has a separate ceiling: in every completed
curved profile in `main_stop/`, using **exact** bases at the 40 centers with
the current nearest-chart rule still gives oracle query maximum sine above
0.2 (profile medians 0.663–0.995). The specified combined center-and-query
recovery is therefore unattainable for those realized query sets by this
chart representation, even with perfect center estimates. This does not
explain the measured center errors; both limitations matter.

The strict `main/` run also has 149 infeasible function-mass targets and 31
rank failures. Those are outer scale/support failures, not evidence about
solver recovery. The exploratory `stop` mode isolates geometric quality but
does not turn this diagnosis into untouched validation.

## Reproduce

From the repository root:

```bash
UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.manifold_generalization --self-check
PYTHONPATH=. UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python docs/experiments/manifold_generalization_2026-09-28/diagnosis.py
```

The first command checks the analytic truth and principal-angle metric. The
second uses one BLAS thread and the frozen five seeds. Its paired spectrum
test is diagnostic and uses `scale_boundary=stop`; no selection or held-out
claim is made. The source hashes in `main/` and `main_stop/` match the live
production `ADP/` files used here; three experiment support files have since
changed, so the paired script explicitly reconstructs the original data and
model settings and reproduces the frozen baseline metrics.
