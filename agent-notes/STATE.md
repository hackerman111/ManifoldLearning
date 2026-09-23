# Current agent state

## Task

- ID: repair-manifold-recovery-2026-09-23.
- Status: done; M1–M5 acceptance evidence is linked from `PLAN.md`.
- Goal: repair manifold recovery to full-center RMS sine `<=0.2` on the radial base scenario, then validate at least 16/20 independent seed without numerical errors.

## Established facts

- M1 artifact: `docs/experiments/manifold_recovery_2026-09-23/diagnostics.md` and `baseline_selection/{manifest.json,runs.json,centers.csv}`. Reproduce with `python -m experiments.manifold_recovery_probe --seeds 0:10 --out ...` using one BLAS thread and `UV_CACHE_DIR=/tmp/adp-uv-cache`.
- Baseline matches old seed-0 quality exactly (`0.48095653044126213`). Selection 0–9 gives 0/10 recovered, six rank-0 slope failures. No algorithm code changed in M1.
- Across 240 selected centers, pilot-gradient RMS sine is 0.266, graph-SVD initialization is 0.479; all initial per-seed errors exceed 0.2. Radius-near-zero centers are worst. Successful ADP updates do not fix the global bias.
- Global mean observation mass 20 allows center mass and `n_eff` as low as one; rank failures arise before B solve. Successful linear solves have certified small residuals. M3 confirmed that a per-center mass floor removes these failures, while narrow graph locality and a quadratic moment correction are jointly needed for quality on the selected radial design.
- `ADP/cli/main.py` still sets manifold `converged=True` after successful `fit`. The current scale schedule has no outer stationarity certificate. For this research, distinguish scheduled completion with certified inner solves from true convergence.
- `tex/manifold-ade.tex:214-225` includes `W_l` in the penalty; `:334-353` omits it. Current code and `tests/test_manifold.py:397-420` choose the latter normalized penalty. This is an unresolved estimator interpretation.
- M2 artifact: `docs/experiments/manifold_recovery_2026-09-23/reference.md` and `oracle_*_selection/`. Direct moments, objective gradient, augmented LS minimizer, adjoint and projector reference pass all 13 focused tests. No confirmed EXACT/NUMERICAL defect in those components.
- With true initial projectors and true directional moments, the first graph/objective update raises aggregate RMS sine from 0 to 0.223; final median among successful fits is 0.243 and zero recoveries. Broad neighborhood bias and low per-center observation mass survive the oracle probes. Manuscript does not specify an explicit `h_M` recurrence; varying it is an ESTIMATOR experiment.
- M3 selection artifact: `docs/experiments/manifold_recovery_2026-09-23/selection.md` and per-variant directories. Explicit local quadratic moment correction + self-only graph + per-center mass floor six recovers 10/10 selection with zero errors (RMS 0.034–0.076). Component variants fail or leave rank errors. This is an ESTIMATOR change specialized to `m=1`, not a confirmed defect in the existing normal operator.
- Production option `ADP_Manifold(estimator="local_quadratic")` is implemented in `ADP/core/manifold/ADP_Manifold.py` and `ADP/engine/manifol_engine/{fit.py,weights.py}`; default remains `estimator="manifold"` because the new estimator assumes a locally quadratic scalar response and removes neighbor pooling. `production_selection/` matches the probe per seed within `1e-12` and reproduces the baseline. Updated route: `agent-notes/ADP/manifold.md`.
- M4 held-out artifact: `docs/experiments/manifold_recovery_2026-09-23/validation.md` and `production_validation/`. Candidate 20/20 recovered on seed 100–119, zero errors, worst RMS 0.1485, all `h_min`, maximum inner residual 5.42e-7. Baseline 0/20 and ten errors. A different noiseless varying geometry gives 5/5 candidate recovery; CG/hybrid selection quality differs by ≤1.34e-9.
- M5 packed the quadratic coefficients into `(J,d(d+1)/2)` without changing the estimator; a second run on the same held-out inputs confirmed max quality drift 1.10e-15 and identical outcomes. Cost and checks: `docs/experiments/manifold_recovery_2026-09-23/completion.md`. Fixed-shape median wall time baseline 0.0708 s, candidate 0.0981 s; traced peaks about 0.646 MiB. Final suite 309 passed, 28 CUDA skips; Ruff on affected trees, format, `uv lock --check`, `git diff --check` pass. Pyright 0 errors, one existing legacy warning. Repository-wide Ruff has ten pre-existing issues only in untouched `test/`.

## Next action

No required work remains in this plan. If broader manifold recovery is requested later, design a new study for non-quadratic links and `m>1` before changing the default.
