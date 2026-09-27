# Проверки 2026-09-27

Это краткая запись фактически выполненных проверок, не полный raw test log.

- `uv run --no-sync python -m pytest tests/test_beta_functional_audit.py -q`:
  8 passed. Проверяются joint Hessian/gradient, rank-deficient augmented QR,
  исходный local cutoff, dense ridge, common-metric spectral solution,
  gradient averaging counterexample, strided kernels и mass rounding.
- Совместно с `tests/test_hybrid.py`, `tests/test_hybrid_optimization.py`,
  `tests/test_hybrid_recycling.py`, `tests/test_lsmr.py`: 70 passed, 1 failed.
  Единственный failure: `test_manifold_hybrid_fit_matches_cg`, ValueError
  `local_quadratic estimator requires index_dim=1` при ADP_Manifold(2).
  Причина проверена в live ADP_Manifold.py:70 и ранее отражена в solver retry
  completion; production/manifold код в этой задаче не менялся.
- Target Ruff для двух новых Python-файлов: All checks passed.
- Target Pyright для двух новых Python-файлов: 0 errors, 0 warnings.
- `ruff format --check` двух новых файлов: 2 files already formatted.
- `git diff --check`: passed.
- Final audit: 6 frozen SHA-256 checked, 2 synthetic rows, 9 paired repeats;
  все шесть trajectories совпали по index/coefficients, accepted/rejected
  counts и linear iterations в проверенном bounded 3-step solve.
- Final source hashes проверены после прогона и совпадают с текущими
  LSMR.py, HYBRID.py и benchmarks/beta_functional_audit.py.
- Pilot строгий local-score comparison на больших mass обнаружил numerical
  difference; reference stress теперь фиксирует scale-bound и явное
  threshold crossing. Production certificate/tolerance не изменён.

Полные timings, profiles, allocations, solver diagnostics и environment:
`audit_final.json`. `audit.json` — отдельный pilot с предыдущим audit-script
hash. Никаких full-fit recovery/held-out/GPU claims.
