# Grassmann promotion, 2026-10-03

The verified optimized CPU solver is now ADP/solver/grassman.py. Default
core_gn, public original settings, HPAOResult, profile objective, rank guards,
workspace fallback and capped convergence semantics are retained. Flat-specific
settings and kernels are excluded from this production module.

ADP/solver/grassman_exp.py reuses promoted numerical kernels and exposes:
- solve(method="flat_gn", flat_max_steps=8), the experimental default;
- solve_full_gn(..., lsmr_maxiter=30), full horizontal matrix-free GN;
- existing baseline methods for paired experimental comparisons.

Both experimental optimizers failed their respective time/quality selection;
this relocation does not promote them or reopen heldout experiments.
ADP/solver/grassman_optim.py is removed. Import core solve from grassman,
experimental flat solve / solve_full_gn from grassman_exp.

The pre-optimization SVD reference is preserved under
experiments/grassman_optim_2026_10_01/reference.py. Optimized/reference benchmark
labels retain their historical names, while module/path mappings point to the
promoted core and archived reference. Historical raw runs/configs/hashes are
unchanged. The full_gn experiment path reexports the package implementation.
The flat harness now hashes both experimental code and shared core kernels.

Selection evidence remains in grassman_optim_2026_10_01/{REPORT,LEAD_AUDIT}.md:
frozen speedup2.52x; fullfit median runtime ratio0.941939 over three Spokoiny
seeds, close final projectors. These are previous measured results, not new
performance runs or a universal fastest-solver claim.

Relocation verification: relocation_parity.json records 120 same-input pairs:
d20/100/300/600, m1/2/4, ridge0/0.2, J8,p7, max_steps3, deterministic seed
88000+d+m. core_gn/rank_one/spectral/flat_gn/full_gn outputs (basis and local
coefficients) and diagnostics match pre-move implementations exactly. The
pre-move source hashes are recorded; temporary module copies were used for
this check. No new statistical recovery/performance claims follow.

Final checks:
- 51 focused tests pass: python -m pytest tests/test_grassman.py
  tests/test_grassman_optim.py tests/test_grassman_flat.py
  tests/test_glm_full_gn.py -q (BLAS threads pinned to one).
- Ruff check and format --check pass on 12 changed Python files.
- Targeted Pyright: 0 errors, 0 warnings for both package solver modules.
- Benchmark module imports, original/optimized routing, combined flat hash,
  and historical full-GN entry point verified. No expensive benchmark reruns.

All unrelated pre-existing dirty/untracked user work is preserved.

## Core-only API correction

User clarified the production module must expose only core_gn. Removed method
and angle_backend selectors and the refit ablation from grassman.solve; all
selectable rank_one/spectral/Schur-refit variants are in grassman_exp. Internal
rank-one/spectral rank/workspace/no-descent guards remain part of core_gn.
Test/benchmark callers migrated. The earlier public-settings retention above
is superseded for these two selectors.

51 focused tests pass. core_only_parity.json contains72 exact same-input core
comparisons: d20/100/300/600,m1/2/4,ridge0/0.2, regular/rank/workspace cases,
J8,p7,max_steps3, seed89000+d+m. Basis, coefficients and diagnostics exactly
match the pre-restriction solver. Benchmark routing/construction checks pass;
Ruff/format and targeted Pyright pass. No new performance runs.
