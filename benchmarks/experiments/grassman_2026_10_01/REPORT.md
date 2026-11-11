# Grassmann solver benchmark (2026-10-01)

## Scope and protocol

The heavy Spokoiny point is `mi-spokoini-m2-dhigh`: `n=800, d=50, m=2, N_J=800, N_phi=10, N_lin=100, N_loc=10, a=exp(1/100), h_min=1`, local initialization, `select_step=last`, no outer-iteration cap. Data use `X=2*Beta(1,1)-1` features (`sigma_x=1`), Gaussian noise (`sigma_eps=0.1`), and the `spokoini_m2` link. All other point defaults are `rho_corr=0`, homoscedastic noise, no outliers or misspecification, `link_scale=1`. Seed bundles come from `_make_seed_bundle("mi-spokoini-m2-dhigh", point, seed)` with its `paired-within-experiment-v1` hash design; config seed is the bundle's `init` stream (438565340, 1686317150, 2169757687 for seeds 0–2).

The effective ADP configuration is `N_loc=10, N_lin=100, N_J=800, N_phi=10, outer_steps=None, lambda_penalty=0.05, local_ridge=1e-8, a=exp(1/100), h_min=1, batch_size=32, index_init=local, estimator=new, direction_mode=auto, multi_tensor=orthogonal, select_step=last, center_displacement=0, training_set=all, redraw_directions=true`, with CPU execution and `smart_weights=false`. The kernel is Epanechnikov `K(u)=max(1-u^2,0)` applied to the normalized anisotropic **squared-distance** argument `u=(alpha^2*residual2+principal2)/h^2`. `direction_mode=auto` resolves to isotropic for this normalized multi-index fit: Gaussian directions are normalized to unit length and redrawn each outer iteration. Centers are selected once per fit, uniformly without replacement from the 800 observations using the seeded center stream; zero displacement makes each center an observation. Grassmann uses `local_ridge=0`; `lambda_prox=0.05` is its GN damping parameter. The full-fit harness uses the existing SVD baseline (`rank=1`, `direct_max_dimension=128`) and Grassmann `core_gn`. Arrays are float64; each method/seed gets a fresh process, BLAS thread environment is pinned to one, full-fit timing is `perf_counter` around `.fit`, RSS is process `ru_maxrss`, and tracing is disabled. Seeds 0, 1, and 2 are paired; order alternates in the 5-step series. Exact commands and raw rows are under `experiments/grassman_2026_10_01/benchmark/`.

The primary endpoint is recovery projector distance to the known Spokoiny truth. Grassmann solver updates optimize the profiled local-refit objective. The current SVD implementation uses a different fixed-coefficient objective; its output is re-evaluated under the common local-refit profile on captured/frozen comparisons. Thus SVD profile loss is a common diagnostic, not its training objective. All full fits stopped at the configured outer `h_min` schedule after 162 outer iterations.

Environment: Python 3.14.7 (GCC 16.1.1), Linux x86_64, NumPy 2.5.2, SciPy 1.18.1; `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`. The worktree was dirty at commit `b7fe68aa7abb10092e1a349ed0b134bdae469765`. Final source hashes: benchmark `77e1beec7e94496c0690df4737b4ea387423695cf79dcb87db55e284bf583647`, Grassmann solver `070d0bb91962f23610b3e0a647e8f10fb417bac75e049305c3ef38223966c1a2`, SVD `030d42fd6a5e7da26b586dfc15f35aceacbbb0c3aa69d4614b7b1272d183814a`. The full-fit run protocol JSONs retain the hashes present at launch; root later changed the Grassmann default `max_steps` from 20 to 5, but every benchmark call passed its step budget explicitly, so the measured runs are unchanged. The later curve-only protocol records the final harness hash.

## Paired full fits

| Grassmann inner-step cap | Median runtime ratio vs SVD | Median projector-distance change vs SVD | Median RSS change | Grassmann final Riemannian gradient (seeds 0/1/2) |
|---|---:|---:|---:|---|
| core GN, 20 | 1.260x (slower) | -0.06385 | +14,064 KiB | 4.84e-4 / 1.40e-3 / 2.00e-3 |
| core GN, 5 | 0.874x (12.6% faster) | -0.06414 | +13,332 KiB | 1.88e-2 / 2.82e-2 / 3.35e-2 |

Per-seed values are preserved in `experiments/grassman_2026_10_01/benchmark/pilot_after_adapterfix/fullfit_runs.jsonl` and `experiments/grassman_2026_10_01/benchmark/fullfit_core5/fullfit_runs.jsonl`. At 20 steps, Grassmann runtime was 43.93/46.35/49.69 s vs SVD 34.86/40.12/37.78 s; projector distances were 0.05446/0.04821/0.04531 vs 0.11831/0.08987/0.12586. At 5 steps, runtime was 31.72/33.48/34.03 s vs SVD 36.32/40.74/37.52 s; projector distances were 0.05417/0.04728/0.04494. Both budgets improve recovery on these three seeds; 5 steps also wins runtime here. Across all 6 Grassmann full fits, each fit used 162 outer iterations and every inner call hit its configured 5- or 20-step cap; all calls reported `converged=false` and `max_steps`. These results do not establish convergence or validate the 5-step choice on held-out seeds. The 5-step setting is a measured budget tradeoff, not a convergence claim.

Grassmann uses roughly 12–17 MiB more peak RSS than SVD for these full fits. At this point `N_phi=10 < d+1=51`, so exact QR compression of the local observation rows cannot reduce the active feature dimension; no compression speedup is expected from that theory branch.

## Frozen first-outer ablations

For each seed, capture the same first-outer `(index_init,U,I,mass)` tuple, then compare a 20-step solver call from the same starting index. This is a local solver comparison, not a full-fit recovery comparison. Median over seeds:

| Method | Median call time | Median final common profile loss | Median truth projector distance |
|---|---:|---:|---:|
| SVD (one call) | 0.0650 s | 76.417 | 0.6444 |
| rank-one, Schur profile | 0.0868 s | 65.003 | 0.3964 |
| rank-one, full refit | 0.4873 s | 65.003 | 0.3964 |
| spectral, fixed rank 1 | 0.1408 s | 65.002 | 0.3983 |
| adaptive-rank spectral | 0.2157 s | 64.9999 | 0.3964 |
| core Gauss-Newton | 0.0949 s | 65.0007 | 0.3967 |

The profile objective is monotonically reduced by all Grassmann variants. The Schur and full-refit rank-one paths reach nearly identical loss (maximum absolute difference `4.25e-8`) and nearly identical scalar truth-projector-distance values (maximum difference `1.63e-8`); this scalar agreement alone does not establish that their estimated projectors agree. Their median paired runtime ratio (refit/Schur) is `5.72x`; the ratio of median times is `5.62x`. All methods used the same 20-step cap; every Grassmann frozen call reached that cap and reported `converged=false`, `max_steps`. The SVD row is included as the baseline one-call update and is separately evaluated by the common profile; it does not receive the same objective or iteration budget. Full rows, diagnostics, and snapshots are in `experiments/grassman_2026_10_01/benchmark/frozen_ablation/frozen_runs.jsonl` and `experiments/grassman_2026_10_01/benchmark/frozen_ablation/frozen/`.

For improvement I, rank-one Schur is the fastest frozen variant while matching the refit reference, and the angle-curve check tests its cached profile directly. For improvement II, core GN reaches essentially the same profile loss as adaptive spectral on the frozen inputs (median `65.0007` vs `64.9999`) at less than half the call time (`0.0949` vs `0.2157` s); on full fits, 5-step core GN improves the observed recovery and runtime on these three seeds. Fixed-rank spectral also reaches similar frozen loss but is slower than rank-one Schur. These results support the measured variants on this point; they do not establish a general rank-adaptivity advantage or general superiority over SVD.

## Cached angle-profile check

For 61 angles over `[0,0.5]`, seven warmed paired repeats per seed compared cached Schur-profile evaluation (including angle-profile setup) to independent full local refits on the same captured heavy-point statistics. These cached-curve times exclude the common `U@Y` and `U@v` operator projections; their separately measured median cost is `1.263 ms`. Adding that once to each 61-angle batch gives median Schur/refit totals `3.038/70.423 ms`; the median paired ratio is `0.0431` (23.2x faster), and the ratio of median totals is `0.0431`. Maximum absolute curve error remains `3.41e-13`, maximum scaled error `2.26e-15`. This supports numerical equivalence on these inputs. The original cached-only rows remain in `experiments/grassman_2026_10_01/benchmark/frozen_ablation/curves_runs.jsonl`; the added shared-projection timing is in `experiments/grassman_2026_10_01/benchmark/curves_with_projection/curves_runs.jsonl`.

## SVD timing audit

The earlier full-fit report (`experiments/lsmr_vs_svd_fullfit_2026-09-29/`) measured seed-0 SVD at 25.7337 s; the current paired run measured 34.8588 s. The projector result agrees to about 1e-15 and both took 162 outer iterations, with matching workload diagnostics reported by the lead (`1977` inner/direct solves and `103614` U passes). NumPy/SciPy versions and platform text match. The recorded Python changed from 3.13.12/Clang 21.1.4 to 3.14.7/GCC 16.1.1. The old record has no usable BLAS/LAPACK metadata; the current system reports distro CBLAS/LAPACK 3.12.0. The recorded execution environment differs; host and BLAS equivalence cannot be established from the old metadata. Available records do not isolate the cause of the timing difference. Treat the contemporary paired times above as the relevant comparison.

## Reproduction

From the repository root (existing completed rows are skipped; use fresh output directories to measure again):

```bash
python3 -m benchmarks.grassman_benchmark --stage fullfit --grassman-steps 20 --seeds 0 1 2 --timeout 240 --output experiments/grassman_2026_10_01/benchmark/pilot_after_adapterfix
python3 -m benchmarks.grassman_benchmark --stage fullfit --grassman-steps 5 --seeds 0 1 2 --timeout 240 --output experiments/grassman_2026_10_01/benchmark/fullfit_core5
python3 -m benchmarks.grassman_benchmark --stage capture --seeds 0 1 2 --timeout 240 --output experiments/grassman_2026_10_01/benchmark/frozen_ablation
python3 -m benchmarks.grassman_benchmark --stage frozen --grassman-steps 20 --seeds 0 1 2 --timeout 240 --output experiments/grassman_2026_10_01/benchmark/frozen_ablation
python3 -m benchmarks.grassman_benchmark --stage curves --seeds 0 1 2 --timeout 240 --output experiments/grassman_2026_10_01/benchmark/frozen_ablation
python3 -m benchmarks.grassman_benchmark --stage capture --seeds 0 1 2 --timeout 240 --output experiments/grassman_2026_10_01/benchmark/curves_with_projection
python3 -m benchmarks.grassman_benchmark --stage curves --seeds 0 1 2 --timeout 240 --output experiments/grassman_2026_10_01/benchmark/curves_with_projection
```

The initial full-fit Grassmann setup raised `TypeError: ADP_solver.__init__() got multiple values for argument 'method'`; the row remains in `experiments/grassman_2026_10_01/benchmark/fullfit_runs.jsonl`. The harness was corrected and the valid core-GN series is under `experiments/grassman_2026_10_01/benchmark/pilot_after_adapterfix/`. The initial frozen SVD adapter omitted required keyword-only `rank` and produced three harness errors; these are preserved in `experiments/grassman_2026_10_01/benchmark/frozen_ablation/frozen_harness_errors.jsonl`. The adapter was fixed, and all three SVD rows were rerun successfully. No numerical solver failure occurred in either retry. The valid frozen results contain 18 successful method/seed rows and no failed runs.
