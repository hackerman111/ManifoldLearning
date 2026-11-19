# LOO1-WLS-ADP vs LiveADP Paired Benchmark Report

## Protocol & Environment
- Branch / Commit: `feat/loo1-wls-adp` @ `2190b230198fa5668de473bece2d4c1de59692e0` (dirty=True)
- Python: `3.13.12`, NumPy: `2.5.2`, SciPy: `1.18.1`
- Platform: `Linux-7.2.9-arch1-1-x86_64-with-glibc2.44`
- Threads: `{"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}`
- `LiveADP`: `ADP_Config(N_J=NJ, h_min=hmin, N_loc=Nloc, lambda_penalty=0.05)`, `ADP_solver(grassman.solve, max_steps=20)`
- `LOO1-WLS`: `ADP_Config(N_J=NJ, h_min=hmin, N_loc=Nloc, estimator='loo1_wls', loo1_wls=LOO1WLSConfig(spectrum_exponent=beta, max_steps=25))`
- Timing: `time.perf_counter()`; RSS: `/proc/self/status` `VmHWM` and `resource.getrusage(RUSAGE_SELF).ru_maxrss`.

## Suite 1: Weak(8,2) & Mixture(8,2) (n=960, d=8, m=2, N_J=816)

| Arm | OK / Total | Rec <= 0.1 | Mean EDR | Median EDR | Max EDR | Mean fit (s) | Median fit (s) | Median VmHWM (MiB) | Max VmHWM (MiB) | Max ru_maxrss (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LiveADP | 12/12 | 2/12 | 0.2201 | 0.2543 | 0.3460 | 1.036 | 1.033 | 111.1 | 112.2 | 112.2 |
| LOO1-WLS (beta=0.5) | 12/12 | 10/12 | 0.0809 | 0.0799 | 0.1432 | 0.549 | 0.550 | 111.6 | 112.2 | 112.2 |
| LOO1-WLS (beta=2/3) | 12/12 | 10/12 | 0.0811 | 0.0766 | 0.1474 | 0.550 | 0.541 | 111.6 | 112.2 | 112.2 |

### Per-Case Breakdown

| Case | Arm | OK/Tot | Rec<=0.1 | Mean EDR | Median EDR | Median fit (s) | Max VmHWM (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| Mixture(8,2) | LiveADP | 6/6 | 1/6 | 0.2452 | 0.2619 | 1.011 | 112.2 |
| Mixture(8,2) | LOO1-WLS (beta=0.5) | 6/6 | 6/6 | 0.0718 | 0.0757 | 0.550 | 112.2 |
| Mixture(8,2) | LOO1-WLS (beta=2/3) | 6/6 | 6/6 | 0.0725 | 0.0741 | 0.526 | 112.2 |
| Weak(8,2) | LiveADP | 6/6 | 1/6 | 0.1951 | 0.1885 | 1.053 | 111.1 |
| Weak(8,2) | LOO1-WLS (beta=0.5) | 6/6 | 4/6 | 0.0900 | 0.0868 | 0.548 | 111.1 |
| Weak(8,2) | LOO1-WLS (beta=2/3) | 6/6 | 4/6 | 0.0896 | 0.0840 | 0.568 | 111.1 |

### Paired Comparison vs LiveADP

- **LOO1-WLS (beta=0.5)** vs `LiveADP` (12 pairs): wins/losses/ties = 11/1/0, mean ΔEDR = -0.1392, median ΔEDR = -0.1775, median speedup = 1.89x (mean 1.89x)
  - Worst regression: `Weak(8,2)` seed `4111803` (0.1179 -> 0.1432, Δ=+0.0252)
  - Best improvement: `Mixture(8,2)` seed `4111800` (0.3460 -> 0.0835, Δ=-0.2625)
- **LOO1-WLS (beta=2/3)** vs `LiveADP` (12 pairs): wins/losses/ties = 11/1/0, mean ΔEDR = -0.1391, median ΔEDR = -0.1803, median speedup = 1.91x (mean 1.89x)
  - Worst regression: `Weak(8,2)` seed `4111803` (0.1179 -> 0.1474, Δ=+0.0295)
  - Best improvement: `Mixture(8,2)` seed `4111800` (0.3460 -> 0.0868, Δ=-0.2592)

## Suite 2: Seven Families (n=1000, d=25, m=3, N_J=850, seeds 0..9)

| Arm | OK / Total | Rec <= 0.1 | Mean EDR | Median EDR | Max EDR | Mean fit (s) | Median fit (s) | Median VmHWM (MiB) | Max VmHWM (MiB) | Max ru_maxrss (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LiveADP | 70/70 | 23/70 | 0.3087 | 0.1225 | 0.9770 | 3.239 | 3.286 | 131.7 | 133.0 | 133.0 |
| LOO1-WLS (beta=0.5) | 70/70 | 39/70 | 0.2495 | 0.0861 | 0.9754 | 2.365 | 2.231 | 131.7 | 133.0 | 133.0 |
| LOO1-WLS (beta=2/3) | 70/70 | 40/70 | 0.2461 | 0.0864 | 0.9748 | 2.430 | 2.302 | 131.7 | 133.0 | 133.0 |

### Per-Case Breakdown

| Case | Arm | OK/Tot | Rec<=0.1 | Mean EDR | Median EDR | Median fit (s) | Max VmHWM (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| family-additive | LiveADP | 10/10 | 0/10 | 0.3617 | 0.3740 | 3.409 | 130.0 |
| family-additive | LOO1-WLS (beta=0.5) | 10/10 | 0/10 | 0.3844 | 0.3804 | 3.838 | 130.0 |
| family-additive | LOO1-WLS (beta=2/3) | 10/10 | 0/10 | 0.3619 | 0.3531 | 3.762 | 129.3 |
| family-localized | LiveADP | 10/10 | 3/10 | 0.3368 | 0.3847 | 3.384 | 133.0 |
| family-localized | LOO1-WLS (beta=0.5) | 10/10 | 9/10 | 0.0732 | 0.0705 | 2.119 | 133.0 |
| family-localized | LOO1-WLS (beta=2/3) | 10/10 | 9/10 | 0.0728 | 0.0667 | 2.047 | 133.0 |
| family-multiplicative | LiveADP | 10/10 | 8/10 | 0.0934 | 0.0913 | 2.747 | 131.7 |
| family-multiplicative | LOO1-WLS (beta=0.5) | 10/10 | 10/10 | 0.0765 | 0.0746 | 2.291 | 131.7 |
| family-multiplicative | LOO1-WLS (beta=2/3) | 10/10 | 10/10 | 0.0766 | 0.0750 | 2.368 | 131.7 |
| family-nonsmooth | LiveADP | 10/10 | 8/10 | 0.0913 | 0.0907 | 3.169 | 133.0 |
| family-nonsmooth | LOO1-WLS (beta=0.5) | 10/10 | 10/10 | 0.0775 | 0.0772 | 1.648 | 133.0 |
| family-nonsmooth | LOO1-WLS (beta=2/3) | 10/10 | 10/10 | 0.0780 | 0.0775 | 1.674 | 133.0 |
| family-oscillatory | LiveADP | 10/10 | 0/10 | 0.9439 | 0.9375 | 3.232 | 131.8 |
| family-oscillatory | LOO1-WLS (beta=0.5) | 10/10 | 0/10 | 0.9452 | 0.9477 | 2.851 | 131.8 |
| family-oscillatory | LOO1-WLS (beta=2/3) | 10/10 | 0/10 | 0.9434 | 0.9442 | 2.902 | 131.8 |
| family-quadratic | LiveADP | 10/10 | 0/10 | 0.2210 | 0.1427 | 3.357 | 131.7 |
| family-quadratic | LOO1-WLS (beta=0.5) | 10/10 | 0/10 | 0.1147 | 0.1145 | 2.078 | 131.7 |
| family-quadratic | LOO1-WLS (beta=2/3) | 10/10 | 1/10 | 0.1157 | 0.1142 | 2.282 | 131.7 |
| family-saturating | LiveADP | 10/10 | 4/10 | 0.1130 | 0.1071 | 3.305 | 132.7 |
| family-saturating | LOO1-WLS (beta=0.5) | 10/10 | 10/10 | 0.0754 | 0.0745 | 1.643 | 132.7 |
| family-saturating | LOO1-WLS (beta=2/3) | 10/10 | 10/10 | 0.0747 | 0.0719 | 1.684 | 132.7 |

### Paired Comparison vs LiveADP

- **LOO1-WLS (beta=0.5)** vs `LiveADP` (70 pairs): wins/losses/ties = 55/15/0, mean ΔEDR = -0.0592, median ΔEDR = -0.0197, median speedup = 1.46x (mean 1.49x)
  - Worst regression: `family-additive` seed `7` (0.3737 -> 0.5490, Δ=+0.1753)
  - Best improvement: `family-localized` seed `2` (0.5837 -> 0.0618, Δ=-0.5219)
- **LOO1-WLS (beta=2/3)** vs `LiveADP` (70 pairs): wins/losses/ties = 55/15/0, mean ΔEDR = -0.0626, median ΔEDR = -0.0187, median speedup = 1.43x (mean 1.44x)
  - Worst regression: `family-additive` seed `5` (0.4099 -> 0.5233, Δ=+0.1134)
  - Best improvement: `family-localized` seed `2` (0.5837 -> 0.0621, Δ=-0.5216)

## Suite 3: Seven Families Scaling (n=2000, d=50, m=3, N_J=1700, seeds 0..2)

| Arm | OK / Total | Rec <= 0.1 | Mean EDR | Median EDR | Max EDR | Mean fit (s) | Median fit (s) | Median VmHWM (MiB) | Max VmHWM (MiB) | Max ru_maxrss (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LiveADP | 21/21 | 6/21 | 0.3203 | 0.1386 | 0.9849 | 12.147 | 12.129 | 211.9 | 231.9 | 231.9 |
| LOO1-WLS (beta=0.5) | 21/21 | 10/21 | 0.2719 | 0.1099 | 0.9828 | 9.196 | 9.443 | 173.1 | 174.5 | 174.5 |

### Per-Case Breakdown

| Case | Arm | OK/Tot | Rec<=0.1 | Mean EDR | Median EDR | Median fit (s) | Max VmHWM (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| family-additive | LiveADP | 3/3 | 0/3 | 0.3475 | 0.3467 | 12.045 | 211.8 |
| family-additive | LOO1-WLS (beta=0.5) | 3/3 | 0/3 | 0.3481 | 0.3574 | 12.387 | 174.5 |
| family-localized | LiveADP | 3/3 | 0/3 | 0.5073 | 0.5831 | 12.254 | 212.5 |
| family-localized | LOO1-WLS (beta=0.5) | 3/3 | 2/3 | 0.2093 | 0.0704 | 9.526 | 173.4 |
| family-multiplicative | LiveADP | 3/3 | 1/3 | 0.1009 | 0.1034 | 11.794 | 231.9 |
| family-multiplicative | LOO1-WLS (beta=0.5) | 3/3 | 2/3 | 0.0971 | 0.0920 | 9.570 | 173.1 |
| family-nonsmooth | LiveADP | 3/3 | 3/3 | 0.0835 | 0.0834 | 12.542 | 219.1 |
| family-nonsmooth | LOO1-WLS (beta=0.5) | 3/3 | 3/3 | 0.0801 | 0.0801 | 7.142 | 173.3 |
| family-oscillatory | LiveADP | 3/3 | 0/3 | 0.9803 | 0.9826 | 11.583 | 230.6 |
| family-oscillatory | LOO1-WLS (beta=0.5) | 3/3 | 0/3 | 0.9744 | 0.9702 | 10.981 | 174.1 |
| family-quadratic | LiveADP | 3/3 | 0/3 | 0.1258 | 0.1213 | 11.896 | 217.2 |
| family-quadratic | LOO1-WLS (beta=0.5) | 3/3 | 0/3 | 0.1229 | 0.1257 | 8.038 | 173.1 |
| family-saturating | LiveADP | 3/3 | 2/3 | 0.0966 | 0.0855 | 12.216 | 224.6 |
| family-saturating | LOO1-WLS (beta=0.5) | 3/3 | 3/3 | 0.0713 | 0.0711 | 7.208 | 174.5 |

### Paired Comparison vs LiveADP

- **LOO1-WLS (beta=0.5)** vs `LiveADP` (21 pairs): wins/losses/ties = 16/5/0, mean ΔEDR = -0.0484, median ΔEDR = -0.0047, median speedup = 1.32x (mean 1.38x)
  - Worst regression: `family-additive` seed `1` (0.3296 -> 0.3574, Δ=+0.0278)
  - Best improvement: `family-localized` seed `1` (0.5868 -> 0.0600, Δ=-0.5268)

## Suite 4: Streaming Check (n=4000, d=100, m=3, N_J=3400, quadratic seed 0)

| Arm | OK / Total | Rec <= 0.1 | Mean EDR | Median EDR | Max EDR | Mean fit (s) | Median fit (s) | Median VmHWM (MiB) | Max VmHWM (MiB) | Max ru_maxrss (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LOO1-WLS (beta=0.5, stream=True) | 1/1 | 0/1 | 0.1379 | 0.1379 | 0.1379 | 70.647 | 70.647 | 188.7 | 188.7 | 188.7 |
| LOO1-WLS (beta=0.5, stream=False) | 1/1 | 0/1 | 0.1379 | 0.1379 | 0.1379 | 43.734 | 43.734 | 427.9 | 427.9 | 427.9 |

## Hostile Code Audit Verdict (`ADP/` diff & new modules)

- **Verdict:** **PASS (0 blocking defects)** across `ADP/core/loo1_wls.py`, `ADP/engine/common/loo1_wls.py`, `ADP/solver/loo1_wls.py`, `ADP/core/ADP_Config.py`, `ADP/engine/common/index_fit.py`, `ADP/core/multi/ADP_multi_index.py`, and `ADP/__init__.py`.
- **Engineering & allocation rules:** Zero `np.linalg.inv`, zero `(J, n, d)` / `(E, m, m)` / `(d, d)` forbidden tensors, `from __future__ import annotations` and strict boundary validation present across all new modules, `ruff check` / `ruff format --check` / `pyright` all pass cleanly.
- **Mathematical & numerical invariants:**
  - Exact 2-DGEMM evaluation (`U_all = X @ Y`, chunked `reduceat`/`repeat`/batched `np.linalg.solve` + `np.bincount`, `grad_E = X.T @ Gamma`), including exact centering adjoint and both hard (`smooth_cap_eps == 0`) and overflow-safe $C^1$ softplus leverage capping.
  - Preconditioner guard `tau = preconditioner_reg * max(1.0, tr(R_+))` enforces $\kappa_2(R_+ + \tau I_m) \le 1 + 1 / \rho_{\text{reg}}$, with polar SVD retraction, BB step scaling, and Armijo fallback to unpreconditioned Riemannian gradient.
  - Streaming `distance2`-free path matches dense path (`0.1379` vs `0.1379` EDR at `(4000, 100, N_J=3400)` while reducing `VmHWM` from `427.9 MiB` to `188.7 MiB`, `-55.9%`).
- **Non-blocking observations:** `evaluate_loo1_wls` projects `X @ Y` directly without copying `(X - X[0])` (intentional memory avoidance in the solver hot loop; assumes standard feature scale rather than $O(10^8)$ offsets during optimization). In `family-additive` (`beta=0.5`, `(1000, 25)`), `LOO1-WLS` can regress vs `LiveADP` (worst seed 7: `0.3737 -> 0.5490`), and `family-oscillatory` (`0.94+` / `0.97+`) remains unsolved by both methods.


