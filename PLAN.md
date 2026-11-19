# LOO1-WLS-ADP Multi-Index Implementation (`feat/loo1-wls-adp`)

Task `loo1-wls-adp-implementation-20261119` is **done** on branch `feat/loo1-wls-adp`.

**Goal:** Implement the opt-in `ESTIMATOR` multi-index variant `estimator="loo1_wls"` (`LOO1-WLS-ADP` with subcritical anisotropy spectrum $\ell_{k+1,a} = (s_a/s_1)^\beta$, $\beta \in (0, 1)$, $R^{-1}$-preconditioned Grassmann solver with $\kappa_2$-guarded regularization, chunked vectorized 2-DGEMM CSR evaluation, and streaming `distance2`-free execution for $N_J \approx n$ at large $(n, d)$), verified by TDD unit tests and a reproducible paired benchmark against live ADP.

**Non-goals:**
- Do NOT alter default `estimator="new"` or `"legacy"` behavior or public API compatibility.
- Do NOT add external dependencies (pure NumPy/SciPy only).
- Do NOT add GPU/CuPy kernels for `loo1_wls` in this branch (raise `NotImplementedError` explicitly for `gpu=True` and single-index mode).
- Do NOT claim finite-sample uniform per-seed dominance or unproven global convergence.

**Change class:** `ESTIMATOR` (explicit opt-in via `ADP_Config(estimator="loo1_wls", loo1_wls=LOO1WLSConfig(...))`).

**Known evidence & proved mathematical invariants:**
1. *Theorem 1.1' (exact 3-term regularized & clipped LOO1 noise identity):* For $\tau_j \ge 0$, $\ell_{\max} \in (0, 1)$, arbitrary weights $a_{ji}$, and heteroskedastic $\sigma_i^2$, `LOO1` ($r_{ji}^2/(1-\tilde\ell_{ji})$) cancels the $O(\bar\ell)$ linear noise bias identically (exact zero for boxcar at $\tau=0$; bounded clipping + ridge + $O(\ell_{\max,j}^2)$ remainder otherwise).
2. *Theorem 1.2 (population minimizer):* Under non-degenerate support (A1)–(A3), global minimizers of $F_{\rm WLS}^{\rm pop}$ satisfy $m D^2 \le L_2^2 \sum_j N_j r_{\parallel,j}^4 / (\underline\lambda \lambda_m(M_g^{\rm pop})) \to 0$ as $h_k \downarrow 0$ without assuming Gaussian $X$.
3. *Theorem 1.3 (exact gradient & $O(m)$-invariance):* $F_{\rm LOO1}(YQ) = F_{\rm LOO1}(Y)$ for all $Q \in O(m)$; exact Euclidean gradient is $X^\top \Gamma$ (2-DGEMM: $4 n d m$ ambient flops, independent of $J$).
4. *Theorem 2.1 (anisotropy phase transition):* Updating $\ell_{k+1,a} = (s_a/s_1)^\beta$ with $0 < \beta < 1$ (default $\beta = 0.5$) makes $0$ repelling and eliminates the quadratic weak-direction collapse of live $\beta = 2$.
5. *Conditioning guard:* Preconditioner $R = \sum_j N_j c_j c_j^\top$ uses $\tau = \rho_{\rm reg} \max(1.0, \operatorname{tr}(R))$, strictly bounding $\kappa_2(R + \tau I_m) \le 1 + 1/\rho_{\rm reg}$.
6. *Streaming memory invariant:* When `stream_distances=True` (or auto-triggered for $J n > 4\times 10^6$), no dense $(J, n)$ `distance2` or $(J, p, d)$ sketch arrays are materialized; results match non-streaming to $\le 10^{-11}$.

**Read set:**
- `ADP/core/ADP_Config.py`, `ADP/core/experimental_centers.py`, `ADP/core/ADP_Solver.py`, `ADP/core/multi/ADP_multi_index.py`, `ADP/__init__.py`
- `ADP/engine/common/index_fit.py`, `ADP/engine/common/weights.py`, `ADP/engine/common/calculus.py`, `ADP/engine/common/initialize.py`, `ADP/solver/LSMR.py`
- `pyproject.toml`, `docs/agent-notes/contracts/{engineering,numerics,research}.md`

**Bounded work units:**
1. [x] **WU1 (TDD RED — `tests-hydrogen`):** Wrote `tests/test_loo1_wls.py`; `uv run pytest tests/test_loo1_wls.py` failed during collection as expected with `ImportError: cannot import name 'LOO1WLSConfig' from 'ADP'`.
2. [x] **WU2 (Core Implementation — `core-impl`):** Implemented `ADP/core/loo1_wls.py`, `ADP/engine/common/loo1_wls.py`, `ADP/solver/loo1_wls.py`, and integrated into `ADP/core/ADP_Config.py`, `ADP/engine/common/index_fit.py`, `ADP/core/multi/ADP_multi_index.py`, `ADP/__init__.py` without altering `"new"`/`"legacy"` behavior.
3. [x] **WU3 (TDD GREEN & Repo Gates — `tests-hydrogen`):** `uv run pytest tests/test_loo1_wls.py -v` passed **80/80** (`2.45s`, zero warnings with `filterwarnings = ["error"]`); `tests/test_loo1_wls.py` + `tests/test_index_models.py -k "not pilot"` passed **116/116**; repo-wide non-collection-error `pytest` passed **1172** (with 32 pre-existing unrelated failures / 54 skipped / 5 pre-existing collection errors); `ruff check`, `ruff format --check`, and `pyright` on all new/modified files passed with **0 errors / 0 warnings**.
4. [x] **WU4 (Paired Benchmark & Hostile Audit — `bench-hydrogen`):** Independent hostile code audit passed (`0` blocking defects). Single-threaded paired benchmark (`benchmarks/experiments/loo1_wls_study.py`, `290/290` jobs OK, `benchmarks/experiments/LOO1_WLS_REPORT.md`): Suite 1 `Weak/Mixture` mean EDR `0.2201 -> 0.0809` (`10/12` recovery, `11/12` wins, `1.89x` speedup); Suite 2 `(1000,25)` mean EDR `0.3087 -> 0.2495` (`39/70` vs `23/70`, `55/70` wins, `1.46x` speedup); Suite 3 `(2000,50)` mean EDR `0.3203 -> 0.2719` (`10/21` vs `6/21`, `16/21` wins, `1.32x` speedup, max `VmHWM` `231.9 -> 174.5 MiB`); Suite 4 `(4000,100,N_J=3400)` `stream=True` matched `stream=False` EDR (`0.1379`) while cutting `VmHWM` `427.9 -> 188.7 MiB` (`-55.9%`).
5. [x] **WU5 (Checkpoint, Docs & Clean Commits):** Updated `PLAN.md`, `docs/agent-notes/STATE.md`, `docs/agent-notes/DECISIONS.md`, and `docs/agent-notes/ADP/{multi-index,solvers,index-pipeline}.md`, verified post-edit checks, and committed cleanly on `feat/loo1-wls-adp`.

**Verification:**
- `pytest tests/test_loo1_wls.py -v` passes with zero warnings (`filterwarnings = ["error"]`).
- Full `pytest` passes without breaking any existing test.
- `ruff check .` and `ruff format --check .` pass cleanly.
- Paired benchmark artifacts and report verify wall-clock speedup (vectorized 2-DGEMM vs Python loop & live ADP), bounded RSS under streaming, and honest per-family/per-seed EDR comparison.

**Stop conditions:**
- Stop and re-evaluate if analytic vs FD gradient exceeds `1e-7`, chunked vs reference differs by `> 1e-11`, streaming vs dense differs by `> 1e-10`, or existing tests regress.
