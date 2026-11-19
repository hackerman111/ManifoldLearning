# Current state

Task `loo1-wls-adp-implementation-20261119` is **done** on branch `feat/loo1-wls-adp`.

## Active truth
- Added opt-in `ESTIMATOR` multi-index variant `ADP_Config(estimator="loo1_wls", loo1_wls=LOO1WLSConfig(...))` in `ADP/core/loo1_wls.py`, `ADP/engine/common/loo1_wls.py`, `ADP/solver/loo1_wls.py`, and `ADP/engine/common/index_fit.py` (`_fit_multi_loo1_wls`).
- Defaults (`estimator="new"` / `"legacy"`) and `ADP_single_index` behavior are unchanged; `estimator="loo1_wls"` rejects single-index mode and `gpu=True` with `NotImplementedError`.
- Core design:
  - Sketch-free `LOO1-WLS` objective over CSR active neighborhoods (`ridge=1e-4`, `leverage_cap=0.85`, optional $C^1$ `smooth_cap_eps`) evaluated in vectorized chunks via 2 DGEMMs (`U_all = X @ Y`, `grad_E = X.T @ Gamma`) with ambient cost $4ndm$ independent of $N_J$.
  - Subcritical anisotropy spectrum $\ell_{k+1,a} = (s_a/s_1)^\beta$ (`spectrum_exponent=0.5` default) preventing quadratic collapse of weak directions ($\beta=2$ in live code).
  - $R^{-1}$-preconditioned Grassmann solver with condition-number guard $\tau = \rho_{\rm reg}\max(1.0, \operatorname{tr}(R_+))$ ($\kappa_2 \le 1 + 1/\rho_{\rm reg}$), BB step scaling, polar SVD retraction, and Armijo fallback.
  - Streaming `distance2`-free initialization, bandwidth search, weight blocks, and $\alpha$-bisection (`stream_distances=True` or auto when $J n > 4\times 10^6$).

## Verified evidence
- `tests/test_loo1_wls.py`: **80 passed in 2.45s**, zero warnings (`filterwarnings = ["error"]`); `ruff check`, `ruff format --check`, and `pyright` pass with 0 errors/warnings on all modified/created files.
- Hostile code audit of `ADP/` diff: **PASS** (0 blocking defects).
- Single-thread paired benchmark (`benchmarks/experiments/loo1_wls_study.py`, `290/290` jobs OK, `benchmarks/experiments/LOO1_WLS_REPORT.md`):
  - Suite 1 (`Weak/Mixture`, `960x8`, `N_J=816`, 12 pairs): mean EDR `0.2201 -> 0.0809` (`10/12` recovery, `11/12` wins, `1.89x` speedup).
  - Suite 2 (7 families, `1000x25`, `N_J=850`, 70 pairs): mean EDR `0.3087 -> 0.2495` (`39/70` vs `23/70` recovery, `55/70` wins, `1.46x` speedup; `beta=2/3` gave `0.2461`).
  - Suite 3 (7 families, `2000x50`, `N_J=1700`, 21 pairs): mean EDR `0.3203 -> 0.2719` (`10/21` vs `6/21` recovery, `16/21` wins, `1.32x` speedup, max `VmHWM` `231.9 -> 174.5 MiB`).
  - Suite 4 (`4000x100`, `N_J=3400`, `quadratic s0`): `stream=True` matched `stream=False` EDR (`0.1379`) with `VmHWM` `427.9 -> 188.7 MiB` (`-55.9%`).

## Open gaps & handoff
- No finite-sample uniform per-seed dominance (`family-additive` regressed on worst seeds, e.g., seed 7 at `(1000,25)`: `0.3737 -> 0.5490`), and `family-oscillatory` (`~0.94–0.98`) remains unsolved by both methods.
- Relevant notes: `docs/agent-notes/ADP/{multi-index,solvers,index-pipeline}.md`, `docs/agent-notes/DECISIONS.md`, `PLAN.md`.
- Next step when ready: review/merge `feat/loo1-wls-adp` or run an independent held-out multi-seed evaluation against MEANMAVE.
