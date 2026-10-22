# Armijo paired development result

Decision: **EDR gate fails; do not advance to validation**.

- Clean output: `benchmark_outputs/r_sdr/grassman-armijo-dev-2026-10-18-repaired/`;
  baseline ADP-Grassmann, candidate ADP-Grassmann-armijo, R MAVE-package MEANMAVE.
- Protocol: d-5/d-20/d-50, n=500, m=2; ten paired seeds 95800–95809 per case;
  Armijo max_steps=10, tol=1e-6, local_ridge=0.
- Completion: 90/90 recorded, 90 `ok`, 30/30 paired fits, zero raw errors.
  All 30 input blocks have matching X/Y/data hashes and method seeds; no bad
  blocks. Hash audit and individual digests are in `analysis_summary.json`.
- Paired candidate/baseline EDR geometric ratio **0.955496** (fixed-seed 10k
  bootstrap 95% CI 0.906726–1.019278); required <=0.95. **Fail.** 21 EDR wins,
  9 losses, 0 ties. Worst regression: d-20 seed 95809, EDR .216796 vs .110160,
  ratio 1.9680, delta +.106636.

| Case | EDR ratio / median delta | Time ratio median / p90 | MEANMAVE median EDR / fit s |
|---|---:|---:|---:|
| d-5 | 1.00116 / +.0000284 | .98579 / 1.10533 | .024407 / .151 |
| d-20 | 1.01719 / -.003782 | .85165 / 1.31037 | .079303 / 1.087 |
| d-50 | .85661 / -.025853 | 1.02266 / 1.09474 | .152124 / 4.577 |

- Paired EDR-ratio 95% CIs: d-5 [.99322, 1.01065], d-20 [.91434, 1.19771],
  d-50 [.80555, .91016]; paired median EDR-delta CIs: d-5 [-.000229, .000113],
  d-20 [-.007741, -.000056], d-50 [-.049124, -.015413].
- Paired median time-ratio 95% CIs: d-5 [.88089, 1.04860],
  d-20 [.80718, 1.10639], d-50 [.89490, 1.06538].
- Overall fit-time ratio median/p90 .95545/1.12086 (95% median CI .88737–1.03308;
  gates <=1.10/1.25 pass). Worst paired time ratio is 1.4797 at d-20/95804.
- Median RSS ratio .99781 (candidate 88.10 MiB, baseline 88.29 MiB); candidate
  max orthogonality error 1.24e-15. Every ADP fit stopped at `h_min`; zero inner
  certificates. Candidate accepted steps by case: 200/495/635; rank-guard
  fallbacks 200/490/570; GN solves 0/5/65. No convergence claim.
- Prior `grassman-armijo-dev-2026-10-18/` is excluded: a runner metadata route
  sent Armijo JSON to the R TSV parser, producing `StopIteration` after the
  worker had written its basis. The repaired output above uses the same solver,
  seeds and settings; only integration was repaired.
- Validation seeds remain untouched. No gate was changed.

Reproducibility: manifest fingerprint `97e8457076337ebf99e5992274eae73c8447006d0470826ee8f863585ad0ac40`;
environment SHA-256 `e9f6d449455c9a4e4ae90830bfd71ff8c3d1b5c593557091a159711e9b153ee2`;
Python 3.13.12, NumPy 2.5.2, SciPy 1.18.1, R 4.6.1/MAVE 1.3.12, one thread.
Solver/helper/runner source hashes and the full paired audit are in the output
manifest and `analysis_summary.json`.
