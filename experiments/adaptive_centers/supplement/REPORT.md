# P1 localization microbenchmark

This supplementary experiment measures the projected-neighbor localization
path and emitted CSR weights. It does not run ADP fits or measure recovery.
Each point is a fresh one-thread process; each median is over three paired
seeds (`93000..93002`). The same `X`, centers, basis, eigenvalues, and cached
distance matrix hashes matched for each arm pair.

| Workload | Mode | Dense baseline median | Projected median | Projected/baseline | Median process peak RSS, baseline / projected |
|---|---|---:|---:|---:|---:|
| `n=500,d=30,J=500,h=1` | weights | 0.0063 s | 0.0172 s | 2.738 | 79.6 / 80.2 MiB |
| same | alpha target 20 + weights | 0.0150 s | 0.0217 s | 1.448 | 78.9 / 81.3 MiB |
| `n=10000,d=100,J=500,h=.4` | weights | 0.0997 s | 0.0492 s | 0.493 | 213.8 / 171.5 MiB |
| same | alpha target 20 + weights | 0.1633 s | 0.0676 s | 0.414 | 213.8 / 171.5 MiB |

Distance-matrix construction was outside both timed paths and measured a median
0.0033 s (1.91 MiB) for the small case and 0.0485 s (38.15 MiB) for the large
case. It was computed from the same `X` and centers in every fresh process.
Timing includes the arm's projected/dense localization preparation and common
CSR output assembly. In alpha-plus-weights mode, the projected path reuses its
prepared geometry for alpha and weights, while the dense baseline follows its
existing separate calculations.

Numerical checks were exact at the emitted weights: all 12 pair/mode checks had
zero maximum absolute weight error and zero support mismatches; all alpha
checks had zero error. Median active weight counts were 81,536 (32.6% density)
versus 81,536 for the small weight-only check, and 203,677 (4.1% density) for
the large weight-only check. The alpha target reduced those counts to 27,200
(10.9%) at small size and 46,126 (0.92%) at large size. The large projected
case used about 42 MiB less process peak RSS; the small case used slightly
more. Baseline weights are converted from streamed dense blocks into the same
CSR output format for this comparison, so this does not measure native
streamed-baseline peak RSS.

Inputs use AR(1) Gaussian features (`rho=.5`), basis dimension two,
eigenvalues `[1,.25]`, and alpha `.02`. Centers are deterministic evenly spaced
row indices from `X`; this is a fixed candidate pool, not random center
sampling. See `PROTOCOL.md`, `results/manifest.json`, per-worker JSON records,
`results/pairing_audit.json`, and `results/paired_results.csv` for full
provenance. The supplement manifest pins the original full-fit freeze manifest
and source snapshot; it is intentionally independent of the main freeze list.
The supplementary runner verified its own and numerical source hashes against
that manifest after the 24 workers completed. A later compatibility-only
`seed=None` entropy patch was applied after timing; it is outside the measured
snapshot and is documented in
[`verification/compatibility_overlay.json`](../verification/compatibility_overlay.json).

The size crossover is consistent with the search paying off only when `n` is
large enough for a sparse projected support to outweigh tree and construction
cost. It neither establishes a universal threshold nor implies faster full
fits. In the frozen full-fit development run (`n=500`), the exact-search arm
was slower than floor25 despite nearly identical recovery metrics.
