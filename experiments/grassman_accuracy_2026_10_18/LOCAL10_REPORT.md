# Frozen 10-step candidate development result

**Decision:** R3 fails the runtime gate. Close this candidate and leave
validation seeds 92800–92809 untouched, as required by the frozen protocol.

Protocol: `PROTOCOL.md`; saved run:
`benchmark_outputs/r_sdr/grassman-accuracy-dev-2026-10-18/`.
It completed 90/90 fits: 30 paired datasets across d-5, d-20, d-50, with no
failed/invalid rows. Every case/seed has matching `data_sha256`, X/Y CSV hashes,
and method seed across baseline, candidate, and MEANMAVE.

Manifest fingerprint:
`a54ea61c276b5cae2546276b036b10f7bf79debe9d3f397583a63fe965f455a8`.
Candidate source SHA256:
`cd6261601278414b8696363359b3f04a256f9562f976ee6218231ebfe7ad65a7`.
Baseline `ADP/solver/grassman.py` SHA256:
`6d5aa753da9d9c8532545fa7519e63ba479c0f4082a8a6d52358cd4c66af72a0`.

| Case | Geometric mean paired EDR ratio | Paired median EDR delta | Time ratio median / p90 | Baseline fit median / p90 s | Candidate fit median / p90 s | Baseline / candidate RSS median MiB |
|---|---:|---:|---:|---:|---:|---:|
| d-5 | 1.0074 | +0.000051 | 1.384 / 1.503 | 0.299 / 0.326 | 0.419 / 0.467 | 83.71 / 83.85 |
| d-20 | 0.9502 | -0.005350 | 1.525 / 1.847 | 0.485 / 0.597 | 0.731 / 0.787 | 88.04 / 88.31 |
| d-50 | 0.7854 | -0.027564 | 1.279 / 1.711 | 0.849 / 0.999 | 1.094 / 1.181 | 91.29 / 91.38 |

Across all pairs, the geometric mean EDR ratio is 0.9093 (passes <=0.95),
the largest case median EDR delta is +0.000051 (passes <=0.005), median/p90
paired fit-time ratios are 1.368 / 1.729 (fail <=1.10 / <=1.25), and the
candidate/baseline median RSS ratio is 1.003 (passes <=1.10). All 30 pairs are
finite; maximum candidate orthogonality error is 9.49e-16 (passes 1e-10).

Both Grassmann methods stopped at `h_min` in all fits; neither has an inner
certificate. Outer convergence remains unknown. No convergence claim is made.

Detailed paired records and failure denominators are in `paired.csv`,
`summary.csv`, and `runs.csv` in the saved run directory. MEANMAVE is recorded
as `MAVE-package`; its package API does not expose a convergence certificate.
