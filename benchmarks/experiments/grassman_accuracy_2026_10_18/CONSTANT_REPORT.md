# Constant-center development result

Decision: **closed at development; do not use validation seeds**.

- Protocol: `CONSTANT_PROTOCOL.md`; candidate `ADP-Grassmann-constant`.
- Development output: `benchmarks/results/outputs/r_sdr/grassman-constant-dev-2026-10-18/`.
- Planned/observed: 30/30 paired fits across d-5, d-20, d-50; zero errors.
- Pair/input audit: all 30 `data_sha256` values unique; baseline and candidate
  X/Y hashes match within each pair.
- Candidate/baseline paired EDR geometric ratio: **0.9997287337** (95% paired
  bootstrap interval 0.9804761–1.0237507); required ratio is <=0.95. **Fail.**
- Per-case paired geometric EDR ratios: d-5 0.99999997; d-20 1.00660872;
  d-50 0.99262645. No case median EDR delta exceeds +0.005.
- Paired fit-time ratio median/p90: 0.87490/1.15042 (gates <=1.10/1.25 pass).
- Median RSS ratio 1.00301 (gate <=1.10 passes); max candidate orthogonality
  error 2.05e-15. All 30 candidate fits stopped at `h_min`; no inner call was
  certified, so this is not a convergence claim.
- Validation seeds 94800–94809 remain untouched.

Raw evidence: `runs.csv`, `paired.csv`, `runs.jsonl`, `manifest.json`,
`completion.json`, `fits/`, `inputs/`, `truth/`, and `source/` in the development
output directory. Reproducible aggregate and per-case audit:
`analysis_summary.json` (analyzer SHA-256
`a2113c9273af1c82d2af8c98214d34b67a3f7c0697ec96c2b0e54e26cb7c838f`).

The failure is the prespecified EDR quality gate; favorable timing does not
advance this candidate to validation.
