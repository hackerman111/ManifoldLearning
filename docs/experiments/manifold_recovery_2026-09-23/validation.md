# M4: frozen held-out validation

Run from the repository root with one BLAS thread and
`UV_CACHE_DIR=/tmp/adp-uv-cache`:

```bash
python -m experiments.manifold_recovery_validate --split validation \
  --out docs/experiments/manifold_recovery_2026-09-23/production_validation
```

The script pairs `estimator="manifold"` and `"local_quadratic"` on identical
generated data and model seeds, alternates fit order, counts all failures in
the recovery denominator, and saves every run in `production_validation/runs.csv`.
Its manifest has the exact seed split, source hashes, versions, BLAS environment,
metric and stop rule. Selection runs are in `production_selection/`; production
quality agrees with the separately implemented probe to `<1e-12` per seed.

| Frozen radial base (`n=240,d=4,sigma_eps=0.05`) | Original | Local quadratic |
|---|---:|---:|
| Selection recovered | 0/10 | 10/10 |
| Held-out validation recovered | 0/20 | **20/20** |
| Validation numerical failures | 10 | **0** |
| Validation successful-fit median RMS sine | 0.455 | **0.0562** |
| Validation RMS sine range | 0.409–0.567 | **0.0378–0.1485** |
| Validation certified inner residual maximum | 6.16e-7 | **5.42e-7** |
| Validation terminal scale | `h_min` for 10 successes | **`h_min` for all 20** |

The candidate meets the predeclared ≥16/20 recovery, zero-error, all-center
quality and certified-inner-solve requirements. It completes the scheduled
scale loop, but neither variant has an outer fixed-point convergence
certificate; this report calls that `completion_pass`, not stationarity.
The original failed ten times: eight rank-zero local slope errors and two
rank-deficient local-linear pilots. All ten failures are preserved in `runs.csv`.

For an independent noiseless varying geometry, five seeds of
`g(x)=sin(x_1)+0.5*x_2^2` with exact local direction
`(cos(x_1),x_2,0,0)/||(cos(x_1),x_2)||` give 5/5 candidate recoveries,
zero errors and RMS sine 0.0758–0.1443. The original gives 0/5, including
one failed pilot. See `noiseless_geometry/`.

On all ten selection seed, a full `solver="hybrid"` reference run for the
candidate differs from CG in final RMS sine by at most `1.34e-9` and also
recovers 10/10. See `selection_hybrid_reference/`. The direct augmented LS
reference for the local B step passes in `tests/test_manifold.py`.

After this single candidate evaluation, M5 packed the symmetric quadratic
coefficients to avoid per-center dense `d×d` matrices. This was an algebraic
representation change, with no candidate reselection or threshold change.
`production_validation_packed_equivalence/` rechecks the same 20 seed against
the preserved original validation artifact: maximum quality change
`1.10e-15`, identical recovery classifications and identical errors. The
packed implementation is the final code path.
