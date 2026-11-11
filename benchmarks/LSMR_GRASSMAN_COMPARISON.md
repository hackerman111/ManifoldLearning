# LSMR and Grassmann comparison for `m >= 3`

Run the paired comparison with:

```bash
rtk proxy python -m benchmarks.lsmr_grassman_comparison
```

The per-fit records and source hashes are written to
`benchmark_outputs/lsmr_vs_grassman.json`.

## Protocol

- Multi-index ranks `m=3` and `m=4`, with three identical data/init seeds per
  solver and workload.
- Workloads `(n, d, m) = (120, 10, 3)` and `(200, 20, 4)`; one outer ADP step
  per fit.
- Both solvers receive a 20-step cap and `tol=1e-6`; the final index is scored
  by trace score against the known synthetic subspace.
- All 12 fits completed without exceptions. The JSON retains every fit,
  including full diagnostics and any failures if they occur on a rerun.

`lambda_penalty=0.05` has different roles: it damps Grassmann's GN step and
regularizes LSMR's correction subproblem. The same numeric value is used, but
the optimization paths are not parameter-identical.

## Results

Medians over the three seeds:

| `(n, d, m)` | Solver | Solver time (s) | Total time (s) | Trace score | Converged |
|---|---|---:|---:|---:|---:|
| `(120, 10, 3)` | Grassmann | 0.0185 | 0.0216 | 0.9905 | 1/3 |
| `(120, 10, 3)` | LSMR | 0.7827 | 0.7852 | 0.9502 | 0/3 |
| `(200, 20, 4)` | Grassmann | 0.0200 | 0.0235 | 0.9738 | 0/3 |
| `(200, 20, 4)` | LSMR | 3.2068 | 3.2101 | 0.8037 | 0/3 |

At the same 20-step cap, LSMR's median solver time was about 42x higher for
`m=3` and 161x higher for `m=4`. Its median trace score was lower by 0.0403
and 0.1701, respectively. LSMR did not meet its convergence certificate on
any of these six fits; it also rejected 5–8 trial steps per fit for exceeding
the trust radius. Grassmann converged on one of three `m=3` fits and none of
the `m=4` fits, so most trajectories are budget-limited.

This bounded synthetic comparison favors Grassmann for the tested `m>=3`
workloads. It does not establish the best-tuned LSMR configuration or predict
performance on other data and full multi-step outer schedules. Keep Grassmann
as the default; LSMR remains available explicitly with `--solver lsmr`.
