# SVD vs HYBRID on the heavy 30-second multi-index point

## Frozen workload and protocol

This uses the saved `mi-spokoini-m2-dhigh` point exactly: `m=2`, `d=50`,
`n=800`, `N_loc=10`, `N_lin=100`, `N_J=800`, `N_phi=10`, `a=exp(1/100)`,
`h_min=1`, local initialization, `select_step=last`, no outer-step cap,
`lambda_penalty=0.05`, and solver `max_steps=5`. Data and initialization seeds
0, 1, and 2 were regenerated with the original split seed-bundle scheme; all
three initialization seeds match the saved runs. Each method ran to completion
in a fresh process with float64 and one BLAS thread. The 30-second mark is a
target check, not a cutoff.

Quality is normalized projector distance
`||B_est B_est.T - B_true B_true.T||_F / sqrt(2m)`; smaller is better. RSS is
the process high-water mark in KiB, converted below to MiB. It includes Python,
imports, generated data, and fitting. `tracemalloc` was disabled for this point
so it would not affect the timing target. Raw runs, traces, and protocol are in
`runs.jsonl` and `protocol.json`.

The objectives differ: SVD solves the rank-constrained fixed-g objective from
`SVD.tex`; HYBRID solves the current HPAO correction-penalty objective. SVD
uses rank `r=1`; its other public basis direction is completed from the prior.
Thus the recovery numbers compare these end-to-end procedures, not equivalent
inner objectives.

## Results

| Seed | SVD time (s) | HYBRID time (s) | SVD distance | HYBRID distance | SVD RSS (MiB) | HYBRID RSS (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 43.58 | 58.99 | 0.1183 | 0.0546 | 94.7 | 122.9 |
| 1 | 48.66 | 59.39 | 0.0899 | 0.0481 | 96.9 | 120.3 |
| 2 | 47.56 | 58.92 | 0.0951 | 0.0454 | 95.9 | 125.1 |
| Median | 47.56 | 58.99 | 0.0951 | 0.0481 | 95.9 | 122.9 |

None of the six fits met the 30-second target. SVD was faster in all three
pairs: median paired time ratio SVD/HYBRID was `0.807` (about 19% less time,
or HYBRID was 1.24x slower). SVD's median process peak was 27.0 MiB lower
(about 22% below HYBRID). The earlier single traced HYBRID timing probe was
58.35 s; the untraced seed-0 run was 58.99 s, so it does not explain the
difference from the historical LSMR runs.

HYBRID had lower projector distance in every seed. Its median distance was
0.0481 versus 0.0951 for SVD; the paired SVD-minus-HYBRID differences were
`[0.0637, 0.0418, 0.0496]`. Both improved substantially from the paired initial
distances (median `0.642`). HYBRID's result here used the `cached-svd` linear
backend for all 810 linear solves per fit; its zero LSMR iteration count does
not mean the solver did no work.

All runs completed 162 outer iterations and stopped at `h_min`. HYBRID's
HPAO convergence diagnostic was false for all three runs. For SVD, the
rank-one inner iteration hit its 20-iteration cap in 22, 49, and 43 of the
162 outer updates for seeds 0, 1, and 2. These are solver diagnostics, while
the fit itself completed the prescribed bandwidth schedule.

The older saved experiment on this point was run with `solver=lsmr`, not
HYBRID; its three times were 37.20, 34.53, and 35.39 seconds. It establishes
the original 30-second target context but is not included in this paired
SVD/HYBRID table.

## Conclusion

On this heavy point, SVD traded some recovery accuracy for lower runtime and
RSS: it was about 19% faster and used about 22% less process peak memory, but
its median projector error was nearly twice HYBRID's. Neither method met the
30-second target. With three paired seeds and distinct inner objectives, this
is workload-specific evidence rather than a general ranking.
