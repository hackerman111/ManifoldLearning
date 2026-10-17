# ADP outer-stage profile

## Method

Captured the existing `ADP_multi_index.profile_` from the matching production
baseline fit for seeds 0, 1, and 2. No solver or estimator instrumentation was
added. Each run used float64, `n=800, d=50, m=2, N_J=800, N_phi=10`, one
OpenBLAS/OMP/MKL thread, and the existing 900-second per-fit timeout. The
profile's `time_fraction` divides by the sum of its named stage times, not by
the full fit timer; pooled percentages below instead divide pooled stage
seconds by pooled `fit_time_sec`.

## Results

| Stage | Pooled seconds | Share of pooled fit time |
|---|---:|---:|
| Statistics | 43.344 | 56.4% |
| Update | 16.514 | 21.5% |
| Solver | 11.865 | 15.4% |
| Directions | 2.653 | 3.5% |
| Initialization | 2.264 | 2.9% |
| Bandwidth | 0.059 | 0.1% |
| Fit timer not assigned to a named stage | 0.138 | 0.2% |

Across seeds, statistics took 14.294–14.653 s (51.8–59.9% of the sum of named
stage times), update 5.440–5.603 s (19.7–22.8%), and solver 2.472–6.278 s
(10.4–22.6%). Statistics is the largest stage, consistently about 2.6 times
the update time. In the live outer loop, `statistics` encloses creation of the
multi-weight generator and the call to `calculate_statistics`; because the
weights are lazy, its time includes their consumption. `update` calls
`calculate_alpha_k` for multi-index fits. Thus the top-level profile identifies
the dominant work without attributing lazy weight generation separately from
moment/statistic assembly.

| Seed | Fit seconds | Solver wall seconds | Projector distance | Peak RSS MiB | Outer iterations / stop |
|---:|---:|---:|---:|---:|---|
| 0 | 25.088 | 3.054 | 0.05417 | 105.2 | 162 / `h_min` |
| 1 | 23.916 | 2.412 | 0.04728 | 111.0 | 162 / `h_min` |
| 2 | 27.833 | 6.219 | 0.04494 | 107.8 | 162 / `h_min` |

For each seed, input hashes, final basis hash, and solver diagnostics hash match
the earlier baseline artifact exactly. All three fits selected iteration 161.
Peak RSS comes from `ru_maxrss`; `tracemalloc` stage memory figures are Python
allocation diagnostics and are not additive physical-memory peaks.

## Reproduction and artifacts

Run from the repository root:

```sh
UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.outer_stage_profile_2026_10_06.run_outer --output experiments/outer_stage_profile_2026_10_06/results
```

`runs.jsonl` preserves each raw result and captured `model_profile_`;
`summary.json` contains the compact profile and parity flags; `protocol.json`
records environment, thread limits, and hashes of the harness and profiled
source files. Source locators: `ADP/engine/common/index_fit.py:350-390`
(`statistics` stage), `ADP/engine/common/index_fit.py:477-505` (`update`),
`ADP/engine/common/logger.py:197-220` (profile fractions).
