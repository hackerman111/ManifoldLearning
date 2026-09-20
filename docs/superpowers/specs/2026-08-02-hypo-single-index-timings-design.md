# Hypo single-index timing output

## Scope

Add aggregated wall-clock timings to the existing
`hypo/single_index/ADP_single_index.py` implementation and print them from
`hypo/single_index/tester.py`.

## API and measurement boundaries

After every `fit`, `model.timings_` contains seconds measured with
`time.perf_counter`:

- `initialization`: validation, center selection, distances, bandwidths, and
  initial beta;
- `rho`: all adaptive rho searches;
- `directions`: all direction generation;
- `weights`: all weight construction;
- `statistics`: all calls to `calculate_statistics`;
- `slopes`: all local-slope updates;
- `lsmr`: all matrix-free beta solves;
- `total`: the complete `fit` call.

The dictionary is reset at the start of every `fit`. Outer-iteration timings
are aggregated rather than printed separately.

## Output and validation

The single-index tester keeps its existing recovery summary and prints one
additional compact line with every timing in seconds. The focused check
verifies that the expected keys exist, values are finite and nonnegative, and
`total` is positive. No timing thresholds are used because runtime depends on
the host.
