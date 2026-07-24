# Single-Index Manual Parameters Design

## Goal

Extend `python run_benchmarks.py single-index` so one benchmark series can
override the standard experiment grids with explicit data axes and core ADP
settings. The ordinary scenario registry, deterministic seed streams,
checkpointing, resume validation, CSV schemas, and report generation remain
the execution path.

Without the new flags, the command must produce the same jobs, identifiers,
algorithm configuration, and artifacts as before.

## Command-line interface

The command accepts comma-separated selections for every data parameter:

- `--d`;
- `--n-over-d`;
- `--sigma` and its alias `--sigma-eps`;
- `--sigma-x`;
- `--rho-corr`;
- `--link`;
- `--x-distribution`;
- `--noise-distribution`;
- `--heteroscedastic`;
- `--outlier-fraction`;
- `--outlier-scale`;
- `--delta`;
- `--center-fraction`.

It also accepts one value per series for the core ADP settings:

- `--outer-steps`;
- `--inner-steps`, with `--max-inner-steps` as an alias;
- `--tol`;
- `--objective-check-every`;
- `--n-directions`;
- `--min-neighbors`;
- `--lambda-penalty`;
- `--ridge`.

Data selections form a Cartesian product. ADP settings do not form additional
grid axes: one supplied value applies to every job in the series. Omitted flags
preserve the current profile and `ADPConfig` defaults.

Example:

```bash
python run_benchmarks.py single-index \
  --profile full \
  --experiments 2,3,4,5,6 \
  --d 50,100 \
  --n-over-d 5,10 \
  --sigma 0.5 \
  --sigma-x 1 \
  --rho-corr 0 \
  --link quadratic \
  --x-distribution gaussian \
  --noise-distribution gaussian \
  --heteroscedastic false \
  --outlier-fraction 0 \
  --outlier-scale 1 \
  --delta 0 \
  --center-fraction 1 \
  --outer-steps 4 \
  --inner-steps 500 \
  --tol 1e-6 \
  --objective-check-every 2 \
  --n-directions 32 \
  --min-neighbors 10 \
  --lambda-penalty 1 \
  --ridge 1e-10 \
  --seeds 0:99 \
  --jobs 9 \
  --output benchmark_outputs/single_index_manual
```

## Grid expansion

The selected profile and experiment registry remain the source of experiment
identity. For each selected experiment:

1. build its normal smoke or full parameter grid;
2. replace every supplied data field with the requested selection;
3. form the Cartesian product of all requested selections;
4. retain all unsupplied fields from each original grid row;
5. remove duplicate `ExperimentParameters` rows while preserving deterministic
   order.

This makes `--d 50,100 --n-over-d 5,10` replace those two axes while experiment
3 can still retain its normal noise axis when `--sigma` is omitted. Supplying
`--sigma 0.5` replaces that axis as well.

The current `--center-fraction` scalar syntax remains compatible; it becomes a
comma-separated data selection using the same validation and grid semantics as
the other data fields.

## Series configuration and identity

`SingleIndexSeriesConfig` stores normalized manual data selections and optional
ADP overrides as immutable, typed values. These fields participate in the
existing canonical configuration fingerprint.

Expanded `ExperimentParameters` continue to participate in each job's
`run_id`. The effective ADP overrides are also included in the job identity
fingerprint. Therefore:

- changing any requested data value changes the affected job identifiers;
- changing an ADP override changes both the series fingerprint and affected
  job identifiers;
- resume accepts only an identical manual grid and identical ADP overrides;
- reordered or duplicated selections normalize to the same canonical
  configuration.

Seeds remain paired within each experiment under the existing `SeedBundle`
design. Manual parameter values do not alter unrelated seed streams.

`--dry-run` expands the same jobs as a real run and reports the resulting count
without creating output.

## ADP configuration

The executor starts from the current `_benchmark_adp_config(job)` defaults.
Each non-null series override is then applied to the corresponding
`ADPConfig` field:

| CLI flag | ADPConfig field |
| --- | --- |
| `--outer-steps` | `outer_steps` |
| `--inner-steps`, `--max-inner-steps` | `inner_steps` |
| `--tol` | `tol` |
| `--objective-check-every` | `objective_check_every` |
| `--n-directions` | `n_directions` |
| `--min-neighbors` | `min_neighbors` |
| `--lambda-penalty` | `lambda_penalty` |
| `--ridge` | `ridge` |

The explicit `--n-directions` override supersedes the current
`max(4, min(d, 32))` derived value. `execute_job` passes the same resolved
direction count to both synthetic-data generation and `_benchmark_adp_config`.
The generator retains its current derived default when the override is absent.
This keeps direction-array shapes and algorithm metadata consistent.

Every run records all effective `adp_*` values in `run_summary.csv`. The
requested manual selections and ADP overrides are also recorded in
`series.csv`, so a saved series can be reconstructed without guessing its
configuration. Canonical selections are encoded as stable delimiter-separated
values. Adding these public columns increments the single-index schema version;
older saved series remain readable only by their matching older code version,
rather than being silently resumed under a different schema.

## Validation and errors

Parsing and validation happen before output creation:

- integer counts must be positive;
- `tol`, `min-neighbors`, `lambda-penalty`, and `ridge` use their corresponding
  finite positive or nonnegative constraints;
- data values use `ExperimentParameters` validation;
- categorical selections reject unknown values;
- boolean selections accept explicit `true` or `false`;
- empty lists and malformed comma selections are rejected.

`--resume`, `--reports-only`, and `--retry-failed` retain their existing
combination rules. A resume configuration mismatch remains a hard error.

## Artifacts and reports

The existing artifact set remains unchanged:

- `run_summary.csv`;
- `outer_iterations.csv`;
- `inner_iterations.csv`;
- `local_diagnostics.csv`;
- `solver_iterations.csv`;
- `series.csv`;
- `artifacts.csv`;
- generated plots and summaries.

Reports consume the expanded persisted rows, so manual parameter values appear
in the same grouping and plotting pipeline as standard profile values. No
separate manual-experiment reporting path is introduced.

## Testing

Implementation follows test-driven development.

Tests cover:

1. help output exposes every new flag and both inner-step aliases;
2. two manual data axes expand to their exact Cartesian product;
3. supplied axes replace standard experiment axes while unsupplied axes remain;
4. duplicate and reordered selections normalize deterministically;
5. every core ADP override reaches the effective `ADPConfig`;
6. `--n-directions` controls both generated direction shape and saved metadata;
7. run IDs change for data overrides and series fingerprints change for ADP
   overrides;
8. resume succeeds with identical overrides and rejects mismatches;
9. `series.csv` and `run_summary.csv` retain requested and effective values;
10. invalid data and ADP values fail before output creation;
11. a real CLI smoke run completes with a small manual grid;
12. existing no-override scenario, CLI, storage, schema, and report tests remain
    unchanged.

## Non-goals

- Replacing the experiment registry with anonymous ad-hoc scenarios.
- Turning ADP settings into Cartesian grid axes.
- Adding a generic untyped `--adp-config`.
- Changing convergence criteria or ADP mathematics.
- Changing seed generation, local-solver comparison semantics, or report
  formulas.
