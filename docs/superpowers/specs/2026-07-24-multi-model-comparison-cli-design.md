# Multi-Model Comparison CLI Design

## Goal

Turn the existing two-model efficiency comparison into a reusable CLI workflow
for comparing two or more ADP-compatible implementations. A user should only
need to write a small Python file containing named model factories and run one
command to receive complete machine-readable results and plots.

The existing two-model Python API remains supported. The new CLI is exposed
through `run_benchmarks.py compare`.

## User-facing model definition

Models are declared in a regular Python file through an insertion-ordered
`MODELS` mapping:

```python
from adp import ADP, ADPConfig


def baseline():
    return ADP.create(
        "new",
        ADPConfig(show_progress=False),
    )


def candidate():
    return ADP.create(
        "new",
        ADPConfig(show_progress=False),
        stages={"beta_solver": "experimental"},
        registry=build_registry(),
    )


MODELS = {
    "baseline": baseline,
    "candidate": candidate,
}
```

Every value must be a zero-argument callable returning an ADP-compatible model.
The model must support
`fit(X, y, centers=..., beta0=..., directions=...)`. Model names must be
non-empty and unique after converting them to strings and trimming whitespace.
At least two models are required.

The first entry is the baseline. All other entries are candidates and are
compared to it. Mapping order is therefore part of the public configuration and
is preserved in persisted artifacts.

The loader accepts an explicit filesystem path. It imports the file in an
isolated module namespace and reports errors using the path and the invalid
entry name. Validation happens before the output directory is created. Each
factory is called before benchmark work starts, and its result is serialized
with `cloudpickle` to prove that it can be sent to isolated fit processes.

## Command-line interface

The primary command is:

```bash
python run_benchmarks.py compare \
  --models experiments/my_models.py \
  --profile smoke \
  --seeds 0:4 \
  --jobs 4 \
  --output benchmark_outputs/my_comparison
```

The compare subcommand supports:

- `--models PATH`, required model-factory file;
- `--profile {smoke,full}`, using the current experiment-2 parameter grids;
- `--seeds`, using the existing inclusive range or comma-separated syntax;
- `--jobs`, the maximum number of parallel comparison groups;
- `--sample-interval`, the resource-monitor sampling interval;
- `--dpi`, plot resolution;
- `--no-progress`, disabling the progress bar;
- `--require-equivalent`, making numerical disagreement with the baseline a
  command failure;
- `--output PATH`, required artifact directory.

The existing standalone invocation of
`experiments/compare_model_efficiency.py` remains available as a two-default-
model self-check. The existing `compare_models(first, second, ...)` function
remains compatible for callers that already use the Python API.

## Execution model

A comparison group is one `(experiment parameters, seed)` combination. All
models in a group receive identical generated arrays, centers, initial beta,
directions, and true beta.

Groups may run in parallel up to `--jobs`. Fits within one group run strictly
sequentially and use the same assigned CPU where process affinity is available.
Every fit runs in a fresh spawned process, with BLAS and the model's internal
statistics parallelism limited to one worker. Process startup and model
serialization are excluded from `fit_time_sec`.

The model order is cyclically rotated by group:

```text
group 0: baseline, candidate_a, candidate_b
group 1: candidate_a, candidate_b, baseline
group 2: candidate_b, baseline, candidate_a
```

This prevents one model from systematically receiving the first or last
position. The planned and actual order are persisted in `runs.csv`.

Factories are evaluated independently and models are deserialized afresh inside
fit processes. Mutations, caches, and allocated memory cannot leak between
runs.

The runner catches per-fit exceptions. It records a failed row with the model,
scenario, seed, order, exception type, and message, then proceeds with the next
model and group. A group is complete after every selected model has either
produced metrics or a failure row.

## Internal boundaries

The comparison module is split into focused responsibilities:

1. a model-spec loader imports and validates the external `MODELS` mapping;
2. the runner expands comparison groups, schedules isolated fits, and returns
   one long-form runs table;
3. reporting converts runs into per-model and baseline-relative tables and
   plots;
4. the CLI composes loading, running, writing artifacts, and exit-status
   policy.

The general runner accepts an ordered sequence of `(name, model)` entries. The
existing two-model `compare_models` wrapper delegates to it, preserving the
current function signature and two-model AB/BA results.

Baseline-relative comparison uses one row per
`(scenario, seed, candidate_model)`. It does not generate candidate-to-candidate
combinations.

## Persisted artifacts

One invocation writes:

- `runs.csv`: one row per model fit, including scenario identity, seed, model,
  fit order, timing, RSS metrics, quality, objective, encoded beta, selected
  stage timings and calls, result-finiteness fields, and failure details;
- `model_summary.csv`: per-model aggregates grouped by dimension and sample
  ratio, including run count, success count, failure count, median and quartile
  timing, memory, quality, and objective values;
- `comparisons.csv`: each candidate paired with the baseline for the same
  scenario and seed, with speedup, memory ratios, quality gaps, beta/projector
  errors, objective gaps, pair validity, and numerical equivalence;
- `comparison_summary.csv`: per-candidate and per-scenario pair counts, valid
  and equivalent counts, equivalence rate, and median/quartile comparison
  metrics;
- `manifest.json`: schema version, command arguments, source model-file path,
  ordered model names, baseline name, profile, seeds, jobs, sampling interval,
  and artifact inventory;
- `plots/runtime_vs_dimension.png` and
  `plots/memory_vs_dimension.png`: common long-form plots containing every
  model;
- `plots/<candidate>/time_speedup_heatmap.png` and
  `plots/<candidate>/memory_ratio_heatmap.png`: baseline-relative candidate
  plots with filesystem-safe candidate directory names.

CSV files are authoritative. Plots are derived only from saved table content.
The CLI prints a concise final table with each model's success count, median fit
time, median peak RSS increase, and median absolute cosine. It also prints each
artifact path.

Artifact names replace the old ambiguous `paired.csv` and `summary.csv` names
in the new multi-model CLI. The existing two-model writer keeps its current file
names for compatibility.

## Status and error policy

Configuration errors fail before creating output:

- missing or unreadable model file;
- missing `MODELS`;
- fewer than two models;
- empty or duplicate normalized names;
- non-callable factory;
- factory failure;
- incompatible or non-serializable factory result;
- invalid CLI numeric or seed arguments.

Runtime model failures are persisted and do not abort the series. The normal
command returns a nonzero status when any fit failed, after all artifacts have
been written.

Numerical disagreement is informative by default because different algorithms
may intentionally produce different results. With `--require-equivalent`, the
command additionally returns a nonzero status when any complete candidate pair
is not numerically equivalent to the baseline. The equivalence tolerances stay
equal to the existing comparison defaults.

Incomplete pairs caused by a failed baseline or candidate are marked invalid
and are not labeled equivalent.

## Testing

Implementation follows test-driven development.

Unit tests cover:

1. loading an ordered `MODELS` mapping from a temporary Python file;
2. rejection of missing mappings, fewer than two entries, invalid or duplicate
   names, non-callables, factory exceptions, incompatible models, and
   serialization failures;
3. deterministic cyclic model order for three implementations;
4. identical input fingerprints for every model in one group;
5. fresh process IDs and sequential non-overlapping fit intervals inside a
   group;
6. continuation and persisted error details after one model raises;
7. baseline-relative pairing for two candidates without candidate-to-candidate
   rows;
8. model and comparison summaries preserving model identity;
9. strict and non-strict equivalence exit policy;
10. compatibility of the existing two-model API and artifact writer.

An end-to-end CLI smoke test writes a temporary three-model definition, runs
`run_benchmarks.py compare` on one smoke scenario and seed, and verifies all
CSV, JSON, and PNG artifacts plus their model names and row counts.

README documentation includes the minimal model-definition template, the exact
smoke command, artifact descriptions, baseline-order semantics, failure
behavior, and the difference between latency-oriented `--jobs 1` and
parallel-throughput runs.

## Non-goals

- Discovering models automatically from source files.
- Supporting fewer than two models.
- Comparing every candidate pair with every other candidate.
- Sharing mutable model instances across fits.
- Treating numerical disagreement as a default failure.
- Replacing the current single-index benchmark series.
- Changing ADP solver mathematics, convergence rules, or stage APIs.
