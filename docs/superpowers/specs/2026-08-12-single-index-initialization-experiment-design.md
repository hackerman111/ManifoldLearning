# Single-Index Initialization Experiment

Date: 2026-08-12

## Goal

Compare `index_init="local"` and `index_init="random"` for the uppercase
single-index ADP model. Each comparison uses identical generated data and the
same effective model seed. The existing `ADP_Experiment` runner owns execution,
resume, telemetry, CSV export, paired deltas, and plots.

## Experiment Grid

Use 25 seeds, `0..24`, and nine unique points:

- vary `n` over `250, 500, 1000, 2000, 4000` at `d=25`;
- vary `d` over `5, 10, 25, 50, 100` at `n=2000`;
- include `(n=2000, d=25)` only once.

This produces `9 * 25 * 2 = 450` fits. All points use Gaussian features,
`Y = sin(X @ beta_true) + 0.05 * noise`, and the built-in paired data factory.

## Configuration

Add one file, `ADP/examples/experiment_single_index_init.py`, exporting one
`ADP_Experiment` with variants `local` and `random`. They differ only in
`ADP_Config.index_init`.

The common settings are:

```text
N_loc=32
N_lin=128
N_J=128
N_phi=10
lambda_penalty=100
local_ridge=1e-8
solver=lsmr
solver max_steps=10
```

`N_lin=128` is valid for the largest `d=100`; `N_J=128` is valid for every
point because `ceil(4000 / 32) <= 128 <= 250`.

## Execution and Verification

The experiment runs through the existing CLI:

```bash
python ADP/cli.py \
  --experiment-file ADP/examples/experiment_single_index_init.py \
  --output-dir ADP/experiment_outputs
```

Before the full run, `--dry-run` must validate the file and report 450 jobs.
A small direct model smoke must execute both variants on the same generated
point and seed. No new runner, report format, or model behavior is added.
