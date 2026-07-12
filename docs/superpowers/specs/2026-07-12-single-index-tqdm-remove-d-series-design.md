# Single-index tqdm and D-series removal design

## Goal

Show job-level `tqdm` progress while running the single-index benchmark and
remove every D-series experiment from the executable benchmark catalog.

## Scope

- Remove D01, D02, D03, and D04 from the scenario registry and every profile.
- Remove CLI and README wording that presents `adp_D1_data` as an input to the
  supported benchmark series.
- Keep the untracked local `adp_D1_data/` directory intact. Removing local data
  is not part of this change.
- Keep the generic real-data loader and executor implementation intact. They are
  reusable infrastructure and are not executable experiments by themselves.
- Add one job-level progress bar to the single-index runner in both serial and
  process-pool modes.

## Progress contract

The progress bar has `total=len(pending_jobs)`, uses `job` as its unit, and
advances only after a job has been persisted. Its postfix reports the completed
job's scenario and method. Resume runs count only pending jobs.

The existing flushed newline progress records remain in place. They keep
redirected background logs readable even when terminal-style carriage-return
rendering from `tqdm` is unavailable or inconvenient.

If process-pool creation fails and the runner falls back to serial execution,
the same progress bar continues from the already completed count without
double-counting jobs.

## Tests

- A registry test proves that no scenario ID begins with `D` and no profile
  contains a D-series ID.
- Runner tests use a fake progress bar to verify total, unit, postfix updates,
  serial completion, parallel completion, and fallback accounting.
- Existing persistence, resume, and report tests remain green.
- CLI/README assertions are updated so they no longer advertise D01-D04 or the
  D-series data directory.

## Non-goals

- Deleting `adp_D1_data/` or any other local dataset files.
- Removing real-data schema columns or generic executor code.
- Adding progress bars to the legacy benchmark and stress subcommands.
