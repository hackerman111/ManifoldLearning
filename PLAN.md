# Repository folder organization

Task `four-directory-reorganization-20261107`; status **done**.

**Goal:** Leave four visible project directories at the repository root: `docs/` for documents and notes, `ADP/` for solver code, `tests/` for all tests, and `benchmarks/` for benchmark/experiment code and saved results.

**Non-goals:** No deletion of user files or research artifacts, no numerical or estimator changes, no edits to frozen result contents or source snapshots, no commit, and no test-suite run unless requested.

**Known evidence:** `ADP/` is the solver package. `experiments/` is an importable package invoked as `python -m experiments...` and included by `pyproject.toml`. Preserve that public import and module-run namespace with a root `experiments.py` shim whose package path points to `benchmarks/experiments/`. `SVD/` contains manuscripts and research notes; the live SVD solver is already in `ADP/solver/`. `verification/` and `benchmark_outputs/` contain frozen historical artifacts. The working tree has extensive pre-existing changes that must be retained.

**Change class:** Structural/path-only; mathematical invariants are not applicable. Do not write into saved result trees. Keep `experiments.*` imports and `python -m experiments...` available through the shim; update active filesystem path references, package configuration, and navigation paths. Do not rewrite files under saved results to repair historic path strings; provide a relocation map instead.

**Target layout:**
- `docs/agent-notes/`, `docs/tex/`, `docs/SVD/`, existing `docs/`, and loose root notes/PDFs.
- `ADP/` unchanged as the live solver package.
- `tests/` with old singular `test/` suite under `tests/legacy/`.
- `benchmarks/experiments/`, existing benchmark scripts, and `benchmarks/results/{outputs,verification}/`; root `experiments.py` maintains the old Python namespace.

**Read set:** `AGENTS.md`; `docs/agent-notes/WORKFLOW.md`; `pyproject.toml`; root `README.md`; `benchmarks/experiments/README.md`; `benchmarks/results/verification/README.md`; `docs/agent-notes/README.md`; active Python/docs references to the former paths.

**Work units:**
1. [x] Preserve the pre-task `PLAN.md` and `STATE.md`; inventory source/result files and destination collisions.
2. [x] Move whole directories and loose notes into the four target directories. The 50,668-file path inventory has no missing or extra entries. The SHA-256 comparison has one expected difference from the `STATE.md` checkpoint rewrite and 74 unexplained differences in older captures/results; repeated post-move scans are stable.
3. [x] Add the `experiments` package-path shim, update package configuration, output defaults, active filesystem paths, and navigation; add a path migration note for frozen artifacts.
4. [x] Verify the four non-hidden root directories, pre-move path inventory, `experiments` imports and module CLI, and relocated navigation links. Record the hash discrepancy without restoring or editing any result artifact.

**Stop conditions:** Any destination collision, missing file, or need to write into frozen results stops that portion; document unresolved integrity discrepancies rather than changing evidence. The initial 50,668-file inventory matched counts but 75 hashes differ from the post-move scan; repeated post-move scans are stable and no active Python writer was found. Cause remains unknown, so no frozen result content will be changed.
