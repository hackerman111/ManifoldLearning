# Current state

Task `four-directory-reorganization-20261107` is complete. The four visible project directories at the root are `ADP/`, `benchmarks/`, `docs/`, and `tests/`. Root configuration files and hidden tool/cache directories remain at root.

The previous `PLAN.md` and `STATE.md` are preserved at `docs/agent-notes/history/plans/folder-reorganization-20261107-preexisting-plan.md` and `docs/agent-notes/history/state/folder-reorganization-20261107-preexisting-state.md`.

The old `SVD/`, `tex/`, and `agent-notes/` trees are under `docs/`; experiment modules are under `benchmarks/experiments/`; prior root `benchmark_outputs/` and `verification/` trees are under `benchmarks/results/outputs/` and `benchmarks/results/verification/`; legacy `test/` files are under `tests/legacy/`. Loose root research notes and the PDF are under `docs/`. The path map is `docs/PATH_MIGRATION.md`.

The root `experiments.py` shim keeps `experiments.*` imports and `python -m experiments...` working. `pyproject.toml` includes the shim and moved package; the experiment CLI output default points to `benchmarks/results/outputs/experiments/`. Active navigation was updated. A relative-link check across 249 working documents found no missing links.

Verification: importing `experiments.data` resolved to `benchmarks/experiments/data.py`; `python -m experiments.multi --list` completed successfully. The test suite was not run.

The before/after path inventory contained 50,668 files with no missing or extra paths. SHA-256 comparison showed one expected `STATE.md` checkpoint rewrite and 74 unexplained differences in older captures/results. Repeated post-move scans matched exactly. No result files were restored or edited; the unexplained hash discrepancy remains unresolved.

The working tree had extensive pre-existing changes; they were retained. No commit was created.
