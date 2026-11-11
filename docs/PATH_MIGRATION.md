# Repository path migration

The working tree now groups its project directories under four root folders.
This table maps former root paths to their new locations.

| Former path | Current path | Contents |
|---|---|---|
| `agent-notes/` | `docs/agent-notes/` | Agent routing notes and task state |
| `SVD/` | `docs/SVD/` | SVD/Grassmann manuscripts and research notes |
| `tex/` | `docs/tex/` | TeX manuscripts, figures, and archives |
| `experiments/` | `benchmarks/experiments/` | Experiment modules, protocols, and code |
| `benchmark_outputs/` | `benchmarks/results/outputs/` | Saved benchmark outputs |
| `verification/` | `benchmarks/results/verification/` | Frozen verification campaigns and audits |
| `test/` | `tests/legacy/` | Older tests, excluded from the default pytest run |
| `test.py` | `benchmarks/hybrid_optimization_summary.py` | Manual summary script for hybrid benchmark outputs |
| root research notes and `Vlad_EDR.pdf` | `docs/` | Loose documentation |

The Python namespace remains `experiments.*`, and `python -m experiments...`
continues to work through the root `experiments.py` shim. Filesystem paths in
active code and navigation use the new locations. Paths embedded in frozen
manifests, source snapshots, and saved result contents remain historical
provenance and were not rewritten. The integration evidence remains in place;
its links to relocated top-level paths now point to their new locations.
