# Verification record

Before implementation Luna ran tests/test_grassman_rank_k.py:13 failed,
1 passed, expected missing grassman.block_rank_k. Two fixture issues corrected.
After implementation:46 existing Grassmann tests pass;15 new block tests pass.
Independent reference tests:2 pass (profile/FD and multistart behavior).
Combined focused run:63 passed in0.67s. Scoped solver Ruff and Pyright pass.
Core solve AST identical HEAD:
511142112ec1c69b24192cc57ad55e6f4ba1cf6ed423b5a3101a82240d09320a.
Luna read-only diff audit found no concrete defect. Final rerun recorded below.

Full suite attempted with `uv run pytest -q`: collection stops4 errors,
including benchmarks namespace import (project root absent from script path).
Retried with `PYTHONPATH=. uv run pytest -q`: collection stops3 errors, all
KeyError 'mi-spokoini-m2-n' in experiments/multiv2.py:79, reached through:

- tests/test_experiment.py
- tests/test_multi_center_diagnostic.py
- tests/test_multiv2_quality.py

These catalog failures predate this task per prior STATE/report; no related
files edited. Whole suite has not passed; do not extrapolate focused success.

## Final focused verification, 2026-10-16

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache
uv run pytest -q tests/test_grassman.py tests/test_grassman_flat.py
tests/test_grassman_optim.py tests/test_grassman_rank_k.py
tests/test_grassman_rank_reference.py tests/test_grassman_rank_counts.py
tests/test_grassman_rank_polish.py`: **71 passed in0.64s**.

Includes46 existing tests,15 block tests,3 independent reference tests,
5 exact evaluation-counter tests and2 chart reference-polish tests. The chart
tests were RED (missing module) before implementation and passed before the
supplementary reference runs. Core/default AST and all32 frozen-source hashes
remain unchanged throughout numerical stages.

Lead final raw audit:54 development fits,30 heldout fits,792 inner workers,
all statusok. All768 solver objective counters match diagnostics+initial/final;
28 frozen input snapshots and fullfit captures/pairing/selection hashes match.
Worker thread limits1, max endpoint orthogonality4.8667e-14, no positive
profile-history drift. Audit details: results/lead_audit.json.

Final scoped `ruff check` and `pyright` cover grassman.py, grassman_exp.py,
all experiment Python files and all four new test modules: all checks passed;
Pyright0 errors/0 warnings. No frozen numerical source changed.
Separate polish manifest/source hashes pass;5/5 endpoints improve,3 newly
certified, total22/24 references. Two m2/d300 references remain uncertified.
All three original/polished/certified analysis variants regenerated. Synthetic
checks cover missing worker metrics, same-seed/two-scenario pairing, matched
EDR trace points and duplicated terminal-certificate iteration indexing.
Direct k>1 comparison audit includes all200 pairs,100pervariant,0missing;
cost-grid pairs need no F_ref for direct objective/EDR/time comparisons.
