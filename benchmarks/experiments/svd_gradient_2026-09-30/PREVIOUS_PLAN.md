task_id: svd-current-summary-2026-09-30
status: done

# Current task: live SVD.py description
Goal: standalone Russian SVD/SVD_cur.tex, approximately 1–2 pages, describing
actual entry/refit, fixed-g loop, numerical guards, options and output.
No code or estimator changes. Source of truth: ADP/solver/SVD.py.
Read set: agent-notes/ADP/{solvers,multi-index}.md; SVD.py functions
solve, _solve_with_metric, _solve_fixed_coefficients_validated, _v_step,
_v_certificate, _complete_basis; LSMR.py::_local_refit.
Units: write document; compile; verify 2 pages and source correspondence;
checkpoint. Stop on formula mismatch or unresolved compilation failure.
Evidence: SVD/SVD_cur.tex and SVD/SVD_cur.pdf; LuaLaTeX succeeds, 2 pages,
no Warning/Overfull/Underfull/Missing messages. Formulas, numerical guards
and defaults checked against live SVD.py and imported LSMR.py helpers.

# Previous completed document
previous_task_id: svd-algorithm-summary-2026-09-30
status: done

# Current task: concise TeX algorithm description
Goal: standalone Russian 1–2 page description based on SVD/SVD_solver.tex,
preserving notation, initialization, rank-one search, compression, refit,
stopping and output. No algorithm/code changes or new scientific claims.
Read set: AGENTS.md, agent-notes/{STATE,README,WORKFLOW}.md,
SVD/SVD_solver.tex sections 1–3 and basis extraction in section 5.
Units: write SVD/SVD_algorithm.tex; compile with LuaLaTeX; verify 1–2 pages,
formulas and compilation warnings; checkpoint final paths and evidence.
Stop: resolve any formula mismatch or compilation error before completion.
Evidence: SVD/SVD_algorithm.tex and SVD/SVD_algorithm.pdf; LuaLaTeX succeeds,
PDF has 2 pages, log has no Warning/Overfull/Underfull/Missing messages.
Cache for restricted environments: TEXMFVAR=/tmp/svd-tex-cache and
TEXMFCACHE=/tmp/svd-tex-cache; intermediate build files are in /tmp.

# Previous completed task and evidence
previous_task_id: svd-a-metric-2026-09-30
status: done
change_class: ESTIMATOR (explicit option); whitening is EXACT for the new objective

# Goal
Implement and test spectral proximal A norms in SVD matrix/correction modes.
Keep p=0 default. No default promotion or global recovery claim.

# Evidence / invariants
Current solver minimizes weighted fixed-g data loss plus Frobenius penalty.
P has orthonormal rows. Full kernel tensor is alpha^2 I + P.T Lambda P;
orthogonal kernel instead has alpha^2(I-P.T P)+P.T Lambda P.
Remove h^-2 and normalize by actual largest eigenvalue. A is SPD for alpha>0;
whitening preserves predictions and rank of B and Delta, but transformed
prior is not orthonormal. Complete basis in original coordinates.
No d*d metric. U whitening costs O(J*p*d*m) and one U-sized array, so not free.

# Read set
ADP/solver/SVD.py; ADP/core/{ADP_Solver.py,multi/ADP_multi_index.py};
ADP/engine/common/{index_fit.py,weights.py}; tests/test_svd_solver.py;
benchmarks/svd_vs_hybrid.py; agent-notes/ADP/{multi-index,solvers}.md.

# Units and verification
- [x] R1: derive whitening, SPD/rank domains and audit with dense reference.
- [x] R2: explicit metric_power=0/.5/1, metric_floor, current alpha/spectrum hook;
  reference tests, p=0 regression, invalid/nonfinite input and public fit tests.
- [x] R3: frozen selection seeds 11,23,37; variants p=0,.5,1,1+floor=.1;
  full-fit small/medium and fixed-g d=100/300, both rank targets.
  Separate quality, convergence, objective, timing, memory; preserve failures.
- [x] R4: held-out seeds 101,103,107 only if selection candidate has no failures,
  max quality worsening <=.02 and median paired quality improvement >0.
  Hold configs fixed; no tuning. This is a bounded pilot, not recovery proof.
- [x] R5: report, notes, decisions, done checkpoint.

# Stops
Stop correctness work on reference mismatch; fix before experiments. Stop
candidate validation at failed selection gate; leave held-out seeds untouched.
Resource budget: 240 s per fit; do not rerun old heavy schedule gratuitously.


# Result
Selection: all 96 runs completed. p=1 passed the frozen gate for both rank
modes. Held-out: all 24 runs completed. Matrix candidate passed the pilot
threshold; correction candidate failed it (median quality delta +0.0000703,
max worsening .01061). No default promotion; incomplete inner convergence and
three seeds per case limit the evidence. Full tables and raw evidence:
experiments/svd_a_metric_2026-09-30/REPORT.md. Dense whitening audit:
experiments/svd_a_metric_2026-09-30/AUDIT.md. 91 tests pass; Ruff and diff
check pass; standard Pyright did not finish in 30 s. Six final source replays
match the saved outcomes exactly at recorded metrics.
