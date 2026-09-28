task_id: hpao-a-structure-2026-09-28
status: done
change_class: mathematical and diagnostic research only; production unchanged

# Structure and rank of the HPAO correction operator

Goal: determine whether the live fixed-coefficient HPAO operator A has an
exploitable shift/other fast structure, whether its rank makes a polar-based
solve useful, and whether a minimal-polynomial Krylov method could reduce cost.

Non-goals: change the estimator, solver defaults, tolerances, certificates, or
the parallel manifold-generalization work. Root PLAN.md and agent-notes/STATE.md
belong to that work and must not be edited here.

Known evidence: A is matrix-free with blocks sqrt(mass_j) U_j weighted by
local coefficients; d100 profiling attributes 58.31/74.75 s to A/A* actions.
The previous beta-functional audit found no common U_j.T U_j metric and no
general QR row reduction when P<d.

Hypotheses: H1, there is a common shift or transform basis across A's blocks;
H2, A has useful exact/numerical rank deficiency; H3, the normal operator has
a low-degree minimal polynomial or clustered spectrum allowing cheaper Krylov
solves. Reject a hypothesis if only generic structure is present or setup
exceeds the bounded matrix-free solve.

Invariants: use the live float64 A and A*, original masses, coefficients,
lambda and correction certificate. Distinguish exact rank from numerical
rank; polar factor is not a replacement for ridge. Do not form large JP-by-md
designs or md-by-md normal matrices outside bounded diagnostic snapshots.

Exact read set: agent-notes/ADP/solvers.md, ADP/solver/LSMR.py:429-596,
ADP/engine/common/statistic.py:29-229, docs/experiments/beta_functional_2026-09-27/report.md,
docs/experiments/multi_solver_search_2026-09-24/s1.md, and the six outer0
frozen NPZ inputs/manifest under benchmark_outputs/diagnostic/multi_solver_search_s1_20260924.

Work units:
- [x] Derive block/Kronecker structure and state precise conditions for
  circulant, low-rank, and polar shortcuts; cross-check against live actions.
  Evidence: audit.py tiny explicit-design check; report draft formulas.
- [x] Run bounded, read-only numerical diagnostics on frozen d10/d100 outer0
  inputs: rank/spectrum of A via a small Gram matrix, shift-structure residuals,
  and representative Krylov convergence; save script and machine-readable data.
  Evidence: audit.py and results.json, six manifest-hash-verified inputs.
- [x] Write a conclusion with applicability limits and next candidate only if
  mathematical and measured evidence support it; update this PLAN and local
  STATE, then run script/reproducibility and git diff --check.
  Evidence: report.md; audit.py and results.json; Ruff check/format,
  six-case JSON assertions, and git diff --check passed.

Stop conditions: no production change; stop after the three frozen pairs and
independent tiny algebra check. A numerical shortcut is not declared useful
from one kernel measurement or a matrix identity alone.
