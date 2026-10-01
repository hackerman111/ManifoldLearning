# Current state
Completed svd-geometric-synthesis-2026-10-01; PLAN.md status done.
Deliverable: SVD/EDR_unified_theory.tex (standalone Russian synthesis),
SVD/theory_2026_10_01/EDR_unified_theory.pdf (15 A4 pages).
Evidence/commands/primary research routes: theory_2026_10_01/REPORT.md.
203 independent checks in 35 groups pass (formula_results.json).
XeLaTeX build, references/glyph/overflow audit pass (build_audit.json);
rendered title/contents, dense math and source-map pages checked visually.
Three frozen rank-one angle-curve seeds: 61 angles, J128/p40/d300/m3,
seven warmed repetitions, one BLAS thread; time ratio .160-.167 including
shared projection; curve discrepancy <=1.7e-16 (angle_results.json).
Memory numbers are tracemalloc-only; no full-fit recovery/optimizer or
native peak memory conclusion. Proposed optimizers not integrated.

Common model: variable projection + reduced operator actions + matrix/chart
or retraction. SVD is exact for separable full G X H and decomposes a fixed
horizontal tangent into commuting plane rotations. Matrix/correction/tangent
rank constraints and fixed-g/profiled losses remain distinct.
New proposals: scalar Schur angle search, adaptive rank-q turns, full core/GN,
spatial tangent Tucker, explicit separation of anisotropy and solver metrics.
Corrected draft claims: negative ridge sign, squared kernel-weight covariance,
normalized Gaussian sketch law, polar boundary, global-angle/Fisher/convergence
claims without assumptions. Independent Luna audit added explicit sqrt(mass)
in joint GN stack; lead added deficient-rank derivative guard.

Routes updated: agent-notes/ADP/multi-index.md and tex/README.md.
Correction: old SVD.tex/SVD_form.tex/chat files are absent in root SVD;
old SVD_solver source map is historical. Current seven-file map: sec:map.
User exclusions respected; preexisting user edits/deletions/moves preserved.
No ADP production source changed. Earlier unfinished joint optimizer task:
SVD/theory_2026_10_01/PREVIOUS_{PLAN,STATE}.md; not resumed here.
Follow-up requires a new scope: isolated profiled optimizer implementation,
paired frozen and full-fit validation before speed/recovery promotion claims.
