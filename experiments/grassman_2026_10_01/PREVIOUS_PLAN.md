task_id: svd-geometric-synthesis-2026-10-01
status: done
change_class: theory only; EXACT/NUMERICAL/APPROXIMATE/ESTIMATOR per construction

# Goal / scope
Russian standalone TeX unifying allowed SVD notes: EDR/ADE, anisotropic
localization, SVD as a special case, geometric/tensor generalizations,
primary literature, improvements with proofs and explicit conditions.
No production edits or unverified recovery/global-optimum/novelty claims.
Excluded documentatio/documentation/archive/pdf/PDF directories not read.
Prior unfinished joint optimizer plan/state preserved verbatim in
SVD/theory_2026_10_01/PREVIOUS_{PLAN,STATE}.md; not resumed.

# Evidence / hypotheses / invariants
Accepted construction: profiled operator objective + reduced coordinates
on matrix/tangent/spatial-field/PSD anisotropy representations.
Fixed-g, profiled, matrix-rank, correction-rank and tangent-rank goals differ.
Preserve finite sketch, normalization/mass, kernel arguments, cross terms,
rank/gauge, SPD and constant-rank derivative guards. Full-fit advantage of
proposed optimizers remains a hypothesis; frozen angle evaluation measured.

# Exact read set
SVD/{SVD_solver,SVD_algorithm,SVD_cur,grassmann_rank_one,multiindex}.tex;
SVD/{Tucker,cladue}.md. Critical source ranges in new document Appendix A.
ADP/solver/SVD.py:164-301,690-728,895-1025; LSMR.py:429-489;
ADP/engine/common/weights.py:88-184,calculus.py:390-422.
Primary source links and retrieved claim limits in theory_2026_10_01/REPORT.md.

# Completed units / acceptance evidence
- [x] Seven allowed notes routed; source disagreements explicitly corrected.
- [x] Primary literature verified; known/derived/proposed claims distinguished.
- [x] SVD/EDR_unified_theory.tex: common model, short proofs, SVD limits,
  commuting rotations, Schur profile, GN/core, tangent Tucker, metric guards.
- [x] 203 independent checks in 35 groups passed: formula_results.json;
  independent audit corrected explicit sqrt(mass) in GN stack. No production tests.
- [x] Narrow paired angle benchmark, three seeds, seven warmed repetitions:
  total time ratio .160-.167 including shared projection; curve discrepancy
  <=1.7e-16. No full-fit/recovery/native-memory conclusion: angle_results.json.
- [x] XeLaTeX PDF 15 pages; visual audit and clean references/overflow/glyph
  audit: build_audit.json. Reproducible commands/environment: REPORT.md.
- [x] Affected multi-index/TeX routes corrected; STATE and DECISIONS updated.

# Limits / stop conditions
No production implementation or broad full-fit benchmark required here.
Correct contradictions before using formulas; do not infer GLS/Fisher or
outer convergence from the frozen inner objective. All required artifacts
exist; proposed production integration and recovery experiments are follow-up.
