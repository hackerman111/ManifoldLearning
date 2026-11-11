# Current state

Task `svd-synthesis-2026-10-24` DONE. No commit or push.

Deliverables:
- SVD/AI_research_synthesis.tex and .pdf: 20-page Russian scientific synthesis
  of AI-generated SVD/ ideas, their mathematical domains and verification.
- SVD/synthesis_2026_10_24/VERIFICATION.md, verify_evidence.py,
  evidence_audit.json, fresh formula outputs and final build logs.

Covers fixed-g SVD/coarse/full refinement, rank distinctions, Grassmann and
Schur/GN/metric/flat/rank-k variants, relative10/combined20, nine-candidate
screen, LoRA/projected follow-up, adaptive centers, Tucker fields, operator
approximations and covariance/GLS hypotheses. Positive, failed and unverified
results separated; citations to primary reports and saved artifacts.

Important provenance findings: kimi 350-run numbers and original five-seed
center tables lack linked raw provenance. Center manuscript has internal
inconsistent/nonmonotone tables. Later 120-fit report-inspired center gate
fails, explicitly not a reproduction. edr_note_checks.py is absent.
LoRA has pilot evidence only; its behavior was not tested by projected10 gate.
Combined20 has no established incremental recovery gain over relative10.

Verification: 640 saved fit rows checked for status/uniqueness/input pairing;
key aggregates independently recomputed, CIs retained from saved analyses.
41 source links exist, 47 source SHA-256s saved. Existing 24 and 203 small
formula checks pass in isolated copies. Final XeLaTeX passes 7/8 clean,
20 pages, no warnings/bad boxes/undefined references/missing glyphs; contact
sheets and larger final pages inspected. git diff --check passes.
No ADP implementation diff, new fit/timing/R/CUDA or broad-suite claim.

Routing updated: agent-notes/tex/README.md. Previous completed PLAN/STATE
preserved verbatim in history/plans/svd-synthesis-2026-10-24-previous/.
Existing dirty user/research work preserved. No required task work remains.
