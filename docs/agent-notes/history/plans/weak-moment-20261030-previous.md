# Separate EDR research archive from JSS submission source

Task `edr-m-research-separation-2026-10-29`; status **done**.

Goal: move every experiment, hypothesis, test, and saved verification artifact from `Vlad_EDR_M/Code` into the already authoritative `EDR_code` checkout, then leave M as a focused ADP source/manuscript package.

Constraints: preserve all EDR_code changes; never overwrite an existing target file with different content; retain conflicting M versions in a dated provenance archive; keep historical outputs byte-identical. This is repository organization only: do not change estimator behavior or scientific claims, and do not run research/tests.

Work units:
1. Inventory differences and route missing/equal/conflicting material into EDR_code.
2. Remove research-only trees and experiment CLI helpers from M; retain ADP library and manuscript; replace its research README/package metadata with submission-source navigation.
3. Recorded source/target hashes and updated EDR_code navigation; verified the transfer manifest and remaining M boundary.

Acceptance met: every moved file is accounted for by `docs/integration/vlad_edr_m_20261029/MOVE_MANIFEST.json`; hashes and destinations are checked in `AUDIT.json`; target content is preserved or conflict-snapshotted; M contains no research trees listed above.
