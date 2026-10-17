# Current state

Task grassman-rank-k-2026-10-16 complete; PLAN.md done.
Branch SVD-theory; prior dirty research retained. Previous metric plan archived
at agent-notes/history/plans/metric-spokoiny-2026-10-15-complete.md.

Explicit grassman.block_rank_k delegates to grassman_exp shared loop. Fixed
rank/energy/bounded-cost, spectral or joint optimized angles (default2steps).
Spectral/flat kernels reused. APPROXIMATE optimizer of identical profile;
true local refit/full-gradient stopping, angle/geometry/Armijo/workspace guards.
Workspace is extra GN scratch, not total RSS/input/cache. Production solve API
and AST unchanged HEAD, SHA511142112ec1c69b24192cc57ad55e6f4ba1cf6ed423b5a3101a82240d09320a.

Evidence: experiments/grassman_rank_k/REPORT.md (tables/plots/inference),
PROTOCOL.md/MATH.md/REFERENCE_POLISH.md, freeze_manifest.json32sources28inputs,
source_snapshot.zip/reproducibility_manifest.json for exact dirty-tree sources.
Luna handled routine tests/harness/serial workers/aggregation; lead audited
source, numerical invariants, paired provenance, reference and interpretation.
71 focused tests pass; scoped Ruff/Pyright pass. Full pytest collection blocked
3 existing KeyError mi-spokoini-m2-n catalog errors; verification.md has paths.

All881 numerical workers ok:54 devfits30heldfits792inner5referencepolish.
Fresh dev81600..81602/held81700..81704. selection.json froze block1 beforeheld;
no candidate meets20% time gate. Heldout median paired fit/core .99963,
bootstrap95%[.98914,1.01554]; median EDRdiff0, one positive regression+.000120.
Core/spectral/block1 each complete10/10, recoverEDR<=.1 in5/10, certificate
0/840 calls (budget5). Original MI scenario core/optimized all30updates/fit
use rank guard; spectral lacks that guard, changes trajectory/EDR. Keep this
limitation; never silently regularize statistics. Original Spokoiny162outer
steps retained; core solver only~10.1%fit time. Default core stays unchanged.

Inner768solver24refs, main m2/3/5/8,d20/100/300,J64, extraJ16/256. Counted
observational wrapper separately frozen, all exact profile counters verified.
Max orth4.87e-14, no positive F-history drift. Original19/24refs certified;
independent chart L-BFGS+CG supplement improves all5, certifies3, total22/24.
Two m2d300 refs remain uncertified, retained; Fref never global-minimum proof.
All200 adjacent-k comparisons reduce one-step F. Strict independent heldout
law c=.9779/.9190 (optimized/spectral), logRMSE=.1837/.2269; descriptive trend
only. 340 lower-F/worse-EDR pair comparisons retained. Matched-prefix budgets
use EDR at same state, accepted updates exclude duplicate final certificate.
Direct k>1 block-vs-sequential comparisons200,100pervariant, include no-ref
cost grid. Typically cheaper single block but worse objective/recovery.

No work pending. Future optimizer tuning requires new protocol/seeds; do not
reuse consumed heldout as independent validation. Relevant routes: solvers.md,
experiment report and verification.md; all raw JSONL/CSV/captures saved.
