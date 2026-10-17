# Current state

Task adaptive-centers-2026-10-17 **done**, PLAN.md done. Branch SVD-theory.
User requested P0-P3 experimental implementation plus Luna benchmarks.
Initial untracked SVD/adaptive_centers_anisotropic_adp_edr_report.tex preserved.
Previous plan archived at history/plans/grassman-rank-k-2026-10-16-complete.md.

API AdaptiveCentersConfig through ADP_Config.experimental_centers=None;
defaults/solver unchanged. CPU,multi,new,builtin Epanechnikov only.
P0 floor changes localization only; raw canonical spectrum retained.
P1 projected KDTree candidates/exact residuals/CSR and full-pool alpha.
P2 cheap projected moments, score/random controls; P3 normalized LogDet with
coverage, fixed/dynamic budgets, optional unnormalized PoU. Full U only chosen
centers; full-pool init/sketch/error population preserved; coefficient mapping
and fallbacks exposed. Existing D=(J0,n) and full-pool sketches remain allocated.
Guide/routes: agent-notes/ADP/adaptive-centers.md. Math: experiments/
adaptive_centers/MATH.md (frozen). TeX logdet ratio corrected to centered
logdet(I+sum uuT/eps); scalar quadrature does not imply operator/recovery.

Luna completed120 paired development fits plus10 excluded smoke fits, all ok.
4new report-inspired AR1Gaussian cases,n500,d30/40/50,m2,J500,Nphi10,Nloc20,
local init,default LSMR,hmin1,a sqrt2,select last,dev92800..2,threads1.
All h_min6/7steps; **zero converged inner certificates**. No global gate passes;
selection=null,heldout92900..4 untouched. These results are conditional on the
frozen schedule/solver budget and are not a replication of TeX raw experiments.

Median paired floor25 EDRdiff0,time.994baseline; additive30 medianEDR
.4190->.2486,not general. Floor50 EDR+.00150,recovery2/12 vs1/12.
P1/floor25 time1.682; P2/P1 time1.179,no EDR change.
P3 LogDet+coverage J60/80/100,EDR+.03968,time2.487,RSS1.191baseline;
vs equal-J random EDR-.03773,time1.338. No fullfit quality/speed promotion.
Separate24-worker localization benchmark:n10000,d100,J500,h.4,common D,
P1/baseline time.493weights/.414alpha+weights,exact outputs,n500 slower.
Timing includes geometry+CSR output; D separately timed/retained. No inference
of end-to-end speed or native streamed-baseline RSS saving.

Same-state divergent-pair replay: identical support/weights/alpha,I/U<1.4e-15;
two raw initial/final hashes reproduced,finalEDRdiff-.01036 remains. Specific
solver branching cause unproven. Eta singleton ratio unstable,unused by solver.
After timings only seed=None compatibility regression fixed, integer streams
unchanged. Original source snapshot/manifests immutable; explicit AST-verified
overlay and current-source ZIP in verification/. Original harness frozen check
intentionally rejects patched tree; reproduce snapshot or use lead_audit.py.

Final evidence:27feature tests included in86 shared passes,6manifold deselected;
Ruff check/format clean,Pyright0errors/0warnings. Existing3catalog KeyErrors
block fullsuite collection;3manifold reference failures m2/local_quadratic
requiresm1 remain. Logs and independent audits in experiments/adaptive_centers/
verification/,reports/raw/plots under same root and supplement/.
No pending authorized work. New research requires fresh protocol; heldout
seeds remain available. No commit/push performed.
