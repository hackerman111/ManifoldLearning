# Current agent state

Active: svd-high-dimensional-recovery-2026-09-29; see PLAN.md for gates.
User requests improved high-dimensional SVD recovery and optimization.
Initial dirty tree contains only untracked SVD/, SVD_form.tex, problem.md;
preserve these user theory inputs. Existing production code is clean.

Theory identifies structural rank(B)<m loss; existing correction mode already
exists and must not be reinvented. Proposed full fixed-g ridge solve reuses
LSMR._linear_operator with explicit original-coordinate certificate, retaining
all m basis directions. Derivation and isolated prototype precede promotion.

Prior completed LSMR/SVD comparison: experiments/lsmr_vs_svd_fullfit_2026-09-29/
REPORT.md, 350/350 fits; archived plan in current experiment directory.
Next: derive/audit, dense reference, exploratory paired pilots, then freeze.
