# Current state
Active svd-joint-rank-r-2026-09-30; bounded acceptance gates in PLAN.md.
78 baseline SVD/gradient/metric tests pass. Existing dirty edits preserved.
Implement experiments/svd_joint_2026_09_30/prototype.py: augmented full-core
LS and horizontal steps on both spaces, unchanged fixed-g objective.
Matrix and correction retain distinct rank constraints. No metric extension
or default change. Callable through ADP_solver custom hook.
Next: implementation and independent dense-reference checks before selection.
Prior plan: experiments/svd_joint_2026_09_30/PREVIOUS_PLAN.md.
Previous evidence remains under experiments/svd_*_2026-09-30/.
