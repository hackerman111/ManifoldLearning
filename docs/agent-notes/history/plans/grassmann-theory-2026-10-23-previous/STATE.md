# Current state

Task export-2026-10-23 DONE. User-requested autonomous multi-index/current
Grassmann/CLI/testing bundle is in export/. No commit/push.

Entry: export/README.md (install, single Grassmann CLI, paired SVD/Grassmann,
R MAVE/MEANMAVE, catalog CLI, own-data API). Independent pyproject.toml ships
worker.R and benchmark entrypoint in wheel. SOURCE_MANIFEST.json records123
files; all numerical ADP modules byte-identical to original working tree.
Only export/ADP/cli/main.py adds --solver grassman for multi, current20-step
default; root package/defaults unchanged. Export is a fixed source snapshot.

Evidence: export/VERIFY.md + verification/. Detached copy outside repository,
PYTHONPATH removed:139 passed,2 strict xfailed; default20/explicit2 CLI output
bit-identical to public model. Actual paired SVD/core full fits2/2 ok(seed0,
n800/d50/m2); R MEANMAVE/current Grassmann20/20 ok(10seeds,n80/d5/m2).
These are packaging smoke checks, not quality/speed promotion evidence.
Local clean wheel+sdist build, installed wheel import/resource/CLI smoke passed
from /tmp; system NumPy/SciPy shared in isolated venv. R tests ran; CUDA not run.
Root tests/test_export_bundle.py verifies completeness, hashes, byte equality
and detached CLI execution; passes. Targeted Ruff/diff checks clean.

Inherited limitations reproduced in original: optional multiv2 crashes on
mi-spokoini-m2-n; excluded from export along with coupled test_experiment.py.
Main multi catalog/Spokoiny series preserved. Root full pytest stops at3
collection errors(test_experiment,multi_center_diagnostic,multiv2_quality).
Manifold CLI m=2/default local_quadratic mismatch reproduces2 smoke failures;
those2 copied tests strict-xfailed with explicit reason; numeric code untouched.
Affected routing: agent-notes/ADP/cli-and-compat.md autonomous-export section.

Prior lowrank research plan/state preserved unchanged under
agent-notes/history/plans/export-2026-10-23-previous/. Its failed quality gate,
no promotion and untouched validation remain valid; earlier dirty work kept.
No required work remains for export; future updates require explicit recopying
and rerunning bundle isolation/hash tests.
