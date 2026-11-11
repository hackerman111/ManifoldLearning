# Numerical verification

18 tests in tests/test_grassman.py pass (Python3.14.7, NumPy2.5.2,
SciPy1.18.1, float64, OPENBLAS_NUM_THREADS=1). Independent reference:
local augmented lstsq across tau=0/.2/1e-9 and m1/2/4; Schur across17 angles;
full eq:dg FD convergence and FD-coordinate augmented dense ridge solve;
Jacobian adjoint, gradient identity, monotonicity, gauge invariance,
deficient ranks/workspace fallback, noiseless perturbed recovery,
collinear ridge and uncertified rank boundary. Final focused shared suite:110 passed (focused_tests.txt).

Ruff0.16.9 format/check passes for solver/tests; Pyright1.1.414 zero errors
and warnings for solver using project standard settings, same pythonVersion
3.11, temporary derived config removing absent .venv and adding live system
site-packages. Tools installed in /tmp/grassman-tools (not project dependency).

Commands:
```
OPENBLAS_NUM_THREADS=1 python3 -m pytest tests/test_grassman.py -q
PYTHONPATH=/tmp/grassman-tools python3 -m ruff check ADP/solver/grassman.py tests/test_grassman.py
PYTHONPATH=/tmp/grassman-tools python3 -m ruff format --check ADP/solver/grassman.py tests/test_grassman.py
```
Full `python3 -m pytest -q` stopped during collection on4 unrelated existing
modules: test_experiment/test_multi_center_diagnostic/test_multiv2_quality
raise KeyError mi-spokoini-m2-n from experiments/multiv2.py; test_manifold_grid
requires absent threadpoolctl. No production catalog/dependency repair mixed
into this solver task. Broad suite with those4 modules excluded:
388 passed,28 GPU skips,19 failed. Rerunning those19 via --lf --tb=line
reproduced them in0.62s (broader_failures.txt). Causes: absent threadpoolctl
and scikit-learn; existing multiv2 catalog KeyError; existing manifold default
local_quadratic rejects index_dim2. No failed test invokes new Grassmann code;
old production defaults/sources are unchanged by this task.

Focused shared command (final110 passed in1.04s):
```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest tests/test_svd_solver.py tests/test_svd_gradient.py tests/test_svd_metric.py tests/test_multi_operator.py tests/test_multi_solver_certificate.py tests/test_reduced_gauss_newton.py tests/test_grassman.py -q
```

Pyright fallback to the installed interpreter is reproducible without changing
project configuration:
```
python3 - <<'CHECK'
from pathlib import Path
import json, sys, tomllib
cfg = tomllib.loads(Path('pyproject.toml').read_text())['tool']['pyright'].copy()
cfg.pop('venvPath', None)
cfg.pop('venv', None)
cfg['include'] = ['../home/papayka/AMI/Vlad_edr/ManifoldLearning/ADP/solver/grassman.py']
cfg['extraPaths'] = [str(Path('.').resolve())] + [p for p in sys.path if p.endswith('site-packages')]
Path('/tmp/grassman_pyright.json').write_text(json.dumps(cfg))
CHECK
PYTHONPATH=/tmp/grassman-tools python3 -m pyright -p /tmp/grassman_pyright.json ADP/solver/grassman.py
```

