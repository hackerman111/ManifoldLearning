# Автономный комплект ADP / Grassmann

Папку можно скопировать отдельно от репозитория. Нужен Python >= 3.11.
Все команды ниже выполняются **из этой папки**: `cd export` либо `cd <копия>`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,bench]'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Альтернатива с uv: `uv sync --extra test --extra bench`, затем
`uv run --no-sync python ...`. Для самого solver достаточно `pip install -e .`;
`test` добавляет pytest, `bench` — зависимости отчётов/графиков и benchmark-ов.

## Один запуск multi-index

Текущий Grassmann (`relative_gn`, quadratic3, constant centers; default 20 шагов):

```bash
python -m ADP.cli --mode multi --solver grassman --n 240 --d 5 --index-dim 2 --data-seed 0 --seed 42
```

Короткая проверка запуска:

```bash
python -m ADP.cli --mode multi --solver grassman --n 48 --d 4 --index-dim 2 --N_J 8 --N_phi 3 --N_loc 6 --N_lin 10 --outer_steps 1 --solver-max-steps 2 --index_init random --h_min 1000000
```

Для single/multi доступен Grassmann (по умолчанию) и HPAO-LSMR:

```bash
python -m ADP.cli --mode multi --solver lsmr --n 240 --d 20 --index-dim 2 --data-seed 0 --seed 42
```

Single-index решается обоими солверами как rank-one задача с векторным API.
В manifold сохранён внутренний CG (`--mode manifold --solver cg`).
Список параметров: `python -m ADP.cli --help`.
`--solver-max-steps` и `--solver-tol` задают общий бюджет и точность.
Для LSMR дополнительно доступны `--theta`, `--trust-radius` и `--lsmr-maxiter`.
`lambda_penalty` у Grassmann задаёт демпфирование GN, а у LSMR — штраф
коррекции; значения численно одинаковы, но их роли в солверах различаются.
Парное сравнение для `m=3,4`: [LSMR vs Grassmann](../../../benchmarks/LSMR_GRASSMAN_COMPARISON.md).

## Сравнение с R MAVE / MEANMAVE

Перенесён `experiments.r_sdr`, включая `worker.R`. Для фактических запусков
требуются установленный R (`Rscript`) и пакет `MAVE`:

```bash
Rscript -e 'install.packages("MAVE", repos="https://cloud.r-project.org")'
python -m experiments.r_sdr --profile multi-rank --runs 10 --methods ADP-Grassmann-combined,MAVE,MAVE-package --output benchmarks/results/outputs/r-comparison --plots
```

`ADP-Grassmann-combined` выбирает нынешний solver/default20.
Исторический метод `ADP-Grassmann` сохраняет прежние явные настройки и бюджет5;
он не обозначает нынешний default. `MAVE-package` — пакетный MEANMAVE.
Парный короткий отчёт LSMR vs MAVE/MEANMAVE: [../../../benchmarks/LSMR_MAVE_COMPARISON.md](../../../benchmarks/LSMR_MAVE_COMPARISON.md).
`--dry-run` показывает точный бюджет без R и без обучения, `--cases` ограничивает
задачи, `--resume` продолжает сохранённый запуск. Минимум — 10 повторов.
Протокол и зависимости: [../../../benchmarks/experiments/r_sdr/README.md](../../../benchmarks/experiments/r_sdr/README.md).

## Существующие CLI и каталоги экспериментов

```bash
python -m ADP.cli.experiment --list
python -m experiments.multi --list
python -m experiments.multi --profile smoke --runs 1 --no-plots
python -m experiments.diagnostic --mode multi --profile smoke --dry-run
```

Каталоги single/multi используют Grassmann. HYBRID, LSMR, SVD и экспериментальные
солверы из `ADP/solver` удалены.

## Использование своих данных

```python
import numpy as np
from ADP import ADP_Config, ADP_multi_index, ADP_solver
from ADP.solver.grassman import solve

data = np.load("data.npz")  # X: (n,d), y: (n,)
model = ADP_multi_index(
    index_dim=2,
    config=ADP_Config(seed=42),
    solver=ADP_solver(solve),
).fit(data["X"], data["y"])
np.savez("fit.npz", basis=model.basis_, coordinates=model.transform(data["X"]))
print(model.result_.stop_reason, model.solver_diagnostics_)
```

Grassmann работает только на CPU. `gpu_solver=True` не поддерживается.
Опциональный `.[gpu]` и `gpu=True` доступны для расчёта статистик на CUDA
с передачей статистик CPU-солверу.

## Проверки и состав

```bash
python -m pytest -q
```

[VERIFY.md](../vlad_edr_m_20261026/source-root/VERIFY.md) и
`SOURCE_MANIFEST.json` относятся к первоначальному экспорту, до удаления
солверов. Архив `verification/` сохраняет исходный снимок.
