# Метрики для полных локальных моментов

Реализация оставшихся геометрических гипотез из
`verification/anisotropic-geometry-20261026/latex/anisotropic_full_moments.pdf`:
`experiments/anisotropic_metrics.py`.

| arm | Геометрия |
|---|---|
| `euclidean` | A: обычный горизонтальный градиент |
| `left` | C: левосторонняя метрика с дополнением Шура средней ковариации |
| `two_sided` | D: дополнительно текущий Gram локальных коэффициентов |
| `schur` | E-SC: полная Schur Gauss–Newton форма + демпфирование |
| `varpro` | E-VP: точный якобиан профилированного остатка + демпфирование |
| `kernel` | C с двойственной метрикой фактического ADP-ядра |

Все arms используют общий градиентный алгоритм, полярную ретракцию,
Armijo и стандартную норму горизонтального градиента для остановки.
Локальные моменты, статистический ridge и профиль берутся из прежнего
`full_ridge_batched`. Это отдельный эксперимент; штатные настройки ADP
не меняются. Пример для D:

```python
from functools import partial

from ADP import ADP_Config
from ADP.engine.common.index_fit import fit_index
from ADP.engine.common.logger import IndexProfiler
from experiments.anisotropic_metrics import calculate_statistics, solve

profiler = IndexProfiler()
try:
    result = fit_index(
        X, y,
        config=ADP_Config(
            seed=42, N_lin=50, N_J=300, N_phi=12, N_loc=20,
            index_init="local", select_step="best",
        ),
        mode="multi", index_dim=3,
        statistics_function=calculate_statistics,
        solve=partial(solve, arm="two_sided"),
        profiler=profiler,
    )
finally:
    profiler.finish()
basis = result.index.T
```

На независимой серии d25/n5000, m3/5, additive/quadratic/sine, шум 0.3
метрика D прошла все заданные критерии: средняя EDR 0.43556→0.31502
относительно общего градиентного алгоритма A, медианное отношение времени
fit 1.012. Recovery осталось 20/60. С прежним L-BFGS (EDR 0.31940)
сделано описательное сравнение; статистическое превосходство не проверялось.

Для `kernel` нужны контекст текущей локализации и
`solver_metric_context=True`; готовый адаптер находится в
`experiments.anisotropic_geometry_study.fit`. Параметр `metric_tensor`
самого solver принимает матрицу T, а контекст `fit_index` содержит строку
типа тензора: преобразование выполняет `localization_tensor`.

Hook статистик возвращает `U=C,I=c` и предназначен только для данного
семейства full-moment solvers. Он не подходит обычному проекционному solver.

Проверки, замороженные входы, исходники, результаты Luna и независимый
пересчёт: [отчёт](../results/verification/anisotropic-metrics-20261026/REPORT.md),
[протокол](../results/verification/anisotropic-metrics-20261026/PROTOCOL.md),
[вывод](../results/verification/anisotropic-metrics-20261026/PROOF.md).

Анализ сохранённых результатов:

```bash
python -m experiments.anisotropic_geometry_study analyze selection
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 \
  python verification/anisotropic-metrics-20261026/audit_results.py selection
```

Команды `run` отказываются перезаписывать существующую фазу. Проверочные
seed открываются только для прошедших отбор и независимый аудит arms.
Уменьшение внутреннего профиля само по себе не означает улучшение EDR.
