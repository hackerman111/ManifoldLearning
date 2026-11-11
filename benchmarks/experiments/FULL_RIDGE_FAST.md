# Быстрый профиль полных моментов с ridge

Экспериментальный вариант из `tex/result/full_moments_ridge.tex`:
`full_ridge_batched.py`. Он сохраняет полные локальные моменты, спектральную
маску, ridge, оптимизатор и выбор внешнего шага. Меняется способ вычислений.

Вместо хранения отбелённых матриц на каждой оценке достаточно вычислить
`T=C_hat @ B`, `v=c_hat @ B`, решить независимые системы
`(B.T @ T + tau*I)g=v` и использовать `f=e-v.T@g`. Градиент получается
из `2*(T@g-c_hat)g.T`. Константа `e` и нормировка сохраняются.
Статистики собираются за один проход по блокам весов; проекционные моменты
и граф в основной ветке не нужны. Пакетная спектральная подготовка остаётся.

При `tau <= 1e-8*max(lambda_max(C),tiny)` весь внутренний вызов возвращается
к исходному отбелённому профилю. Решение фиксируется до оптимизации; это
защита от потери точности около ядра ковариации. Спектральный порог не меняется.

Пример подключения для `n=5000,d=25,m=3`:

```python
from ADP import ADP_Config
from ADP.engine.common.index_fit import fit_index
from experiments.full_ridge_batched import calculate_statistics, solve

result = fit_index(
    X, y,
    config=ADP_Config(
        seed=42, N_lin=50, N_J=300, N_phi=12, N_loc=20,
        index_init="local", select_step="best",
    ),
    mode="multi", index_dim=3,
    solve=solve, statistics_function=calculate_statistics,
)
basis = result.index.T
```

`calculate_statistics` здесь возвращает `U=C,I=c`; этот hook предназначен
только для данного решателя. Его нельзя передавать обычному Grassmann-солверу.

Результаты, протоколы, входы, исходники, fitted bases и независимые аудиты:

- [Основной отчёт](../results/verification/full-ridge-batched-20261025/REPORT.md).
- [Протокол](../results/verification/full-ridge-batched-20261025/PROTOCOL.md).
- [Математическая проверка](../results/verification/full-ridge-batched-20261025/R1_AUDIT.md).
- [Первый отклонённый кандидат](../results/verification/full-ridge-fast-20261025/STATUS.md).

Сохранённые каталоги запусков защищены от перезаписи. На чистом checkout,
после копирования протокола и R1-аудита, команды из корня проекта:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.full_ridge_batched_study run selection
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 \
  python verification/full-ridge-batched-20261025/audit_results.py selection
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.full_ridge_batched_study run validation
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 \
  python verification/full-ridge-batched-20261025/audit_results.py validation
```

Полные моменты всё ещё требуют `O(J*d²)`, спектральная подготовка —
`O(J*d³)`. Потоковый сбор убирает дополнительное хранение `J*n` весов,
но внешний ADP сохраняет матрицу расстояний `J*n`. Ускорение не доказывает
лучшее восстановление истинного подпространства; штатные ADP/API не менялись.
