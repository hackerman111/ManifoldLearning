# Простые улучшения SVD: результат 2026-09-30

В рабочем `ADP/solver/SVD.py` реализованы две ограниченные по объёму идеи:

1. **NUMERICAL, опция `precondition_v=True`:** диагональное правое предобусловливание LSMR v-подзадач по `SVD/SVD_solver.tex` (`eq:Ha`, `eq:diagprecond`).
2. **EXACT, действует по умолчанию:** прямые forward/adjoint действия расширенного оператора без вложенных вызовов `LinearOperator`.

Общий функционал, ограничения `rank(B)`/`rank(B-P)`, float64, допуски, начальные направления, штраф и внешний ADP-алгоритм сохранены. Предобусловливание по умолчанию выключено; direct d<=128 и нулевой штраф используют прежние пути. Новых зависимостей нет. GPU, блоковые шаги и новые стратегии поиска отложены.

## Математика и память

Для `A=diag(repeat(sqrt(mass)*(g@a),p))*Uflat` вычисляется диагональ `D=diag(A.T@A)+lambda`, `S=diag(D^(-1/2))`. Замена `x=base+S*y` используется в **обоих** блоках `[A; sqrt(lambda)I]`. Правая часть `[target;sqrt(lambda)z]` сохраняет центр `z=(prior-low_rank).T@a`, включая ненулевой warm start. Скалярный damp после масштабирования не используется. При lambda=0 преобразование отключено, поскольку оно может изменить решение минимальной нормы.

Приёмка основана на исходной нормальной невязке, а не на оценке LSMR в преобразованных координатах. При необходимости применяется строгий немасштабированный LSMR. Нечисловые сертификаты и нормы явно отклоняются. Доказательство, независимый аудит и ограничения: [AUDIT.md](AUDIT.md).

Суммы квадратов столбцов E[j,s] кэшируются в пределах 16 MiB. При J*d*8 выше лимита применяется потоковое сокращение по центрам. Нет нового d*d оператора или полного U**2; выход диагонали — O(d). Кэш живёт только внутри одного solve с неизменным U, веса обновляются для каждого a.

## Протокол

Сравнивались baseline, fused и fused+Jacobi на одинаковых статистиках. Fixed-g: J=120,p=8,m=3,r=2,d=100/300/600, lambda=.3, inner_tol=1e-8, inner_maxiter=20, rank_tol=0; обычные столбцы и масштабы geomspace(.2,5). Два измерения времени после прогрева; память измерялась отдельным запуском. Один поток BLAS, float64. Параметры, версии, кодовые SHA256 и seeds находятся в manifest JSON.

Selection fixed-g seeds110..112; full fit seeds120..122. Untouched validation fixed-g seeds210..212; full fit seeds220..222. Полное обучение: n320,d50/150,m3,J40,p8,N_loc24,outer_steps3, seed=config data_seed+10000, штатная локальная инициализация и локализация. Оба rank-режима проверены отдельно.

Первоначальный full-fit fixture n220,d150 был некорректен: автоматический N_lin=2d=300 превышал n. Все 18 попыток baseline/fused/Jacobi отказали **до солвера**; они сохранены в initial `selection_runs.jsonl` и failed `selection_summary.json`. Гипотеза, кандидат и критерии не менялись. Исправление протокола использовало n320 и свежие full-fit selection seeds120..122. Валидные первоначальные d50 fits тоже сохранены, но не входят в новую selection. Старый план и точные версии benchmark сохранены в `initial_improvement_plan.md`, `benchmark_frozen.txt`, `benchmark_selection_fit.txt`.

## Проверка заранее заданных критериев

| Критерий | Selection | Validation | Граница |
|---|---:|---:|---:|
| Успешные строки | 144 | 144 | 144 |
| Max относительная ошибка сырой B | 1.19494e-06 | 5.21022e-07 | 2e-5 |
| Max расхождение проекторов | 1.33402e-06 | 2.75888e-06 | 2e-5 |
| Max изменение ошибки проектора полного fit | 1.44471e-09 | 2.26176e-08 | 2e-5 |
| Изменение числа внешних шагов | 0 | 0 | 0 |
| Медиана time ratio Jacobi/baseline на масштабированных данных | 0.231392 | 0.225485 | <=.8 |
| Медиана U-pass ratio на масштабированных данных | 0.239721 | 0.238095 | <=.8 |
| Медиана default time ratio на обычных данных | 0.935453 | 0.943049 | <=1.1 |

Обе стадии: все 144 строки успешны, все критерии выполнены. Оригинальный objective совпадает с прямой оценкой, не возрастает; effective_rank=2, исходные сертификаты конечны и <=1e-7. Допустимый рост default traced peak (10%+1 MiB) не превышен. Статусы/итерации и исходные normal residual сохранены отдельно; успешный запуск не означает глобального rank-r оптимума.

## Время на отложенных fixed-g данных

Каждая строка объединяет шесть пар (три seeds, два rank-режима). Отношения — медианы парных отношений, времена — медианы отдельных запусков. Значения d100 относятся к прямому решению, поэтому Jacobi там не применяется.

| d | Столбцы | Baseline, с | Fused, с | Jacobi, с | Fused/base | Jacobi/base | Итерации LSMR base / Jacobi |
|---:|---|---:|---:|---:|---:|---:|---:|
| 100 | обычные | 0.0143 | 0.0145 | 0.0145 | 1.010 | 1.006 | 0 / 0 |
| 100 | разные масштабы | 0.0132 | 0.0130 | 0.0131 | 1.000 | 1.001 | 0 / 0 |
| 300 | обычные | 0.2427 | 0.2161 | 0.2266 | 0.903 | 0.927 | 2096 / 2086 |
| 300 | разные масштабы | 1.1508 | 1.0424 | 0.2156 | 0.901 | 0.192 | 10374 / 2011 |
| 600 | обычные | 1.3096 | 1.2171 | 1.2282 | 0.943 | 0.946 | 7278 / 7257 |
| 600 | разные масштабы | 4.8464 | 4.6222 | 1.1306 | 0.931 | 0.225 | 28793 / 6867 |

Медианный выигрыш Jacobi на масштабированных данных: **4.43 раза**, U-проходы сокращены на **76.2%**. Для обычных данных default Fused сокращает время на **5.7%**. Это результаты заданных задач, не обещание такого ускорения на любой статистике U.

## Полное обучение

| d | Fused/base time ratio | Jacobi/base time ratio | Max изменение ошибки проектора |
|---:|---:|---:|---:|
| 50 | 0.988 | 0.964 | 0 |
| 150 | 0.864 | 0.880 | 2.26e-08 |

Качество полного обучения проверено на численное совпадение с прежним солвером. Это не эксперимент улучшения восстановления EDR и не доказательство сходимости внешнего ADP. Ограничение rank(B)<m и prior completion сохраняются.

## Память

Отдельные повторные измерения для fixed-g d600, разные масштабы, seed210:

| Режим | Метод | Traced peak, MiB | Размер кэша, MiB | Sampled RSS peak, MiB |
|---|---|---:|---:|---:|
| matrix | baseline | 1.377 | 0.000 | 89.11 |
| matrix | fused | 1.371 | 0.000 | 89.29 |
| matrix | jacobi | 1.886 | 0.549 | 89.29 |
| correction | baseline | 1.395 | 0.000 | 89.31 |
| correction | fused | 1.392 | 0.000 | 89.32 |
| correction | jacobi | 1.908 | 0.549 | 89.32 |

Tracemalloc исключает уже созданные входы и может не охватывать все native allocations. RSS опрашивается каждые 5 ms в общем процессе и зависит от allocator/cache history; он приведён как контекст, не как чистая разница методов. Ограничение дополнительной памяти установлено отдельно через размер кэша и тест принудительного потокового пути.

## Запуск и проверки

```python
from ADP import ADP_Config, ADP_multi_index, ADP_solver
from ADP.solver.SVD import solve as solve_svd

model = ADP_multi_index(
    3, ADP_Config(),
    ADP_solver(solve_svd, rank=2, precondition_v=True, low_rank_target="correction"),
)
```

59 тестов SVD, общего solver certificate и index model/API прошли. 18 из них — SVD, включая 9 новых: плотный augmented lstsq, сопряжение, разные массы/масштабы, почти зависимые/нулевые столбцы, lambda=0/1e-4/.3, ненулевой warm start, ограниченный кэш/stream, лимит итераций, недопустимый сертификат и тип опции. Ruff check/format проходят. Рабочий SVD.py побайтно совпадает с кандидатом, проверенным на selection/validation.

Pyright по штатному pyproject.toml не завершился за 180 секунд (exit 124); полная проверка типов остаётся неподтверждённой. Ограниченные проверки неизменённого baseline и прототипа также не завершились. Итоги и команды проверок сохранены в [VERIFICATION.json](VERIFICATION.json).

Эталон можно повторить из корня:

```sh
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_CACHE_DIR=/tmp/adp-uv-cache PYTHONPATH=. uv run --no-sync python experiments/svd_improvements_2026-09-30/check_reference.py
rtk proxy env OPENBLAS_NUM_THREADS=1 UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync pytest -q tests/test_svd_solver.py tests/test_multi_solver_certificate.py tests/test_index_models.py
```

Для повторения замеров сохраните исходные артефакты, скопировав каталог в новую папку. Скрипт сохраняет результаты рядом с собой:

```sh
rtk proxy cp -a experiments/svd_improvements_2026-09-30 /tmp/svd-repeat
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_CACHE_DIR=/tmp/adp-uv-cache PYTHONPATH=. uv run --no-sync python /tmp/svd-repeat/benchmark.py validation --scope fixed
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_CACHE_DIR=/tmp/adp-uv-cache PYTHONPATH=. uv run --no-sync python /tmp/svd-repeat/benchmark.py validation --scope fit
rtk proxy env UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python /tmp/svd-repeat/analyze.py validation
```
