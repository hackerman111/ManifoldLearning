`ADP_single_index` и `ADP_multi_index` используют общий с CLI/экспериментами
цикл `ADP/engine/common/index_fit.py` и по умолчанию HPAO-LSMR.

```python
from ADP import ADP_Config, ADP_single_index, ADP_multi_index

config = ADP_Config(N_phi=10, outer_steps=3, seed=42)
single = ADP_single_index(config).fit(X, Y)
multi = ADP_multi_index(2, config).fit(X, Y)
```

Публичные формы сохранены: `single.beta_` — `(d,)`, `multi.basis_` и
`multi.beta_` — `(d,m)`. Внутри HPAO multi-index хранится строками `(m,d)`.
`fit` возвращает модель; `result_`, `trace_`, `profile_`, `coefficients_`
и multi `transform(X)` доступны после успешного обучения.
`progress` получает независимый снимок очередной строки trace.

Относительно прежних классов это **ESTIMATOR-переход**, запрошенный отдельно:

- `estimator="new"` использует нормированные I/U и внешнюю локальную массу;
  `estimator="legacy"` сохраняет ненормированные статистики, но также решается
  текущим HPAO, а не `legacy_lsmr`.
- Инициализация, `local-cv`/`pilot`, локализация, направления, исключение центров,
  их смещение и `redraw_directions` совпадают с CLI.
- Центры, инициализация и направления имеют отдельные SeedSequence-потоки.
  Старые оценки при том же seed не обязаны повторяться.
- `select_step="best"` выбирает шаг по ошибке локального прогноза S, как CLI;
  `last` возвращает последний. Истинный индекс для fit не нужен.
- `coefficients_` относятся к выбранному шагу и каноническому базису результата.
- `outer_steps` — нормальная причина остановки, а не исключение о превышении лимита.
- По умолчанию HPAO выполняет максимум 3 внутренних шага для single и 5 для multi,
  `tol=1e-6`, как CLI. `local_ridge` используется в локальной инициализации;
  local refit HPAO остаётся minimum-norm задачей без скрытого local ridge.

Другой актуальный solver подключается через сохранённый `ADP_solver`:

```python
from ADP import ADP_solver
from ADP.solver.HYBRID import solve

multi = ADP_multi_index(
    2,
    config=config,
    solver=ADP_solver(solve, max_steps=5, tol=1e-6),
).fit(X, Y)
```

Допустимы `ADP.solver.LSMR.solve`, `CG.solve`, multi `HYBRID.solve` и custom
функция с контрактом `method(index, U, I, *, mass, lambda_prox, **settings)`
и результатом `HPAOResult`. `mass` и `lambda_prox` задаёт модель; их нельзя
повторять в settings. Старый `ADP_solver.fit` для явного legacy-вызова сохранён,
но модели вызывают новый `fit_current`: старый callback
`method(statistics, initial_index, ...) -> ADP_SolverResult` надо перенести
на текущий контракт. Незаметной подмены решателя нет.

`gpu=True` включает CUDA-статистики при CPU-солвере (LSMR, CG или HYBRID).
`gpu=True, gpu_solver=True` сохраняет I/U на GPU и запускает там HPAO-LSMR.
Нужны CuPy и рабочая NVIDIA CUDA; при их отсутствии выдаётся явная ошибка.
Инициализация, направления, веса и поиск масштабов остаются CPU.
Подробности, установка и измерения: [index_gpu.md](index_gpu.md).
Прежняя GPU-ветка legacy helpers не используется моделями.
Отдельные legacy helpers не удалены. Sparse box/plateau helpers также остаются,
но модели теперь строят веса и статистики тем же путём, что CLI; это не заявка
на сохранение прежней скорости sparse-ветки или ускорение полного fit.

При фиксированных X/Y, конфигурации и solver settings класс и CLI возвращают
одинаковый индекс (с транспонированием для multi), спектр и выбранный шаг.
Это проверяется в `tests/test_index_models.py`, включая fixed directions,
смещение и исключение центров, варианты инициализации, CG/HYBRID,
нормировку mass, выбор coefficients и явные ошибки при пустых соседствах.
`docs/index_matrix_gpu_audit.md` описывает состояние до этой миграции.
