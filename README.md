# Average Derivative Procedure

Основная точка входа сохраняет прежний вид:

```python
from adp import ADP, ADPConfig

model = ADP.create("new", ADPConfig(show_progress=False))
result = model.fit(X, y)
```

## Параллельное вычисление статистик NumPy

2 workers включаются явно через `statistics_workers=2`; по умолчанию
остаётся безопасный serial-режим с одним worker:

```python
config = ADPConfig(statistics_workers=2, show_progress=False)
model = ADP.create("new", config)
```

## Замена этапов алгоритма

Основные этапы ADP создаются через изолированный `StageRegistry`. Встроенные
имена можно посмотреть через `registry.available(category)` и выбрать при
создании модели:

```python
from adp import ADP, ADPConfig, StageRegistry

registry = StageRegistry.with_defaults()
model = ADP.create(
    "new",
    ADPConfig(show_progress=False),
    stages={
        "bandwidth_selector": "adaptive_mass",
        "statistics_builder": "random_projection",
        "beta_solver": "cg",
    },
    registry=registry,
)
```

Новый исследовательский solver можно передать напрямую:

```python
model = ADP.create(
    "new",
    config,
    stage_factories={
        "beta_solver": lambda context: ExperimentalBetaSolver(context.config),
    },
)
```

Доступные категории:

- `beta_initializer`;
- `center_selector`;
- `bandwidth_selector`;
- `direction_sampler`;
- `statistics_builder`;
- `local_solver`;
- `beta_solver`;
- `stop_rule`.

После `fit()` результат содержит выбранные реализации, накопленное время и
число вызовов каждого этапа:

```python
result.stage_names
result.stage_timings
result.stage_calls
```

Запускаемый пример честного сравнения matrix-free CG и плотного direct solver
на одинаковых данных, начальном `beta`, центрах и направлениях:

```bash
python examples/compare_adp_solvers.py --n 120 --d 8
```
