# ADP

`adp` - реализация Average Derivative Procedure для восстановления одномерного EDR-направления `beta` в одноиндексной модели

```text
Y = f(beta^T X) + eps.
```

В пакете есть две ветки алгоритма:

- `new` - вариант из `manifold_new.tex`, где локальные суммы строятся по случайным направлениям `phi`.
- `old` - вариант из `manifold_old.tex`, где используются полные локальные моменты без случайных проекций.

Сейчас вычисления выполняются только через `numpy`. Поле `target_dim` оставлено в конфигурации для будущего расширения на несколько EDR-направлений.

## Быстрый пример

```python
from adp import ADP, ADPConfig

model = ADP.create(
    "new",
    ADPConfig(
        n_centers=40,
        n_directions=8,
        outer_steps=3,
        inner_steps=8,
        show_progress=True,
        random_state=1,
    ),
)

data = model.generate_data(n=200, d=6, noise=0.03, link="linear")
result = model.fit(data.X, data.y, centers=data.centers, beta0=data.beta)

print(result.beta)
print(model.score(data.beta))
```

Для старого варианта достаточно заменить имя:

```python
model = ADP.create("old", ADPConfig(show_progress=False, random_state=1))
```

## Архитектура

```text
adp/
  __init__.py              публичные импорты пакета
  core.py                  совместимый вход для старых импортов
  benchmarks.py            совместимый вход для замеров
  common/                  общие типы, ядра, сохранение результата, графики
  backends/                численные суммы NumPy и поиск соседей через faiss/sklearn
  engine/                  общий ход ADP: данные, h, обучение, диагностика
  variants/                формулы вариантов new и old
  evaluation/              сценарии замеров, метрики, отчеты и готовые EDR-методы
tests/                     проверки публичного поведения
manifold_new.tex           описание варианта со случайными проекциями
manifold_old.tex           описание варианта с полными моментами
```

Главная точка входа - фабрика `ADP.create(...)`. Она возвращает один из двух классов:

- `RandomProjectionADP` для `new`;
- `FullMomentADP` для `old`.

Оба класса наследуют общую основу `ADPBase`. Общая часть отвечает за подготовку данных, выбор локальных масштабов, внешний цикл обучения, сохранение результата и диагностику. Отличия вариантов вынесены в четыре метода:

- `_compute_statistics(...)` - строит локальные суммы;
- `_solve_local_coefficients(...)` - решает локальные `c_j` и `l_j`;
- `_solve_beta(...)` - обновляет глобальное направление `beta`;
- `_objective(...)` - считает значение целевой функции.

Такой разрез позволяет добавлять новый вариант алгоритма без переписывания генерации данных, хода обучения и отчетов.

## Ход обучения

`fit(X, y, ...)` выполняет следующие шаги:

1. Приводит `X`, `y`, `centers`, `beta0` к ожидаемой форме.
2. Выбирает локальные центры, если они не переданы явно.
3. Оценивает начальный масштаб `h` по ближайшим соседям.
4. На каждом внешнем шаге обновляет локализацию:
   - `new` уменьшает `h`, подбирает `rho` и при необходимости пересэмплирует `phi`;
   - `old` обновляет продольный масштаб `b` и ортогональный масштаб `h`.
5. Считает локальные статистики `Ima/S/U` или `Ima/N/S/VP`.
6. Попеременно обновляет локальные коэффициенты и `beta`.
7. Возвращает `ADPResult` с `beta`, историей, прогрессом и служебными временами.

## Замеры и диагностика

Для быстрой проверки качества есть готовые сценарии:

```python
from adp.benchmarks import default_scenarios, run_benchmark_suite, benchmark_summary

frame = run_benchmark_suite(default_scenarios(quick=True), show_progress=False)
summary = benchmark_summary(frame)
```

`save_benchmark_report(...)` сохраняет таблицу и графики качества, времени и памяти. У обученной модели доступны:

- `model.score(beta_true)` - метрики восстановления направления;
- `model.summary()` - краткая сводка результата;
- `model.plot_history(save_path=...)` - график цели;
- `model.save_diagnostics(output_dir, beta_true=...)` - набор диагностических графиков.

## Проверка

Основная проверка пакета:

```bash
PYTHONPATH=. pytest
```

Если окружение запускает тесты через модуль Python, можно использовать:

```bash
python -m pytest
```
