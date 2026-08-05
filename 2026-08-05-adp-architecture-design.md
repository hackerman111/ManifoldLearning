# Архитектура пакета ADP

Дата: 2026-08-05
Статус: согласованный дизайн

## Цель

Перестроить пакет так, чтобы:

- изменение ADP-алгоритма не требовало правок генератора данных и solver-кода;
- новый численный метод добавлялся отдельной функцией;
- single-index и multi-index модели развивались независимо;
- пользовательский API оставался коротким и предсказуемым;
- верхнеуровневые импорты `ADP_Config` и `ADP_single_index` продолжали работать.

## Не входит в дизайн

- базовый класс для single-index и multi-index моделей;
- реестр solver-реализаций и выбор solver по строке;
- фабрика моделей;
- хранение обучающих данных внутри модели;
- совместимость со старыми внутренними импортами.

## Целевая структура

```text
ADP/
├── __init__.py
├── config.py
├── data.py
├── statistics.py
├── ADP_solver.py
├── solvers/
│   ├── __init__.py
│   └── lsmr.py
├── single_index/
│   ├── __init__.py
│   ├── model.py
│   ├── algorithm.py
│   └── data.py
└── multi_index/
    ├── __init__.py
    ├── model.py
    ├── algorithm.py
    └── data.py
```

`__init__.py` сохраняет короткий публичный API:

```python
from ADP import (
    ADP_Config,
    ADP_Data,
    ADP_solver,
    ADP_SolverResult,
    ADP_Statistics,
    ADP_single_index,
    ADP_multi_index,
)
```

## `ADP_Config`

`ADP_Config` — неизменяемая конфигурация ADP-алгоритма:

```python
@dataclass(frozen=True, slots=True)
class ADP_Config:
    N_loc: int = 10
    N_lin: int | None = None
    N_J: int | None = None
    N_phi: int | None = None
    outer_steps: int = 8
    lambda_penalty: float = 1.0
    local_ridge: float = 1e-8
    kernel: Callable = epanechnikov
    a: float = sqrt(2)
    h_min: float | None = None
    batch_size: int = 32
    seed: int = 42
    index_init: str = "local"
```

В конфигурацию не входят:

- `n` и `d`: определяются из `X.shape`;
- распределение `X`, шум и функция отклика: принадлежат генератору;
- `tol` и `max_iter`: принадлежат численному solver.

`N_lin`, `N_J`, `N_phi` и `h_min` разрешаются в фактические значения в начале
`fit`. Для этого создаётся новая конфигурация через `dataclasses.replace`;
исходный объект не изменяется. Фактическая конфигурация сохраняется как
`model.effective_config_`.

Валидация независимых от данных значений выполняется в `__post_init__`.
Ограничения, зависящие от `n` и `d`, проверяются при вызове `fit`.
Каждая модель отдельно проверяет, поддерживает ли её алгоритм выбранный
`index_init`.

## Данные

`ADP_Data` — контейнер, а не генератор:

```python
@dataclass(frozen=True, slots=True)
class ADP_Data:
    X: np.ndarray
    y: np.ndarray
    true_index: np.ndarray
```

Внешняя форма матрицы признаков едина во всём пакете:

```text
X.shape == (n, d)
y.shape == (n,)
```

Для single-index `true_index.shape == (d,)`, для multi-index —
`true_index.shape == (d, m)`.

Генераторы находятся рядом с соответствующей моделью:

```python
from ADP.single_index.data import generate_data

data = generate_data(
    n=1000,
    d=20,
    f=np.sin,
    noise_std=0.2,
    seed=42,
)
```

Генератор создаёт локальный `numpy.random.Generator` и за один вызов формирует
`X`, `y` и истинный индекс. Модель принимает только `X` и `y` и не использует
`ADP_Data` внутри обучения.

## Статистики

Общий контракт статистик задаётся контейнером:

```python
@dataclass(frozen=True, slots=True)
class ADP_Statistics:
    I: np.ndarray
    U: np.ndarray
    mass: np.ndarray
    mean: np.ndarray
    n_eff: np.ndarray
    eta: np.ndarray
```

Чистая функция:

```python
def calculate_statistics(
    X,
    y,
    weights,
    directions,
    *,
    batch_size,
) -> ADP_Statistics:
    ...
```

Она не принимает модель и не изменяет её поля. Генерация весов остаётся в
алгоритме конкретной модели, поскольку геометрия single-index и multi-index
различается.

## `ADP_solver`

`ADP_solver` — настраиваемый адаптер функции численного метода:

```python
class ADP_solver:
    def __init__(self, method, **settings):
        ...

    def fit(self, statistics, initial_index, **problem_params):
        ...
```

Он:

- хранит вызываемый `method` и численные настройки;
- передаёт статистики, начальный индекс и параметры задачи реализации;
- требует результат типа `ADP_SolverResult`;
- проверяет конечность возвращённого индекса;
- не содержит LSMR-, VarPro- или Krylov-специфичной логики.

Параметры задачи и численного метода разделены:

```text
ADP_Config:
    lambda_penalty, local_ridge

ADP_solver.settings:
    tol, max_iter и параметры конкретного численного метода
```

Одинаковые ключи в `problem_params` и `settings` считаются ошибкой, чтобы
настройки не переопределялись неявно.

Результат solver:

```python
@dataclass(slots=True)
class ADP_SolverResult:
    index: np.ndarray
    coefficients: np.ndarray | None
    diagnostics: dict[str, object]
```

Конкретная реализация содержит только функцию `fit`:

```python
# ADP/solvers/lsmr.py

def fit(
    statistics: ADP_Statistics,
    initial_index: np.ndarray,
    *,
    lambda_penalty: float,
    local_ridge: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> ADP_SolverResult:
    ...
```

Новый численный метод добавляется новым модулем в `ADP/solvers/` и передаётся
в `ADP_solver`. Базовый класс, наследование и строковый реестр не нужны.

## `ADP_single_index`

Конструктор только принимает зависимости:

```python
class ADP_single_index:
    def __init__(
        self,
        config: ADP_Config | None = None,
        solver: ADP_solver | None = None,
    ):
        self.config = config or ADP_Config()
        self.solver = solver or ADP_solver(lsmr.fit)
```

В конструкторе не создаются данные, RNG, истинный `beta` или результат
обучения.

После успешного `fit` модель предоставляет:

```text
beta_                  shape (d,)
coefficients_
trace_
solver_diagnostics_
effective_config_
n_features_in_
```

`trace_` содержит ADP-уровневую запись каждой внешней итерации: локализацию,
изменение индекса и условие остановки. `solver_diagnostics_` — отдельный список
диагностик численного метода, по одному элементу на внешнюю итерацию.

Основной API:

```python
model.fit(X, y)              # возвращает self
model.transform(X)           # возвращает X @ beta_
model.score_direction(beta)  # метрика для синтетических экспериментов
```

`model.py` содержит порядок шагов и итоговое состояние. `algorithm.py`
содержит чистые функции инициализации, выбора ширины, вычисления `rho`, весов,
случайных направлений и обновления локализации.

Класс `T_k` удаляется. Его три значения передаются функциям явно: `h`, `rho`
и текущий индекс. Если состояние итерации действительно разрастётся, его
можно будет собрать в приватный dataclass без изменения публичного API.

## `ADP_multi_index`

`ADP_multi_index` имеет тот же стиль использования:

```python
model = ADP_multi_index(index_dim=3, config=config, solver=solver)
model.fit(X, y)
Z = model.transform(X)
```

`index_dim` — обязательный положительный аргумент конструктора и задаёт число
восстанавливаемых направлений. Он относится к структуре модели, а не к
`ADP_Config`.

После обучения:

```text
basis_                 shape (d, m)
coefficients_
trace_
solver_diagnostics_
effective_config_
n_features_in_
```

`transform(X)` возвращает `X @ basis_`, а `score_subspace(true_basis)`
оценивает восстановленное подпространство в синтетических экспериментах.

Multi-index не наследуется от single-index. Общими для них остаются
`ADP_Config`, `ADP_solver`, `ADP_Statistics` и валидация входных массивов.
Конкретный solver передаётся явно, пока для multi-index не появится один
проверенный метод по умолчанию.

## Поток `fit`

```text
X, y
  -> валидация и effective_config_
  -> инициализация индекса и локализации
  -> веса и случайные направления
  -> ADP_Statistics
  -> ADP_solver.fit
  -> ADP_SolverResult
  -> обновление индекса, локализации и trace
  -> присваивание обученных атрибутов
```

Все вычисления выполняются в локальных переменных. Обученные атрибуты
присваиваются только после успешного завершения, поэтому ошибка в новом
обучении не оставляет модель в частично обновлённом состоянии.

RNG создаётся в начале каждого `fit` из `config.seed`. Повторный запуск на
одинаковых данных и с одинаковой конфигурацией воспроизводим.

Модель по умолчанию не хранит `X` и `y`.

## Ошибки

- `ADP_Config` проверяет типы, диапазоны и вызываемость `kernel`.
- `fit` проверяет числовой тип, конечность и согласованность форм `X` и `y`.
- `ADP_solver` проверяет вызываемость метода, тип результата и конечность
  `result.index`.
- Модель проверяет форму индекса, поскольку только она знает ожидаемые
  `(d,)` или `(d, m)`.
- `transform` до обучения поднимает понятный `RuntimeError`.
- Исключения конкретного solver не скрываются общим `except`.

## Минимальная проверка

Архитектуру фиксируют небольшие проверки:

1. `ADP_Config` неизменяем и корректно разрешает зависящие от данных значения.
2. Генераторы воспроизводимы и возвращают согласованные формы.
3. `calculate_statistics` сохраняет текущую численную эквивалентность.
4. `ADP_solver` передаёт настройки и отвергает некорректный результат.
5. `ADP_single_index` работает с подменной solver-функцией.
6. Повторный `fit` воспроизводим, а неуспешный `fit` не оставляет частичное
   новое состояние.
7. `transform` возвращает ожидаемые формы single-index и multi-index.

## Порядок миграции

1. Ввести новые контейнеры и сохранить публичные экспорты через `__init__.py`.
2. Перенести рабочий LSMR в `solvers/lsmr.py` и подключить через `ADP_solver`.
3. Перевести `ADP_single_index` на `fit(X, y)` и форму `X == (n, d)`.
4. Вынести его вычислительные шаги в `single_index/algorithm.py`.
5. Удалить старое изменение модели из `ADP_statistic`, wildcard-импорты,
   копирование конфигурации и `T_k`.
6. После стабилизации single-index реализовать multi-index через независимые
   `model.py` и `algorithm.py`.
7. Удалить старые внутренние модули; совместимость сохранять только для
   согласованных верхнеуровневых импортов.
