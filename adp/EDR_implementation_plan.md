# EDR Average Derivative Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** реализовать все функции из `adp/edr.py` для average derivative так, чтобы они напрямую собирались в готовый алгоритм из `adp/edr_adp_single_index.py`.

**Architecture:** реализация остается внутри класса `EDR`, а `EDR_ADP_single_index` остается верхнеуровневым orchestrator. Тяжелые вычисления строятся вокруг `numpy`, `scipy.sparse.csr_matrix`, `scipy.spatial.cKDTree` и, при необходимости, `scipy.linalg` / `scipy.sparse.linalg`.

**Tech Stack:** `numpy`, `scipy.sparse`, `scipy.spatial.cKDTree`, `scipy.linalg`, `scipy.sparse.linalg`, `pytest`.

---

## Files

- Modify: `adp/edr.py`
- Test: `tests/test_edr_average_derivative.py`
- Reference: `manifold_new.tex`, section `Average derivative procedure`
- Keep unchanged unless needed by tests: `adp/edr_adp_single_index.py`, `adp/data.py`, `adp/characteristics.py`

## Общий стиль реализации

Каждая функция должна начинаться с короткого комментария или docstring-блока:

```python
# Вход:
# - ...
#
# Выход:
# - ...
#
# Что делает:
# - ...
#
# Реализация:
# - ...
```

Код внутри функции делится на смысловые блоки пустыми строками. Перед каждым циклом обязательно должен быть комментарий, который объясняет, что именно этот цикл считает. Имена переменных должны быть понятными: `neighbor_indices`, `neighbor_weights`, `centered_neighbors`, `local_mean`, `direction_vectors`, `local_slopes`, а не `idx`, `w`, `tmp`, `arr`, `res`. Короткие математические имена допустимы только там, где они являются формульными объектами: `X`, `Y`, `I`, `U`, `h`, `rho`.

Пример стиля внутри тяжелой функции:

```python
# Считаем статистики отдельно для каждого центра, потому что CSR хранит соседей построчно.
for center_index in range(number_of_centers):
    ...
```

После валидации входов, подготовки данных, основного вычисления и возврата результата должны быть пустые строки. Не смешивать несколько смысловых действий в одну плотную строку. Не писать длинные цепочки выражений, если их можно разбить на понятные промежуточные переменные.

## Общие структуры данных

| Object | Shape / Type | Meaning |
|---|---:|---|
| `self.X` | `(n_samples, n_features)` | обучающие точки |
| `self.Y` | `(n_samples,)` | ответы |
| `self.beta` | `(n_features,)` | истинная beta, если доступна |
| `self.centers` | `(n_centers, n_features)` | точки `x_j` |
| `self.tree` | `scipy.spatial.cKDTree` | быстрый поиск локальных соседей |
| `self.rng` | `np.random.Generator` | воспроизводимая случайность |
| `weights` | CSR `(n_centers, n_samples)` | локальные веса `w_ij` |
| `directions` | `(n_centers, n_directions, n_features)` | направления `phi` |
| `I` | `(n_centers, n_directions)` | статистика `I_{j,phi}` |
| `U` | `(n_centers, n_directions, n_features)` | статистика `U_{j,phi}` |
| `local_slopes` | `(n_centers,)` | локальные коэффициенты `l_j` |

## Task 1: `__init__`

**Purpose:** подготовить состояние объекта и параметры алгоритма.

**Implementation steps:**

- [ ] Добавить импорты `numpy as np`, `scipy.sparse`, `scipy.spatial.cKDTree`, `scipy.linalg`, `scipy.sparse.linalg`.
- [ ] Расширить `__init__` параметрами алгоритма: `n_centers`, `n_directions`, `min_neighbors`, `h_decay`, `h_min`, `rho_min`, `ridge`, `tol`, `max_iter`, `random_starts`, `dtype`, `random_state`.
- [ ] Создать `self.Data`, `self.Characteristics`, `self.rng`.
- [ ] Явно объявить поля состояния: `self.X`, `self.Y`, `self.beta`, `self.centers`, `self.tree`, `self.mean`, `self.scale`, `self.current_outer_step`, `self.rho_current`, `self.eps`.

**Code style requirements:**

- В начале конструктора описать, какие параметры принимает алгоритм и какие поля создает.
- Разделить блоки: параметры алгоритма, вспомогательные объекты, данные, состояние итераций.
- Не использовать неочевидные сокращения кроме `X`, `Y`, `h`, `rho`.

**Tests:**

- [ ] `EDR(random_state=1)` создает `np.random.Generator`.
- [ ] Все численные параметры доступны как атрибуты.
- [ ] `Data` и `Characteristics` инициализированы.

**Verification command:**

```bash
python -m pytest tests/test_edr_average_derivative.py::test_init_sets_algorithm_state -q
```

## Task 2: `Mean_Calculate`

**Purpose:** посчитать центрирование и масштабирование признаков.

**Implementation steps:**

- [ ] Использовать `self.X` как входную матрицу признаков.
- [ ] Посчитать `feature_mean = np.mean(self.X, axis=0)`.
- [ ] Посчитать `feature_scale = np.std(self.X, axis=0)`.
- [ ] Заменить слишком маленькие значения scale на `1.0`, чтобы не делить на ноль.
- [ ] Вернуть `feature_mean, feature_scale` и сохранить их в `self.mean`, `self.scale`.

**Code style requirements:**

- В начале функции описать: вход `self.X`, выход `(feature_mean, feature_scale)`.
- Разделить блоки: проверка данных, расчет mean, расчет scale, сохранение результата.
- Использовать имена `feature_mean`, `feature_scale`, `small_scale_mask`.

**Tests:**

- [ ] Для матрицы с разными колонками функция возвращает правильные mean и std.
- [ ] Для константной колонки scale становится `1.0`.
- [ ] Функция сохраняет `self.mean` и `self.scale`.

**Verification command:**

```bash
python -m pytest tests/test_edr_average_derivative.py::test_mean_calculate_returns_stable_center_and_scale -q
```

## Task 3: `Kernel_Calculate`

**Purpose:** реализовать компактное ядро для sparse-весов.

**Implementation steps:**

- [ ] Трактовать `distance` как нормированное квадратное расстояние.
- [ ] Использовать quartic kernel: `(1 - distance) ** 2` для `distance < 1`, иначе `0`.
- [ ] Реализация должна работать для скаляра и массива.
- [ ] Не брать `sqrt`, потому что формулы используют квадрат нормы.

**Code style requirements:**

- В начале описать вход `distance` и выход kernel values той же формы.
- Разделить блоки: приведение к массиву, маска внутри носителя, расчет значений.
- Переменные назвать `normalized_distance`, `inside_kernel_support`, `kernel_values`.

**Tests:**

- [ ] `distance=0` дает `1`.
- [ ] `distance=1` и `distance>1` дают `0`.
- [ ] Векторный вход возвращает массив той же формы.

**Verification command:**

```bash
python -m pytest tests/test_edr_average_derivative.py::test_kernel_calculate_is_compact_and_vectorized -q
```

## Task 4: `Generate_Data`

**Purpose:** собрать данные из `Data` в состояние `EDR`.

**Implementation steps:**

- [ ] Вызвать `self.Data.Generate_X`, `Generate_beta`, `Generate_func`, `Generate_Noise`, `Generate_Y`, `Generate_Centers` в согласованном порядке.
- [ ] Привести `X`, `Y`, `beta`, `centers` к `np.asarray(..., dtype=self.dtype)`.
- [ ] Сохранить исходные массивы в `self.X`, `self.Y`, `self.beta`, `self.centers`.
- [ ] Вызвать `Mean_Calculate`, стандартизовать `self.X` и согласованно стандартизовать `self.centers`.
- [ ] Построить `self.tree = cKDTree(self.X)`.

**Code style requirements:**

- В начале описать, что функция заполняет состояние объекта и ничего тяжелого не оптимизирует.
- Разделить блоки: генерация, приведение типов, стандартизация, построение дерева.
- Если нужен цикл для выбора центров из `X`, перед ним написать комментарий, что цикл/выбор гарантирует непустые локальные окрестности.

**Tests:**

- [ ] После вызова есть `self.X`, `self.Y`, `self.centers`, `self.tree`.
- [ ] Формы согласованы: `X.shape[0] == Y.shape[0]`, `X.shape[1] == centers.shape[1]`.
- [ ] `self.tree.query(self.centers[:1])` работает.

**Verification command:**

```bash
python -m pytest tests/test_edr_average_derivative.py::test_generate_data_populates_arrays_and_tree -q
```

## Task 5: `Beta_Normalize`

**Purpose:** нормировать beta и стабилизировать знак.

**Implementation steps:**

- [ ] Привести `beta` к одномерному массиву `float`.
- [ ] Посчитать `beta_norm = np.linalg.norm(beta)`.
- [ ] Если норма меньше `self.eps`, заменить beta на случайное направление из `self.rng`.
- [ ] Вернуть `normalized_beta = beta / beta_norm`.
- [ ] Если доступна предыдущая beta, выровнять знак по положительному скалярному произведению.

**Code style requirements:**

- В начале описать вход beta и выход beta единичной нормы.
- Разделить блоки: приведение формы, обработка нулевой нормы, нормировка, выравнивание знака.
- Использовать имена `beta_vector`, `beta_norm`, `normalized_beta`.

**Tests:**

- [ ] Ненулевой вектор возвращается с нормой `1`.
- [ ] Нулевой вектор не приводит к `nan`.
- [ ] При отрицательном скалярном произведении с предыдущей beta знак меняется.

**Verification command:**

```bash
python -m pytest tests/test_edr_average_derivative.py::test_beta_normalize_returns_unit_vector_without_nan -q
```

## Task 6: `Generate_Direction`

**Purpose:** сгенерировать равномерные направления на сфере для step 0.

**Implementation steps:**

- [ ] Взять `number_of_centers = self.centers.shape[0]`, `number_of_features = self.X.shape[1]`.
- [ ] Сгенерировать `direction_vectors = self.rng.normal(size=(number_of_centers, self.n_directions, number_of_features))`.
- [ ] Посчитать нормы по последней оси.
- [ ] Нормировать направления на единичную длину.
- [ ] Для слишком маленьких норм пересемплировать или заменить на первый базисный вектор.

**Code style requirements:**

- В начале описать выход `(n_centers, n_directions, n_features)`.
- Разделить блоки: размеры, случайная генерация, расчет норм, fallback, нормировка.
- Если есть цикл/fallback по плохим направлениям, перед ним написать, что он заменяет численно вырожденные направления.

**Tests:**

- [ ] Форма результата равна `(m, q, d)`.
- [ ] Все направления имеют норму `1` с разумной точностью.
- [ ] Два объекта с одинаковым `random_state` дают одинаковые направления.

**Verification command:**

```bash
python -m pytest tests/test_edr_average_derivative.py::test_generate_direction_returns_unit_sphere_vectors -q
```

## Task 7: `Generate_Anisotropic_Direction`

**Purpose:** сгенерировать направления из анизотропного распределения для step k.

**Implementation steps:**

- [ ] Нормировать `beta_previous` через `Beta_Normalize`.
- [ ] Сгенерировать isotropic noise `standard_noise` формы `(m, q, d)`.
- [ ] Сгенерировать projected noise `beta_noise` формы `(m, q, 1)`.
- [ ] Посчитать `direction_vectors = rho_k * standard_noise + beta_noise * beta_previous`.
- [ ] Нормировать результат по последней оси.

**Code style requirements:**

- В начале описать, что `h_k` входит в теорию, но не влияет после нормировки направлений.
- Разделить блоки: нормировка beta, генерация шумов, сбор анизотропного вектора, нормировка.
- Использовать понятные имена `standard_noise`, `beta_direction_noise`, `anisotropic_vectors`.

**Tests:**

- [ ] Форма результата равна `(m, q, d)`.
- [ ] Все направления нормированы.
- [ ] При маленьком `rho_k` проекции на `beta_previous` в среднем больше, чем при `rho_k=1`.

**Verification command:**

```bash
python -m pytest tests/test_edr_average_derivative.py::test_generate_anisotropic_direction_aligns_with_beta -q
```

## Task 8: `Weight_Calculate`

**Purpose:** построить sparse-веса `w_ij` для step 0 и step k.

**Implementation steps:**

- [ ] Проверить, что `h > 0`, `self.X` и `self.centers` заполнены.
- [ ] Для step 0 использовать радиусный поиск `self.tree.query_ball_point(center, r=h)`.
- [ ] Для каждого центра взять соседей, посчитать `centered_neighbors = self.X[neighbor_indices] - center`.
- [ ] Для isotropic distance считать `squared_distance / h**2`.
- [ ] Для anisotropic distance считать `(rho**2 * squared_distance + projection_on_beta**2) / h**2`.
- [ ] Применить `Kernel_Calculate`, собрать `data`, `indices`, `indptr` и вернуть CSR.
- [ ] Нормировать строки CSR на сумму один для дальнейшего расчета `local_mean`.

**Code style requirements:**

- В начале описать входы `h`, `rho`, `beta` и выход CSR-матрицу.
- Разделить блоки: проверки, поиск соседей, расчет расстояний, расчет kernel, сбор CSR, нормировка.
- Перед циклом по центрам написать комментарий: цикл собирает локальные ненулевые веса для каждой строки CSR.
- Не использовать имена `i`, `j` без необходимости; лучше `center_index`, `neighbor_indices`, `neighbor_weights`.

**Tests:**

- [ ] Возвращается `scipy.sparse.csr_matrix`.
- [ ] Shape равен `(n_centers, n_samples)`.
- [ ] Все ненулевые веса положительны.
- [ ] Суммы непустых строк равны `1`.
- [ ] При `rho` и `beta` результат отличается от isotropic-весов.

**Verification command:**

```bash
python -m pytest tests/test_edr_average_derivative.py::test_weight_calculate_returns_row_normalized_csr -q
```

## Task 9: `Local_Mean_Calculate`

**Purpose:** посчитать `Xbar_j = sum_i X_i w_ij`.

**Implementation steps:**

- [ ] Проверить, что `weights` имеет CSR-формат или привести через `.tocsr()`.
- [ ] Проверить shape `(n_centers, n_samples)`.
- [ ] Посчитать `local_mean = weights @ self.X`.
- [ ] Для строк с нулевой массой заменить `local_mean[j]` на `self.centers[j]`.
- [ ] Вернуть массив `(n_centers, n_features)`.

**Code style requirements:**

- В начале описать вход CSR-весов и выход локальных средних.
- Разделить блоки: приведение CSR, проверка форм, sparse matrix multiply, fallback.
- Перед fallback-циклом написать, что он обрабатывает центры без найденных соседей.

**Tests:**

- [ ] На ручной маленькой матрице весов результат совпадает с явным средним.
- [ ] Нулевая строка не создает `nan`.
- [ ] Shape результата `(m, d)`.

**Verification command:**

```bash
python -m pytest tests/test_edr_average_derivative.py::test_local_mean_calculate_matches_manual_weighted_mean -q
```

## Task 10: `H0_Calculate`

**Purpose:** выбрать начальный bandwidth `h_0` по условию средней массы.

**Implementation steps:**

- [ ] Через `cKDTree.query` найти расстояние до `min_neighbors`-го соседа для каждого центра.
- [ ] Взять нижнюю и верхнюю границу поиска из квантилей этих расстояний.
- [ ] Сделать bisection по `h`.
- [ ] На каждом шаге считать ненормированную массу ядра через локальный радиусный поиск.
- [ ] Вернуть минимальный `h`, где средняя масса `>= min_neighbors`.
- [ ] Сохранить результат через `self.Characteristics.H0_Save(h_0)`.

**Code style requirements:**

- В начале описать вход через состояние `self.X`, `self.centers`, `self.tree` и выход `h_0`.
- Разделить блоки: kNN-границы, bisection, сохранение диагностики.
- Перед циклом bisection написать, что он ищет минимальный bandwidth с достаточной средней массой.
- Внутреннюю функцию для массы назвать понятно: `_Average_Kernel_Mass_Calculate` или локально `calculate_average_kernel_mass`.

**Tests:**

- [ ] Возвращаемое `h_0` положительно.
- [ ] Средняя масса при `h_0` не меньше `min_neighbors`.
- [ ] При немного меньшем `h` масса не больше или близка к порогу.

**Verification command:**

```bash
python -m pytest tests/test_edr_average_derivative.py::test_h0_calculate_satisfies_minimum_kernel_mass -q
```

## Task 11: `H_Update`

**Purpose:** уменьшить bandwidth по правилу `h_k = h_{k-1} / a`.

**Implementation steps:**

- [ ] Проверить `h_previous > 0`.
- [ ] Проверить `self.h_decay > 1`.
- [ ] Посчитать `h_k = h_previous / self.h_decay`.
- [ ] Сохранить `h_k` через `self.Characteristics.H_k_Save(h_k)`.
- [ ] Вернуть `h_k`.

**Code style requirements:**

- В начале описать вход `h_previous` и выход `h_k`.
- Разделить блоки: проверки, расчет, диагностика, возврат.
- Переменные назвать `previous_bandwidth`, `updated_bandwidth`.

**Tests:**

- [ ] При `h_decay=2` значение делится на два.
- [ ] При неположительном `h_previous` возникает понятная ошибка.
- [ ] Вызов диагностики происходит один раз.

**Verification command:**

```bash
python -m pytest tests/test_edr_average_derivative.py::test_h_update_decreases_bandwidth_and_saves_characteristic -q
```

## Task 12: `Step_k_Condition`

**Purpose:** решить, запускать ли следующий adaptive step.

**Implementation steps:**

- [ ] Использовать `self.current_outer_step` и `self.max_outer_steps` или аналогичный параметр.
- [ ] Проверить, что следующий bandwidth не уйдет ниже `h_min / h_decay` согласно текущей схеме `run_step_k`.
- [ ] Вернуть `True`, если следующий шаг допустим.
- [ ] Не менять веса, статистики и beta внутри этой функции.
- [ ] Увеличение счетчика делать либо здесь явно, либо в `run_step_k`, но выбрать один подход и описать его в комментарии.

**Code style requirements:**

- В начале описать вход `h_k` и выход boolean.
- Разделить блоки: проверка лимита шагов, проверка bandwidth, возврат решения.
- Использовать имена `has_remaining_steps`, `has_large_enough_bandwidth`.

**Tests:**

- [ ] Возвращает `False`, когда достигнут лимит outer-шагов.
- [ ] Возвращает `False`, когда bandwidth ниже порога.
- [ ] Возвращает `True`, когда оба условия выполнены.

**Verification command:**

```bash
python -m pytest tests/test_edr_average_derivative.py::test_step_k_condition_combines_step_limit_and_bandwidth_limit -q
```

## Task 13: `Rho_Calculate`

**Purpose:** выбрать анизотропию `rho_k` по условию средней массы.

**Implementation steps:**

- [ ] Нормировать `beta_previous`.
- [ ] Определить функцию массы для фиксированного `rho`.
- [ ] Искать максимальный `rho` из `[rho_min, 1]`, при котором средняя масса `>= min_neighbors`.
- [ ] Использовать bisection, потому что масса монотонно уменьшается при росте `rho`.
- [ ] Сохранить результат в `self.rho_current` и `self.Characteristics.Rho_k_Save(rho_k)`.

**Code style requirements:**

- В начале описать входы `beta_previous`, `h_k` и выход `rho_k`.
- Разделить блоки: нормировка beta, функция массы, bisection, сохранение.
- Перед bisection-циклом написать, что он ищет максимальную допустимую `rho`, то есть минимально необходимую анизотропию.
- Использовать имена `lower_rho`, `upper_rho`, `candidate_rho`, `average_kernel_mass`.

**Tests:**

- [ ] Возвращаемое значение лежит в `[rho_min, 1]`.
- [ ] Масса при найденном `rho` не меньше `min_neighbors`, если это достижимо.
- [ ] Для меньшего `h_k` найденная `rho` не больше, чем для большего `h_k`, на простой выборке.

**Verification command:**

```bash
python -m pytest tests/test_edr_average_derivative.py::test_rho_calculate_finds_mass_preserving_anisotropy -q
```

## Task 14: `Average_Derivative_Statistics_Calculate`

**Purpose:** за один проход посчитать `I` и `U`.

**Implementation steps:**

- [ ] Привести `weights` к CSR.
- [ ] Создать `I_statistics = np.zeros((n_centers, n_directions))`.
- [ ] Создать `U_statistics = np.zeros((n_centers, n_directions, n_features))`.
- [ ] Для каждого центра взять `start = weights.indptr[center_index]`, `end = weights.indptr[center_index + 1]`.
- [ ] Получить `neighbor_indices` и `neighbor_weights`.
- [ ] Посчитать `centered_neighbors = self.X[neighbor_indices] - local_mean[center_index]`.
- [ ] Посчитать `projection_values = centered_neighbors @ directions[center_index].T`.
- [ ] Посчитать `I_statistics[center_index] = (neighbor_weights * self.Y[neighbor_indices]) @ projection_values`.
- [ ] Посчитать `U_statistics[center_index] = projection_values.T @ (neighbor_weights[:, None] * centered_neighbors)`.
- [ ] Вернуть `I_statistics, U_statistics`.

**Code style requirements:**

- В начале описать входы `weights`, `local_mean`, `directions` и выход `(I, U)`.
- Разделить блоки: подготовка CSR, выделение выходных массивов, цикл по центрам, возврат.
- Перед циклом написать, что он обходит только ненулевые CSR-веса и не создает плотный `(m, n, d)` тензор.
- Внутри цикла использовать пустые строки между получением соседей, центрированием, проекциями и накоплением статистик.

**Tests:**

- [ ] На маленьком ручном примере `I` совпадает с прямой формулой.
- [ ] На маленьком ручном примере `U` совпадает с прямой формулой.
- [ ] Память не требует создания массива `(m, n, d)`; в тесте проверять косвенно через отсутствие такого shape в возвращаемых объектах.

**Verification command:**

```bash
python -m pytest tests/test_edr_average_derivative.py::test_average_derivative_statistics_match_manual_formula -q
```

## Task 15: `I_Calculate`

**Purpose:** вернуть только статистику `I`.

**Implementation steps:**

- [ ] Вызвать `Average_Derivative_Statistics_Calculate`.
- [ ] Взять первый элемент результата.
- [ ] Вернуть `I_statistics`.
- [ ] Позже, если потребуется оптимизация, добавить кеширование последней пары `(I, U)`.

**Code style requirements:**

- В начале описать входы и выход только `I`.
- Разделить блоки: расчет общей статистики, выбор `I`, возврат.
- Не дублировать тяжелый цикл, если `Average_Derivative_Statistics_Calculate` уже умеет считать обе статистики.

**Tests:**

- [ ] `I_Calculate` возвращает то же `I`, что первый элемент общей функции.
- [ ] Shape равен `(n_centers, n_directions)`.

**Verification command:**

```bash
python -m pytest tests/test_edr_average_derivative.py::test_i_calculate_reuses_joint_statistics -q
```

## Task 16: `U_Calculate`

**Purpose:** вернуть только статистику `U`.

**Implementation steps:**

- [ ] Вызвать `Average_Derivative_Statistics_Calculate`.
- [ ] Взять второй элемент результата.
- [ ] Вернуть `U_statistics`.
- [ ] Позже, если потребуется оптимизация, использовать тот же кеш, что и `I_Calculate`.

**Code style requirements:**

- В начале описать входы и выход только `U`.
- Разделить блоки: расчет общей статистики, выбор `U`, возврат.
- Не копировать ручной цикл по центрам, чтобы не разъехались две реализации одной формулы.

**Tests:**

- [ ] `U_Calculate` возвращает то же `U`, что второй элемент общей функции.
- [ ] Shape равен `(n_centers, n_directions, n_features)`.

**Verification command:**

```bash
python -m pytest tests/test_edr_average_derivative.py::test_u_calculate_reuses_joint_statistics -q
```

## Task 17: `L_Calculate`

**Purpose:** посчитать локальные наклоны `l_j` при фиксированной beta.

**Implementation steps:**

- [ ] Нормировать beta.
- [ ] Посчитать `projected_U = np.einsum("mqd,d->mq", U, beta)`.
- [ ] Посчитать числитель `slope_numerator = np.einsum("mq,mq->m", I, projected_U)`.
- [ ] Посчитать знаменатель `slope_denominator = np.einsum("mq,mq->m", projected_U, projected_U)`.
- [ ] Вернуть `local_slopes = slope_numerator / (slope_denominator + self.eps)`.
- [ ] Для слишком маленького знаменателя вернуть `0` в соответствующей позиции.

**Code style requirements:**

- В начале описать входы `I`, `U`, `beta` и выход `local_slopes`.
- Разделить блоки: нормировка beta, проекция `U beta`, числитель, знаменатель, стабилизированное деление.
- Использовать имена `projected_U`, `slope_numerator`, `slope_denominator`, `local_slopes`.

**Tests:**

- [ ] На ручном примере функция совпадает с формулой.
- [ ] При нулевом `U beta` результат конечный и без `nan`.
- [ ] Shape результата `(n_centers,)`.

**Verification command:**

```bash
python -m pytest tests/test_edr_average_derivative.py::test_l_calculate_matches_closed_form_slope -q
```

## Task 18: `Beta_Calculate`

**Purpose:** решить регуляризованную задачу для beta при фиксированных `l_j`.

**Implementation steps:**

- [ ] Подготовить `identity_matrix = np.eye(n_features, dtype=self.dtype)`.
- [ ] Для малого/среднего `n_features` собрать матрицу `normal_matrix`.
- [ ] Собрать правую часть `right_hand_side`.
- [ ] Если `beta_previous` передан, добавить ridge-якорь `ridge * beta_previous`.
- [ ] Решить систему через `scipy.linalg.solve(..., assume_a="pos")` или Cholesky.
- [ ] Если `n_features` большой, добавить отдельную ветку `LinearOperator + cg`.
- [ ] Вернуть ненормированную beta; нормировку делает `Alternating_Minimization`.

**Code style requirements:**

- В начале описать входы `I`, `U`, `l`, `beta_previous` и выход новую beta.
- Разделить блоки: размеры, сбор левой части, сбор правой части, решение системы.
- Перед циклом по центрам, если будет явная сборка, написать, что цикл накапливает нормальную систему по центрам.
- Использовать имена `normal_matrix`, `right_hand_side`, `local_slope`, `U_for_center`.

**Tests:**

- [ ] На маленькой задаче результат совпадает с `np.linalg.solve` для явно собранной системы.
- [ ] При вырожденной системе ridge делает решение конечным.
- [ ] Shape результата `(n_features,)`.

**Verification command:**

```bash
python -m pytest tests/test_edr_average_derivative.py::test_beta_calculate_solves_regularized_normal_system -q
```

## Task 19: `Objective_Calculate`

**Purpose:** посчитать целевую функцию alternating minimization.

**Implementation steps:**

- [ ] Посчитать `projected_U = np.einsum("mqd,d->mq", U, beta)`.
- [ ] Посчитать `predicted_I = l[:, None] * projected_U`.
- [ ] Посчитать `residual = I - predicted_I`.
- [ ] Вернуть `objective_value = float(np.sum(residual * residual))`.

**Code style requirements:**

- В начале описать входы `I`, `U`, `l`, `beta` и выход число.
- Разделить блоки: prediction, residual, sum of squares.
- Использовать имена `predicted_I`, `residual`, `objective_value`.

**Tests:**

- [ ] На ручном примере значение совпадает с прямым расчетом.
- [ ] Возвращаемый тип приводится к обычному `float`.
- [ ] Objective равен `0`, когда `I == l[:, None] * (U @ beta)`.

**Verification command:**

```bash
python -m pytest tests/test_edr_average_derivative.py::test_objective_calculate_matches_sum_of_squared_residuals -q
```

## Task 20: `Alternating_Minimization`

**Purpose:** собрать внутреннюю оптимизацию `l_j` и beta.

**Implementation steps:**

- [ ] Если `beta_initial` передан, нормировать его и использовать как старт.
- [ ] Если `beta_initial is None`, создать `random_starts` начальных направлений.
- [ ] Для каждого старта выполнить до `max_iter` итераций.
- [ ] На каждой итерации считать `local_slopes = L_Calculate(...)`.
- [ ] Затем считать `beta_candidate = Beta_Calculate(...)`.
- [ ] Нормировать `beta_candidate`.
- [ ] Считать objective раз в `objective_check_every` или на каждой итерации для первой реализации.
- [ ] Остановиться, если изменение beta или objective меньше `tol`.
- [ ] Вернуть beta с лучшим objective.

**Code style requirements:**

- В начале описать входы `I`, `U`, `beta_initial` и выход `beta_hat`.
- Разделить блоки: подготовка стартов, внешний цикл по стартам, внутренний цикл оптимизации, выбор лучшего решения.
- Перед циклом по random starts написать, что он защищает step 0 от плохой случайной инициализации.
- Перед inner-циклом написать, что он чередует расчет локальных наклонов и обновление beta.
- Использовать имена `initial_beta_candidates`, `current_beta`, `candidate_beta`, `best_beta`, `best_objective`.

**Tests:**

- [ ] На синтетическом согласованном `I`, `U` objective не растет после первой итерации.
- [ ] Возвращаемая beta имеет норму `1`.
- [ ] При переданном `beta_initial` первый старт совпадает с ним.
- [ ] При нескольких random starts выбирается решение с минимальным objective.

**Verification command:**

```bash
python -m pytest tests/test_edr_average_derivative.py::test_alternating_minimization_returns_unit_beta_and_decreases_objective -q
```

## Task 21: End-to-End Average Derivative Smoke

**Purpose:** проверить, что все функции собираются в `EDR_ADP_single_index`.

**Implementation steps:**

- [ ] Подготовить малый synthetic dataset: `n_samples=80`, `n_features=5`, `n_centers=12`, `n_directions=4`.
- [ ] Запустить `run_step_0(beta=true_beta)`.
- [ ] Проверить формы `beta_0`, `h_0`.
- [ ] Запустить один `run_step_k`.
- [ ] Проверить, что `rho_k`, `h_k`, `beta_k` конечные.
- [ ] Проверить, что `Characteristics` получает `h_0`, `h_k`, `rho_k`, cosine.

**Code style requirements:**

- Тест должен быть читаемым: отдельные блоки setup, step 0, step k, assertions.
- Не прятать весь smoke в одну длинную строку.
- Имена переменных в тесте: `true_beta`, `estimated_beta_step_0`, `estimated_beta_step_k`, `initial_bandwidth`, `updated_bandwidth`.

**Verification command:**

```bash
python -m pytest tests/test_edr_average_derivative.py::test_edr_single_index_runs_step_0_and_step_k_smoke -q
```

## Final Verification

После реализации всех задач выполнить:

```bash
python -m py_compile adp/ADP_main.py adp/data.py adp/characteristics.py adp/edr.py adp/edr_adp_single_index.py
python -m pytest tests/test_edr_average_derivative.py -q
git diff --check -- adp/edr.py tests/test_edr_average_derivative.py
```

Дополнительно для контроля памяти:

- не создавать массивы shape `(n_centers, n_samples, n_features)`;
- не хранить плотные веса, если можно использовать CSR;
- считать `I` и `U` вместе в `Average_Derivative_Statistics_Calculate`.

## Implementation Order

1. `__init__`, `Mean_Calculate`, `Generate_Data`, `Beta_Normalize`.
2. `Kernel_Calculate`, `Generate_Direction`, `Generate_Anisotropic_Direction`.
3. `Weight_Calculate`, `Local_Mean_Calculate`, `H0_Calculate`, `H_Update`, `Step_k_Condition`, `Rho_Calculate`.
4. `Average_Derivative_Statistics_Calculate`, `I_Calculate`, `U_Calculate`.
5. `L_Calculate`, `Beta_Calculate`, `Objective_Calculate`, `Alternating_Minimization`.
6. End-to-end smoke through `EDR_ADP_single_index`.
