Ниже — только план для пяти выбранных изменений в старой multi-index версии. Логика такая: сначала ускоряем вычисление весов без изменения результата, затем используем уже найденный sparse support для более разумного выбора центров.

## 1. Расчёт анизотропного расстояния без построения (T_k)

### Было

Для пары ((x_j,X_i)) расстояние концептуально считается как

[
q_{ij}
======

|T_k(X_i-x_j)|^2,
]

где для старого multi-index варианта

[
T_k^2
=====

h_k^{-2}
\left[
\alpha_k^2(I-P_k^\top P_k) +
P_k^\top\Lambda_kP_k
\right].
]

Наивная реализация либо строит (T_k), либо каждый раз применяет (P_k) к новой (d)-мерной разности.

### Стало

Ни (T_k), ни (T_k^2) явно не строятся.

Для

[
\Delta_{ij}=X_i-x_j,
\qquad
u_{ij}=P_k\Delta_{ij}
]

использовать непосредственно

[
\boxed{
q_{ij}
======

\frac{
\alpha_k^2
\left(
|\Delta_{ij}|^2-|u_{ij}|^2
\right) +
u_{ij}^\top\Lambda_k u_{ij}
}{
h_k^2
}.
}
]

Это в точности та же квадратичная форма для используемого multi-index тензора.

### Что реализовать

Сделать отдельную функцию примерно с таким интерфейсом:

```python
exact_anisotropic_distance(
    center_idx,
    candidate_indices,
    Z_data,
    Z_centers,
    P,
    Lambda,
    alpha,
    h,
    X,
    centers,
)
```

Но внутри функция не должна применять (P) к каждой (\Delta_{ij}).

Для кандидатов `idx`:

```python
u = Z_data[idx] - Z_centers[j]
u_norm2 = row_sum(u * u)
proj_energy = row_sum((u @ Lambda) * u)

delta = X[idx] - centers[j]
D2 = row_sum(delta * delta)

q = (
    alpha**2 * (D2 - u_norm2)
    + proj_energy
) / h**2
```

Здесь (D2-|u|^2\ge0) в точной арифметике, поскольку строки (P_k) ортонормированы.

Из-за floating point после вычисления допустимо использовать

```python
orth2 = maximum(D2 - u_norm2, 0.0)
```

если отрицательное значение имеет масштаб только численной ошибки.

### Главное изменение

Функция `exact_anisotropic_distance` вызывается **не для всех (n) наблюдений**, а только для списка кандидатов, полученного screening stage.

---

# 2. Проекции (P_kX_i) считать один раз на outer step

### Было

Во внутренних циклах возникает многократное вычисление

[
P_k(X_i-x_j).
]

Если непосредственно делать это для каждой пары:

```python
for j in centers:
    for i in observations:
        u = P @ (X[i] - center[j])
```

то одинаковые произведения (P_kX_i) вычисляются огромное число раз.

### Стало

В самом начале каждой outer-итерации считать

[
\boxed{
Z_X=XP_k^\top
\in\mathbb R^{n\times m}.
}
]

Для всех потенциальных центров:

[
\boxed{
Z_C=CP_k^\top
\in\mathbb R^{J_0\times m}.
}
]

Тогда

[
P_k(X_i-x_j)=Z_{X,i}-Z_{C,j}.
]

Этот перенос непосредственно следует из анализа multi-index screening.

### Что реализовать

В начале outer step:

```python
Z_X = X @ P.T
```

Если центры являются строками (X), вообще не делать отдельный matmul:

```python
Z_C = Z_X[center_indices]
```

Если центры произвольные:

```python
Z_C = centers @ P.T
```

Дальше запрещаем weight code делать:

```python
P @ (X[i] - center[j])
```

Вместо этого везде:

```python
u = Z_X[i] - Z_C[j]
```

### Стоимость

Было потенциально:

[
O(J_0ndm).
]

Стало:

[
O(ndm+J_0dm)
]

на проекции и затем всего (O(m)) для получения (u_{ij}).

Это особенно существенно, когда

[
m\ll d.
]

Например,

[
m=2,\qquad d=1000.
]

---

# 3. Добавить exact projected screening

Это следующий слой после кеширования (P_kX).

Из формулы расстояния:

[
q_{ij}
======

\frac{
\alpha_k^2
(|\Delta_{ij}|^2-|u_{ij}|^2) +
u_{ij}^\top\Lambda_ku_{ij}
}{
h_k^2
}.
]

Первое слагаемое неотрицательно, поэтому

[
q_{ij}
\ge
\frac{
u_{ij}^\top\Lambda_ku_{ij}
}{h_k^2}.
]

Следовательно,

[
\boxed{
u_{ij}^\top\Lambda_ku_{ij}\ge h_k^2
\quad\Longrightarrow\quad
w_{ij}=0.
}
]

Это exact test: ненулевой вес таким отсечением потерять невозможно. Более того, среди тестов, использующих только (P_k\Delta), эта нижняя граница оптимальна.

## Было

```python
for j:
    for i:
        compute full q_ij
        w_ij = kernel(q_ij)
```

Даже пары, у которых вес заведомо нулевой, проходят полный расчёт.

## Стало

```python
u = Z_X[i] - Z_C[j]

lower = u.T @ Lambda @ u

if lower >= h**2:
    # точно w_ij = 0
    continue

q = exact_anisotropic_distance(...)
w = kernel(q)
```

### Реализация в векторном виде

Для одного центра и некоторого массива индексов:

```python
U = Z_X[idx] - Z_C[j]           # shape (c, m)

UL = U @ Lambda
lower = np.einsum("ij,ij->i", UL, U)

mask = lower < h**2
idx = idx[mask]
```

Только после этого читать соответствующие строки исходной (X).

Это важный момент:

> screening должен происходить **до** обращения к (d)-мерным `X[idx]`, насколько это возможно.

Тогда дешёвые отклонённые пары вообще не создают memory traffic по исходному (d)-мерному массиву.

---

# 4. Не перебирать все (n) наблюдений для каждого центра

Предыдущий screening всё ещё можно реализовать плохо:

```python
for j:
    U = Z_X - Z_C[j]
    lower = ...
```

Это уже намного дешевле исходного алгоритма, но всё равно рассматривает все

[
J_0n
]

пар.

Можно убрать и этот перебор.

## Переход к обычному radius search

Пусть

[
\Lambda_k^{1/2}
]

— PSD квадратный корень (\Lambda_k).

Определим

[
\boxed{
Y_i=\Lambda_k^{1/2}P_kX_i.
}
]

Тогда

[
u_{ij}^\top\Lambda_k u_{ij}
===========================

|Y_i-Y_j^c|^2.
]

Поэтому необходимое условие ненулевого веса

[
u_{ij}^\top\Lambda_ku_{ij}<h_k^2
]

эквивалентно

[
\boxed{
|Y_i-Y_j^c|<h_k.
}
]

То есть screening превращается в обычный exact radius query в размерности (m).

## Реализация

После вычисления `Z_X`:

```python
Lambda_sqrt = sqrt_psd(Lambda)

Y_X = Z_X @ Lambda_sqrt.T
Y_C = Z_C @ Lambda_sqrt.T
```

Если (\Lambda) диагональная:

```python
Y_X = Z_X * np.sqrt(lambda_diag)
Y_C = Z_C * np.sqrt(lambda_diag)
```

Затем один раз построить spatial index:

```python
tree = cKDTree(Y_X)
```

И для каждого потенциального центра:

```python
candidate_idx = tree.query_ball_point(
    Y_C[j],
    r=h,
)
```

После этого `candidate_idx` является **супермножеством точного support**:

[
\mathcal N_j
\subseteq
\mathcal C_j.
]

Теперь только для этих кандидатов:

1. получить (u_{ij});
2. при необходимости ещё раз проверить projected lower bound;
3. посчитать точный (q_{ij});
4. применить kernel;
5. оставить только (q_{ij}<1).

Итог:

```python
support_idx = candidate_idx[q < 1]
support_weights = kernel(q[q < 1])
```

## Почему screening остаётся exact

Spatial tree не заменяет исходное анизотропное расстояние.

Он лишь использует необходимое условие:

[
w_{ij}>0
\Rightarrow
|Y_i-Y_j^c|<h_k.
]

Последнее решение всё равно принимает полный (q_{ij}).

То есть:

```text
radius search
    ↓
candidate set
    ↓
full q
    ↓
true support
```

а не

```text
radius search
    ↓
weight
```

---

## Numerical detail

Чтобы из-за rounding не потерять точку около границы (h_k), radius query лучше делать немного консервативнее:

```python
radius = np.nextafter(h, np.inf)
```

или использовать очень маленький relative margin.

Затем окончательное условие определяется вычислением полного (q).

Таким образом screening может дать несколько лишних кандидатов, но не false negative.

---

# 5. Объединённый weight/search stage

После первых четырёх модификаций я бы сделал отдельный класс/модуль:

```text
ProjectedNeighborhoodEngine
```

На вход outer step он получает:

```text
X
potential_centers
P_k
Lambda_k
alpha_k
h_k
kernel
```

На этапе `prepare()`:

```text
1. Z_X = X @ P.T
2. Z_C = centers @ P.T
3. Lambda_sqrt = sqrt(Lambda)
4. Y_X = Z_X @ Lambda_sqrt.T
5. Y_C = Z_C @ Lambda_sqrt.T
6. build exact radius-search index(Y_X)
```

После этого метод

```python
get_support(j)
```

делает:

```text
1. radius query around Y_C[j]
          ↓
2. candidate indices C_j
          ↓
3. projected exact lower-bound check
          ↓
4. exact anisotropic q only for survivors
          ↓
5. kernel(q)
          ↓
6. exact sparse support N_j
```

Возвращать:

```python
Neighborhood(
    indices,            # exact N_j
    weights,            # exact nonzero weights
    projected_points,   # optional Z_X[N_j]
    mass,
)
```

---

# 6. Добавить support-aware center selection

Его надо расположить **между weight engine и расчётом дорогих статистик**.

То есть раньше:

```text
выбрать J центров
       ↓
посчитать weights
       ↓
посчитать I_j, U_j
```

После модификации:

```text
большой pool возможных центров
       ↓
дешево получить exact supports N_j
       ↓
support-aware selection
       ↓
оставить J_* центров
       ↓
только для них считать I_j, U_j
```

Именно в этом вычислительный смысл: использовать информацию из weight stage, чтобы не строить дорогие статистики для плохих центров.

---

## 6.1. Сначала сформировать pool потенциальных центров

Не надо сразу выбирать окончательные (J_*) центров.

Например, если раньше сразу выбиралось 200 центров:

```text
old:
200 random centers
→ statistics for 200
```

то теперь:

```text
new:
1000 cheap candidate centers
→ supports only
→ choose best 200
→ statistics only for 200
```

Pool может быть:

- текущим множеством возможных (x_j);
- подвыборкой строк (X);
- тем же механизмом генерации центров, который уже использует старая версия.

Эта модификация не требует менять само определение центра.

---

# 7. Для каждого потенциального центра построить exact support

Для каждого центра (j) через предыдущий `ProjectedNeighborhoodEngine` получить

[
\mathcal N_j
============

{i:w_{ij}>0}.
]

Также сохранить:

[
k_j=|\mathcal N_j|,
]

[
M_j=\sum_{i\in\mathcal N_j}w_{ij}.
]

При необходимости:

[
n_{\mathrm{eff},j}
==================

\frac{
(\sum_iw_{ij})^2
}{
\sum_iw_{ij}^2
}.
]

Теперь вся геометрия выбора центров описывается sparse bipartite graph

[
j\longleftrightarrow i
\quad\Longleftrightarrow\quad
i\in\mathcal N_j.
]

Именно этот граф является основой support-aware selection.

---

# 8. Точное удаление бесполезных центров

Первый фильтр не является эвристикой.

Если

[
k_j\le1,
]

то после центрирования локальных данных

[
I_j=0,
\qquad
U_j=0.
]

Поэтому такой центр не влияет на objective и его можно удалить **до построения направлений и статистик**.

### Реализация

```python
if len(neighborhood.indices) <= 1:
    reject_center(j)
```

Это должен быть самый первый фильтр `CenterSelector`.

---

# 9. Удалить почти полные дубликаты центров

Если два центра имеют одинаковый или практически одинаковый support,

[
\mathcal N_j\approx\mathcal N_\ell,
]

строить для обоих дорогие статистики обычно невыгодно.

Для быстрого сравнения support хранить отсортированный массив индексов.

### Полный duplicate

Если

```python
N_j == N_l
```

один из центров можно сильно понизить в приоритете.

Для box это может переходить в exact deduplication при дополнительных условиях, но здесь достаточно использовать это как center-selection rule.

### Частичный duplicate

Можно использовать Jaccard overlap:

[
\operatorname{Jac}(j,\ell)
==========================

\frac{
|\mathcal N_j\cap\mathcal N_\ell|
}{
|\mathcal N_j\cup\mathcal N_\ell|
}.
]

Например, центры с

[
\operatorname{Jac}>0.9
]

считать почти дубликатами.

Это уже эвристический порог, не математическая гарантия.

---

# 10. Основной критерий: coverage

Нужно выбирать центры, которые покрывают разные наблюдения, а не несколько раз одну и ту же локальную область.

Храним массив

```python
coverage_count[i]
```

— сколько уже выбранных центров содержат (X_i) в support.

Для ещё не выбранного центра (j) вычисляем marginal gain.

Простейший вариант:

[
\Delta_j
========

|{
i\in\mathcal N_j:
\operatorname{coverage}_i=0
}|.
]

То есть сколько **новых** наблюдений добавляет центр.

Лучше использовать saturated coverage:

[
F(S)
====

\sum_{i=1}^n
\min
\left{
L,,
\sum_{j\in S}\mathbf1{i\in\mathcal N_j}
\right}.
]

Например (L=2): полезно, чтобы наблюдение покрывалось двумя различными локальными окрестностями.

Такой coverage criterion является monotone submodular; стандартный greedy даёт (1-1/e) приближение именно для этой coverage-задачи. В предыдущем анализе это было выделено отдельно от качества конечного (P).

---

# 11. Greedy center selection

### Инициализация

```python
selected = []
coverage = np.zeros(n, dtype=int)
```

### Итерация

Пока

```python
len(selected) < J_target
```

для каждого ещё доступного центра считаем

[
\Delta_j
========

F(S\cup{j})-F(S).
]

Выбираем

[
j_*=\arg\max_j\Delta_j.
]

После выбора:

```python
selected.append(j_star)
coverage[N[j_star]] += 1
```

и повторяем.

---

# 12. Чтобы selection не смотрел только на размер support

Чистый coverage может предпочесть большой, но статистически почти бесполезный support.

Поэтому из уже дешёво доступной информации желательно добавить quality multiplier.

Минимальный вариант:

[
Q_j
===

f(k_j,n_{\mathrm{eff},j}).
]

Например, center score:

[
\boxed{
S_j
===

\Delta_j
\cdot
Q_j.
}
]

Либо более удобно:

[
S_j=
\Delta_j+\gamma Q_j.
]

В предыдущем анализе наиболее разумный гибрид формулировался как

[
\text{support coverage} +
n_{\rm eff} +
\text{projected local variance} +
\text{previous-step leverage}.
]

Я бы внедрял это поэтапно.

### Версия 1

Только:

[
\boxed{\text{coverage}+n_{\rm eff}.}
]

### Версия 2

Добавить projected local variance.

### Версия 3

На следующих outer steps можно добавить previous-step leverage как tie-breaker.

Не надо сразу смешивать всё, иначе будет трудно понять, какая часть selection реально помогает.

---

# 13. Projected local variance считать без (d)-мерных статистик

Это особенно естественно для multi-index, поскольку `Z_X = X @ P.T` уже рассчитан.

Для центра:

[
z_i= P_kX_i,\qquad i\in\mathcal N_j.
]

Weighted projected mean:

[
\bar z_j
========

\frac{
\sum_iw_{ij}z_i
}{
\sum_iw_{ij}
}.
]

Projected covariance:

[
V_j
===

\frac{
\sum_iw_{ij}
(z_i-\bar z_j)(z_i-\bar z_j)^\top
}{
\sum_iw_{ij}
}.
]

Это матрица всего

[
m\times m.
]

Cheap information score можно взять, например,

[
\operatorname{tr}(V_j).
]

Центр, для которого

[
V_j\approx0,
]

покрывает локально почти вырожденную projected область.

Этот расчёт использует только (K_jm)-мерную информацию и не требует (U_j). Возможность делать center selection в projected (m)-мерной геометрии является одним из преимуществ multi-index screening.

---

# 14. Практическая структура `CenterSelector`

Я бы сделал объект:

```python
CenterCandidate:
    center_index
    support_indices
    weights

    support_size
    mass
    n_eff

    projected_variance_score
```

И отдельно:

```python
CenterSelector.select(
    candidates,
    n_centers,
    coverage_level=2,
)
```

Внутри:

```text
1. удалить k_j <= 1
2. optional: сгруппировать exact duplicate supports
3. инициализировать coverage[n] = 0

4. пока выбрано меньше J_target:
       для каждого кандидата:
           посчитать marginal coverage gain
           добавить quality score
           добавить duplicate penalty

       выбрать лучший центр
       обновить coverage
       удалить его из pool

5. вернуть selected center indices
```

---

# 15. В какой точке старого алгоритма вставить всё вместе

Итоговый участок outer iteration должен стать таким:

```text
P_k, Lambda_k, alpha_k, h_k
        │
        ▼
Z_X = X P_k^T
Z_C = C P_k^T
        │
        ▼
Y_X = Lambda_k^(1/2) Z_X
Y_C = Lambda_k^(1/2) Z_C
        │
        ▼
build exact m-dimensional radius-search index
        │
        ▼
для каждого candidate center
        │
        ├── radius search
        │
        ├── exact projected screening
        │
        ├── exact anisotropic q
        │
        └── exact support N_j
        │
        ▼
дешёвые center metadata
(k_j, mass, n_eff, projected variance)
        │
        ▼
удалить k_j <= 1
        │
        ▼
support-aware greedy selection
        │
        ▼
J_* выбранных центров
        │
        ▼
ТОЛЬКО ЗДЕСЬ
генерировать направления и считать I_j, U_j
```

Главное архитектурное изменение именно последнее: **в старой схеме центр выбирается до того, как известно, насколько полезна его локальная окрестность; в новой схеме сначала дешёво строится support graph, а дорогие статистики считаются только для центров, прошедших selection.**

При этом часть до `support-aware selection` является exact вычислительной оптимизацией: projected screening и radius search не меняют веса. Сам выбор подмножества центров уже меняет используемую objective; для него coverage-гарантия не является гарантией сохранения конечного (P). Это ограничение в предыдущем анализе было отмечено явно.
