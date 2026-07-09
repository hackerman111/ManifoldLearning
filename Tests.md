Нужно разделить работу на две фазы:

1. **tuning-фаза** — ищем начальные параметры ADP.
2. **confirmatory-фаза** — фиксируем найденные параметры и проверяем 5 синих требований на независимых seed и сценариях.

Иначе получится подгонка: параметры выбраны на тех же данных, на которых потом доказывается качество.

---

# 0. Tuning-фаза: поиск начальных параметров ADP

Цель: подобрать рабочие значения

[
h_0,\quad n_{\min},\quad n_{\Phi},\quad \lambda,\quad J,\quad m_{\max},\quad K_{\max},\quad \gamma_h.
]

Здесь:

[
J=\text{число центров},\qquad
n_{\Phi}=\text{число случайных направлений},
]

[
\lambda=\text{регуляризация обновления }\beta,
\qquad
\gamma_h=\text{множитель уменьшения }h_k.
]

## 0.1. Сценарии для подбора

Не надо подбирать параметры только на одном лёгком случае. Нужен небольшой, но разный validation-набор:

| Параметр                | Значения             |
| ----------------------- | -------------------- |
| (d)                     | (50,100,200)         |
| (n/d)                   | (10,20)              |
| (\rho\_{\mathrm{corr}}) | (0,0.3,0.7)          |
| SNR                     | (20,10,5)            |
| link (f(t))             | (t,\tanh(t),\sin(t)) |
| sparsity (q)            | (0.1,0.3,1.0)        |

Полный перебор слишком дорогой:

[
3\cdot2\cdot3\cdot3\cdot3\cdot3=486
]

сценариев. Для tuning достаточно взять **36 сценариев** через стратифицированную выборку: чтобы каждое значение каждого фактора встречалось несколько раз.

## 0.2. Сетка начальных параметров

### (n\_{\Phi})

[
n_{\Phi}\in{16,32,64,128}.
]

Для (d=50) обычно хватит (16)–(32), для (d=100) — (32)–(64), для (d=200) — (64)–(128).

### (n\_{\min})

Задавать не абсолютным числом, а через (n\_{\Phi}):

[
n_{\min}\in{n_{\Phi}+4,;2n_{\Phi},;4n_{\Phi},;8n_{\Phi}}.
]

Причина: если локальная задача использует (n*{\Phi}) направлений, то масса меньше (n*{\Phi}) почти всегда даёт шумную локальную регрессию.

### Правило выбора (h_0)

Проверять три варианта:

[
\text{mean-rule}:\quad \frac1J\sum_j M_j(h_0)\ge n_{\min},
]

[
\text{q05-rule}:\quad Q_{0.05}(M_j(h_0))\ge n_{\min},
]

[
\text{mean+filter}:\quad \frac1J\sum_jM_j(h_0)\ge n_{\min},\quad M_j(h_0)\ge m_{\min}.
]

Для `mean+filter`:

[
m_{\min}=n_{\Phi}+4.
]

По твоим (h_0)-результатам `mean-rule` корректен как проверка формулы, но он допускает плохие центры. Поэтому в реальных ADP-экспериментах `q05-rule` и `mean+filter` обязательны.

### Inflation (h_0)

После выбора (h_0) проверить:

[
h_0^{\mathrm{used}}=\alpha_h h_0,
]

[
\alpha_h\in{1.0,1.1,1.25,1.5}.
]

Это нужно, потому что минимальное (h_0) может быть слишком агрессивным.

### (\lambda)

Лучше использовать относительную регуляризацию:

[
\lambda=\lambda_{\mathrm{rel}}\cdot s_A,
]

где (s_A) — масштаб матрицы в (\beta)-обновлении, например

[
s_A=\frac{\operatorname{tr}(A)}{d}.
]

Проверять:

[
\lambda_{\mathrm{rel}}\in{0,10^{-4},10^{-3},10^{-2},10^{-1},1}.
]

### Число центров (J)

[
J/n\in{0.25,0.5,1.0}.
]

Для статьи основной вариант:

[
J=n.
]

Для практического ускорения:

[
J=0.25n\text{ или }0.5n.
]

### Внешние и внутренние итерации

[
K_{\max}\in{5,8,12},
]

[
m_{\max}\in{10,20,40},
]

[
\gamma_h\in{0.7,0.8,0.9}.
]

## 0.3. Как не делать гигантский перебор

Полный перебор слишком большой. Нужен staged search.

### Stage A: coarse screening

Случайно выбрать 250 конфигураций параметров из сетки.

Для каждой конфигурации:

```text
36 validation-сценариев
20 seed на сценарий
```

Итого:

[
250\cdot36\cdot20=180000
]

коротких запусков.

Чтобы сократить, можно на Stage A использовать:

```text
K_max = 5
m_max = 10
J/n = 0.25
```

Цель Stage A — отсеять плохие параметры, а не получить финальные числа.

### Stage B: refinement

Взять top-30 конфигураций.

Для каждой:

```text
36 validation-сценариев
50 seed на сценарий
```

Итого:

[
30\cdot36\cdot50=54000
]

запусков.

### Stage C: final tuning

Взять top-5 конфигураций.

Для каждой:

```text
36 validation-сценариев
100 seed на сценарий
```

Итого:

[
5\cdot36\cdot100=18000
]

запусков.

## 0.4. Score для выбора параметров

Основная метрика:

[
c_K=|\cos(\hat\beta_K,\beta^*)|.
]

Но выбирать только по медиане (c_K) плохо: можно получить медленный или нестабильный режим.

Использовать score:

[
\mathrm{Score}
==============

\operatorname{median}(c*K)
-0.5\operatorname{FailRate}
-0.1\operatorname{IQR}(c_K)
-0.02\log_2\frac{T}{T*{\mathrm{ref}}}.
]

Где:

[
\operatorname{FailRate}
=======================

\Pr(\text{NaN, divergence, }c_K<0.3).
]

Параметры проходят tuning, если:

[
\operatorname{median}(c_K)\ge0.85,
]

[
\Pr(c_K\ge0.8)\ge0.8,
]

[
\operatorname{FailRate}\le0.05.
]

## 0.5. Кандидаты default-параметров до tuning

Для (d=100,n=1000):

```text
n_phi = 64
n_min = 4 * n_phi = 256
h0_rule = q05 или mean+filter
h0_inflation = 1.1
lambda_rel = 1e-2
J/n = 0.5 для быстрой версии, 1.0 для финальной
K_max = 8
m_max = 20
gamma_h = 0.8
inner_tol = 1e-5
```

Для (d=200,n=2000):

```text
n_phi = 128
n_min = 4 * n_phi = 512
h0_rule = q05 или mean+filter
h0_inflation = 1.1 или 1.25
lambda_rel = 1e-2 или 1e-1
J/n = 0.25 или 0.5
K_max = 8
m_max = 20
gamma_h = 0.8
```

---

# 1. Эксперимент по (h_0)

Синий текст:

[
{\color{blue}\text{please plot }h^{\mathrm{iso}}_0}
]

## Цель

Проверить, что начальный изотропный bandwidth выбирается корректно и устойчиво:

[
M_{\mathrm{iso}}(h_0)\ge n_{\min}.
]

Также проверить, что выбранный (h_0) не создаёт плохие локальные центры.

## Серьёзные значения

| Параметр                | Значения                    |
| ----------------------- | --------------------------- |
| (d)                     | (50,100,200,500)            |
| (n/d)                   | (10,20)                     |
| (\rho\_{\mathrm{corr}}) | (0,0.3,0.7)                 |
| (\sigma_X)              | (0.7,1.0,1.3)               |
| (J/n)                   | (0.25,0.5,1.0)              |
| (n\_{\min})             | (32,64,128,256,512)         |
| правило (h_0)           | mean, q05, q10, mean+filter |

Для каждого сценария:

[
500\text{ seed}.
]

Это дешёвый эксперимент, потому что (Y) и ADP-итерации не нужны.

## Что сохранять

```text
seed
d
n
corr
sigma_x
J
n_min
h0_rule
h0
h0 / sigma_x
M_mean(h0)
M_q05(h0)
M_q10(h0)
M_min(h0)
pass_mean
pass_q05
pass_monotone
```

## Графики

1. (M\_{\mathrm{iso}}(h)) против (h/h_0), но с ограничением:

[
y\in[0,1.5n_{\min}^{\max}].
]

2. Boxplot (h*0/\sigma_X) по (n*{\min}).

3. Median (h_0) по (d).

4. Median (h*0) по (\rho*{\mathrm{corr}}).

5. (Q*{0.05}(M_j(h_0))) против (n*{\min}).

## Критерии прохождения

Для mean-rule:

[
\Pr(M_{\mathrm{mean}}(h_0)\ge n_{\min})\ge0.99.
]

Для q05-rule:

[
\Pr(Q_{0.05}(M_j(h_0))\ge n_{\min})\ge0.99.
]

Монотонность:

[
h_0(n_{\min,1})\le h_0(n_{\min,2})
\quad\text{при }n_{\min,1}<n_{\min,2}
]

должна выполняться в (99%) запусков.

## Главный вывод, который нужен

Этот эксперимент должен ответить:

```text
Можно ли использовать mean-rule из текста, или для реального ADP нужен q05/mean+filter?
```

По твоим текущим результатам ожидаемый ответ:

```text
mean-rule корректен как формула, но для ADP лучше q05 или mean+filter.
```

---

# 2. Эксперимент по информативности (\beta_0)

Синий текст:

[
{\color{blue}\text{check whether }|\cos(\beta_0,\beta^*)|\text{ is significantly positive}}
]

## Цель

Проверить, что Step 0 даёт направление лучше случайного.

[
c_0=|\cos(\beta_0,\beta^*)|.
]

Нулевая модель:

[
u\sim \operatorname{Unif}(S^{d-1}),
\qquad
c_{\mathrm{rand}}=|u^\top\beta^*|.
]

## Серьёзные значения

| Параметр                | Значения                 |
| ----------------------- | ------------------------ |
| (d)                     | (50,100,200)             |
| (n/d)                   | (10,20,50)               |
| (\rho\_{\mathrm{corr}}) | (0,0.3,0.7)              |
| SNR                     | (20,10,5,2)              |
| (q)                     | (0.1,0.3,1.0)            |
| (f(t))                  | (t,\tanh(t),\sin(t),t^2) |

Важно: (t^2) — hard/diagnostic case. Для симметричного (X) он может быть плох для average derivative, потому что производная меняет знак. Его не надо смешивать с обычными сценариями.

## Число запусков

Для core-сценариев:

[
200\text{ seed}.
]

Для hard-сценариев:

[
100\text{ seed}.
]

Core-сценарии:

[
d\in{50,100,200},
\quad n/d\in{10,20},
\quad \rho_{\mathrm{corr}}\in{0,0.3,0.7},
\quad \mathrm{SNR}\in{20,10,5},
\quad f\in{t,\tanh(t),\sin(t)}.
]

Это:

[
3\cdot2\cdot3\cdot3\cdot3=162
]

сценария.

[
162\cdot200=32400
]

Step-0 запусков.

## Что сохранять

```text
seed
scenario_id
d
n
corr
snr
q
link
h0_rule
h0
n_phi
n_min
lambda_rel
cos_beta0
random_cos_median
random_cos_q95
beta0_norm
local_mass_mean
local_mass_q05
failed
```

## Статистическая проверка

Для каждого сценария:

[
H_0:\ c_0\text{ не лучше случайного направления}.
]

Практический критерий:

[
\operatorname{median}(c_0)>
\operatorname{median}(c_{\mathrm{rand}})
]

и

[
\Pr(c_0>q_{0.95}^{\mathrm{rand}})\ge0.7.
]

Для сильного прохождения:

[
\Pr(c_0>q_{0.95}^{\mathrm{rand}})\ge0.8.
]

Также считать bootstrap CI:

[
\operatorname{CI}*{0.95}(\operatorname{median}(c_0-c*{\mathrm{rand}})).
]

Прошло, если нижняя граница (>0).

## Графики

1. Boxplot (c_0) против random baseline.
2. (c_0) против (d).
3. (c_0) против (n/d).
4. (c_0) против SNR.
5. Heatmap success rate по ((d,n/d)).

## Главный вывод

Этот эксперимент должен ответить:

```text
В каких режимах изотропная начальная оценка beta0 вообще содержит сигнал?
```

Если (\beta_0) не лучше random baseline, следующие ADP-итерации не стоит интерпретировать как успешные.

---

# 3. Эксперимент по внутренней оптимизации

Синий текст:

[
{\color{blue}\text{please check}}
]

после формул alternating minimization.

## Цель

Проверить, что формулы обновления и численное решение работают правильно.

Здесь не доказывается качество статистики. Здесь проверяется solver.

## 3.1. Unit-тесты на малых задачах

Параметры:

[
d\in{5,10,20},
\quad J\in{5,20,100},
\quad n_{\Phi}\in{8,16,32}.
]

Для каждой комбинации:

[
1000\text{ synthetic задач}.
]

Проверки:

1. Размерности:

[
U_j\in\mathbb R^{n_\Phi\times d},
]

[
U_j^\top U_j\in\mathbb R^{d\times d},
]

[
U_j^\top(I_j-c_jS_j)\in\mathbb R^d.
]

2. Локальное обновление ((c_j,\ell_j)) сравнить с `np.linalg.lstsq`.

Критерий:

[
\max*j
\left|
(c_j,\ell_j)*{\mathrm{formula}}

---

(c*j,\ell_j)*{\mathrm{lstsq}}
\right|
<10^{-8}.
]

3. (\beta)-обновление сравнить с dense solve:

[
A=\sum_j \ell_j^2U_j^\top U_j+\lambda I.
]

[
b=\sum_j\ell_jU_j^\top(I_j-c_jS_j)+\lambda\beta_{\mathrm{prior}}.
]

Критерий:

[
\frac{|\beta_{\mathrm{cg}}-\beta_{\mathrm{dense}}|}
{|\beta_{\mathrm{dense}}|}
<10^{-6}.
]

## 3.2. Рабочие ADP-задачи

Взять 24 сценария:

| Параметр | Значения     |
| -------- | ------------ |
| (d)      | (50,100,200) |
| (n/d)    | (10,20)      |
| corr     | (0,0.7)      |
| SNR      | (10,5)       |
| link     | (t,\tanh(t)) |

Для каждого:

[
200\text{ seed}.
]

Итого:

[
24\cdot200=4800
]

полных запусков solver-проверки.

## Что сохранять

```text
seed
outer_k
inner_m
objective
objective_delta
beta_delta
beta_norm
cg_residual
cg_iters
condition_estimate
failed
```

## Критерии прохождения

Нормировка:

[
||\beta^{(m)}|-1|<10^{-10}
]

для float64.

Objective:

[
L^{(m+1)}\le L^{(m)}+10^{-8}\max(1,L^{(m)}).
]

CG:

[
\frac{|A\beta-b|}{|b|}<10^{-5}.
]

Сходимость:

[
|\beta^{(m)}-\beta^{(m-1)}|<10^{-4}
]

в (90%) запусков до (m\_{\max}).

## Графики

1. Objective vs inner iteration (m).
2. (|\beta^{(m)}-\beta^{(m-1)}|) vs (m).
3. CG residual vs (m).
4. Histogram числа CG-итераций.
5. Failure rate по (d).

## Главный вывод

Этот эксперимент должен ответить:

```text
Ошибки в ADP идут от статистики или от сломанного solver?
```

Если solver не проходит этот блок, эксперименты 4–5 нельзя интерпретировать.

---

# 4. Эксперимент по (\rho_k=\aniso_k)

Синий текст:

[
{\color{blue}\text{plot }\aniso_k}
]

## Цель

Проверить, что внешний ADP-цикл строит разумную анизотропию.

[
\rho_k\in[0,1]
]

и локальная масса при новом (h_k,\rho_k) не вырождается.

## Серьёзные значения

Использовать параметры, выбранные в tuning-фазе.

Сценарии:

| Параметр | Значения             |
| -------- | -------------------- |
| (d)      | (50,100,200)         |
| (n/d)    | (10,20)              |
| corr     | (0,0.3,0.7)          |
| SNR      | (20,10,5)            |
| link     | (t,\tanh(t),\sin(t)) |

Итого:

[
3\cdot2\cdot3\cdot3\cdot3=162
]

сценария.

Для каждого:

[
100\text{ seed}.
]

Итого:

[
16200
]

полных ADP-запусков.

## Что сохранять

```text
seed
scenario_id
outer_k
h_k
rho_k
local_mass_mean
local_mass_q05
local_mass_min
cos_beta_k
beta_delta_outer
objective_final_inner
failed
```

## Проверки

1. Диапазон:

[
0\le\rho_k\le1.
]

2. Масса:

[
M_{\mathrm{mean},k}\ge n_{\min}
]

для mean-rule.

Для серьёзной устойчивости:

[
Q_{0.05}(M_{j,k})\ge n_{\Phi}+4.
]

3. Поведение (\rho_k):

Строгую монотонность требовать нельзя, но медиана по seed должна иметь понятный тренд:

[
\operatorname{median}(\rho_{k+1})
\le
\operatorname{median}(\rho_k)+0.05.
]

4. Связь с качеством:

Проверить корреляцию:

[
\rho_k\downarrow
\quad\text{и}\quad
|\cos(\beta_k,\beta^*)|\uparrow.
]

Не как строгую теорему, а как диагностический график.

## Графики

1. Median (\rho_k) vs (k) с 25–75% полосой.
2. Median (h_k) vs (k).
3. (Q*{0.05}(M*{j,k})) vs (k).
4. (|\cos(\beta_k,\beta^\*)|) vs (k).
5. Scatter (\rho_k) vs (|\cos(\beta_k,\beta^\*)|).

## Главный вывод

Эксперимент должен ответить:

```text
Анизотропный параметр rho_k действительно управляет локализацией, или просто случайно прыгает?
```

---

# 5. Эксперимент по росту (|\cos(\beta_k,\beta^\*)|)

Синий текст:

[
{\color{blue}\text{check whether }|\cos(\beta_0,\beta^*)|\text{ grows to one with iterations}}
]

В этой фразе, вероятно, опечатка. Проверять нужно:

[
|\cos(\beta_k,\beta^*)|,
]

потому что (\beta_0) не меняется по итерациям.

## Цель

Проверить главное утверждение ADP:

[
|\cos(\beta_k,\beta^*)|\to1
]

или хотя бы устойчиво растёт по внешним итерациям.

## Серьёзные значения

Использовать независимые test-сценарии, которые не участвовали в tuning.

Core test:

| Параметр | Значения             |
| -------- | -------------------- |
| (d)      | (50,100,200)         |
| (n/d)    | (10,20)              |
| corr     | (0,0.3,0.7)          |
| SNR      | (20,10,5)            |
| link     | (t,\tanh(t),\sin(t)) |
| (q)      | (0.3,1.0)            |

Можно взять не полный перебор, а 72 сценария через balanced design.

Для каждого:

[
200\text{ seed}.
]

Итого:

[
72\cdot200=14400
]

запусков для full ADP.

## Ablation-режимы

Чтобы доказать, что рост даёт именно ADP, нужны сравнения:

| Режим               | Что отключено        |
| ------------------- | -------------------- |
| `full_adp`          | ничего               |
| `step0_only`        | нет внешних итераций |
| `no_anisotropy`     | (\rho_k=1)           |
| `fixed_h`           | (h_k=h_0)            |
| `no_regularization` | (\lambda=0)          |
| `random_beta_init`  | старт не из Step 0   |

Для ablation достаточно 36 сценариев:

[
36\cdot200\cdot6=43200
]

запусков.

## Что сохранять

```text
seed
scenario_id
method
outer_k
cos_beta_k
cos_delta_from_k0
success_08
success_09
h_k
rho_k
local_mass_q05
runtime_sec
failed
```

## Основные критерии

Рост:

[
\Delta c=c_K-c_0.
]

Прошло, если:

[
\operatorname{median}(\Delta c)>0,
]

[
\operatorname{CI}_{0.95}(\operatorname{median}\Delta c)>0,
]

[
\Pr(c_K>c_0)\ge0.75.
]

Сильный успех:

[
\Pr(c_K\ge0.9)\ge0.7.
]

Средний успех:

[
\Pr(c_K\ge0.8)\ge0.8.
]

Сравнение с ablation:

[
\operatorname{median}(c_K^{\mathrm{full}})

>

\operatorname{median}(c_K^{\mathrm{no_anisotropy}})

>

\operatorname{median}(c_0).
]

Статистика:

```text
paired bootstrap CI для median difference
Wilcoxon signed-rank test для paired seed
Holm correction по сценариям
```

## Графики

1. Median (|\cos(\beta_k,\beta^\*)|) vs (k).
2. Success rate (\Pr(c_k\ge0.8)) vs (k).
3. Full ADP vs ablations.
4. Heatmap success rate по ((d,n/d)).
5. Failure rate по (d).

## Главный вывод

Эксперимент должен ответить:

```text
Рост качества вызван именно итеративной анизотропной адаптацией, а не случайным улучшением solver-а или большим h0.
```

---

# Рекомендуемый финальный протокол

## Tuning

```text
250 конфигураций × 36 сценариев × 20 seed
top-30 × 36 сценариев × 50 seed
top-5 × 36 сценариев × 100 seed
```

После этого фиксируются:

```text
h0_rule
h0_inflation
n_phi
n_min
lambda_rel
J/n
K_max
m_max
gamma_h
```

## Confirmatory

| Эксперимент  |                             Запуски |
| ------------ | ----------------------------------: |
| (h_0)        | 500 seed на геометрический сценарий |
| (\beta_0)    |           200 seed на core-сценарий |
| solver check |            1000 unit + 4800 рабочих |
| (\rho_k)     |            100 seed на 162 сценария |
| cos growth   | 200 seed на 72 сценария + ablations |

Это уже уровень нормального исследовательского протокола: не один seed, не игрушечное (d=10), есть tuning/test split, ablations, доверительные интервалы и проверка failure rate.

---

# Что считать финальным успехом ADP

ADP можно считать рабочим в режиме ((d,n)), если одновременно:

[
\operatorname{median}|\cos(\beta_0,\beta^\*)|

>

\operatorname{median}|\cos(u,\beta^\*)|,
]

[
\operatorname{median}|\cos(\beta_K,\beta^*)|\ge0.8,
]

[
\Pr(|\cos(\beta_K,\beta^*)|\ge0.8)\ge0.8,
]

[
\operatorname{FailRate}\le0.05,
]

[
Q_{0.05}(M_{j,k})\ge n_{\Phi}+4
\quad\text{на большинстве итераций}.
]

Сильный режим:

[
\operatorname{median}|\cos(\beta_K,\beta^*)|\ge0.9.
]

Пограничный режим:

[
0.7\le
\operatorname{median}|\cos(\beta_K,\beta^*)|
<0.8.
]

Breakdown:

[
\operatorname{median}|\cos(\beta_K,\beta^*)|<0.7
]

или

[
\operatorname{FailRate}>0.1.
]

---

# Практический набор параметров для первого серьёзного запуска

До окончания tuning можно взять такой старт:

```text
d = 100
n = 1000

n_phi = 64
n_min = 256

h0_rule = q05
h0_inflation = 1.1

lambda_rel = 1e-2

J/n = 0.5 для tuning
J/n = 1.0 для финального подтверждения

K_max = 8
m_max = 20
gamma_h = 0.8
inner_tol = 1e-5
```

Для (d=200,n=2000):

```text
n_phi = 128
n_min = 512
h0_rule = q05
h0_inflation = 1.1 или 1.25
lambda_rel = 1e-2 или 1e-1
J/n = 0.25 или 0.5
K_max = 8
m_max = 20
gamma_h = 0.8
```

Главное изменение относительно старого сценария: (n*{\min}=5,10,20,40) больше не годится как основной диапазон для реального ADP при (d=100). Эти значения подходят для проверки формулы (h_0), но для локальных задач с (n*{\Phi}=32)–(64) они слишком малы.
