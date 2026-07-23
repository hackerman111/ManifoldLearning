> **Статус: архивный документ.** Реализации `cpu_batched` и
> `cpu_compact_factored` удалены: на целевых режимах они не дали устойчивого
> выигрыша, а вынесение центра из матричных произведений оказалось численно
> неустойчивым в `float32`. Рабочая реализация — `random_projection` с явными
> разностями `X - center`.

## Что означает батчевая оптимизация

Здесь **батчевая обработка** не означает mini-batch SGD. Все (J) центров и все (n) наблюдений по-прежнему участвуют в вычислении. Меняется только организация операций:

[
\text{много скалярных циклов}
\quad\longrightarrow\quad
\text{несколько матричных умножений}.
]

Поэтому математический результат остаётся тем же с точностью до порядка суммирования чисел с плавающей точкой.

Для центра (x*j) и случайного направления (\phi*{j\ell}) алгоритм вычисляет

[
I\_{j\ell}
=========

\sum*{i=1}^{n}
Y_i
\langle X_i-x_j,\phi*{j\ell}\rangle
w\_{ij},
]

[
S\_{j\ell}
=========

\sum*{i=1}^{n}
\langle X_i-x_j,\phi*{j\ell}\rangle
w\_{ij},
]

[
U\_{j\ell}
=========

\sum*{i=1}^{n}
(X_i-x_j)
\langle X_i-x_j,\phi*{j\ell}\rangle
w\_{ij}.
]

После объединения по направлениям:

[
I_j,S_j\in\mathbb R^{n_\Phi},
\qquad
U_j\in\mathbb R^{n_\Phi\times d}.
]

Наивная реализация содержит три вложенных уровня:

```python
for j in range(J):
    for ell in range(n_phi):
        for i in range(n):
            projection = dot(X[i] - center[j], phi[j, ell])
            I[j, ell] += Y[i] * projection * w[j, i]
            S[j, ell] += projection * w[j, i]
            U[j, ell] += (X[i] - center[j]) * projection * w[j, i]
```

Стоимость равна

[
O(Jn,n_\Phi d).
]

Батчевая реализация не меняет этот порядок, но переносит вычисления из интерпретируемых циклов в GEMM-операции BLAS или GPU kernels. Формулы и обозначения соответствуют рассмотренной постановке из `manifold_new.tex`.

---

## 1. Матричное вычисление расстояний и весов

### Изотропное расстояние

Не следует строить тензор разностей

[
\Delta_{ji}=X_i-x_j
]

размера (J\times n\times d). Квадраты расстояний вычисляются через тождество

[
|X_i-x_j|\_2^2
=============

|X_i|\_2^2+|x_j|\_2^2-2x_j^\top X_i.
]

Пусть

[
X\in\mathbb R^{n\times d},
\qquad
C\in\mathbb R^{J\times d}
]

содержит наблюдения и центры. Тогда

[
D^2
===

|C|\*{\mathrm{row}}^2\mathbf 1_n^\top

- ## \mathbf 1_J|X|\*{\mathrm{row}}^{2\top}

2CX^\top.
]

Основная операция здесь:

[
CX^\top,
]

то есть один GEMM.

В NumPy-подобной записи:

```python
x_norm2 = np.sum(X * X, axis=1)          # (n,)
c_norm2 = np.sum(C * C, axis=1)          # (J,)

D2 = (
    c_norm2[:, None]
    + x_norm2[None, :]
    - 2.0 * C @ X.T
)

D2 = np.maximum(D2, 0.0)
```

Последняя операция удаляет небольшие отрицательные значения, возникающие из-за ошибки округления.

### Анизотропное расстояние в single-index ADP

При

[
T_k^2
=====

h*k^{-2}
\left(
\rho_k^2I+\beta*{k-1}\beta\_{k-1}^\top
\right)
]

имеем

[
|T_k(X_i-x_j)|\_2^2
==================

h_k^{-2}
\left[
\rho_k^2|X_i-x_j|\*2^2

- \bigl(\beta\*{k-1}^\top(X_i-x_j)\bigr)^2
  \right].
  ]

Обозначим

[
p=X\beta_{k-1}\in\mathbb R^n,
\qquad
p^{(c)}=C\beta_{k-1}\in\mathbb R^J.
]

Тогда вся матрица аргументов ядра равна

[
Q
=

h_k^{-2}
\left[
\rho_k^2D^2+
\left(
p^{(c)}\mathbf1_n^\top-\mathbf1_Jp^\top
\right)^2
\right].
]

Реализация:

```python
p_x = X @ beta
p_c = C @ beta

Q = (
    rho**2 * D2
    + (p_c[:, None] - p_x[None, :]) ** 2
) / h**2

W = kernel(Q)
```

Это устраняет тензор (J\times n\times d). Память уменьшается с

[
O(Jnd)
]

до

[
O(Jn)
]

при хранении всех расстояний или до (O(B_Jn)) при обработке блоками центров.

---

## 2. Матричная запись направленных проекций

Для одного центра введём матрицу направлений

[
\Phi_j
======

\begin{bmatrix}
\phi*{j1}^\top\
\vdots\
\phi*{jn*\Phi}^\top
\end{bmatrix}
\in\mathbb R^{n*\Phi\times d}.
]

Все направленные проекции имеют вид

[
R_j
===

(X-\mathbf1*nx_j^\top)\Phi_j^\top
\in\mathbb R^{n\times n*\Phi}.
]

Элемент этой матрицы равен

[
(R*j)*{i\ell}
=============

\langle X*i-x_j,\phi*{j\ell}\rangle.
]

Тензор разностей снова не нужен:

[
R_j
===

## X\Phi_j^\top

\mathbf1_n(x_j^\top\Phi_j^\top).
]

Для блока из (B_J) центров:

[
\Phi\in\mathbb R^{B_J\times n_\Phi\times d},
\qquad
C\in\mathbb R^{B_J\times d},
]

получаем

[
R\in\mathbb R^{B_J\times n\times n_\Phi}.
]

NumPy:

```python
# Phi: (B, n_phi, d)
# X:   (n, d)
# C:   (B, d)

x_proj = np.matmul(
    X[None, :, :],                    # (1, n, d)
    Phi.transpose(0, 2, 1),           # (B, d, n_phi)
)                                     # (B, n, n_phi)

c_proj = np.einsum(
    "bd,bld->bl",
    C,
    Phi,
)                                     # (B, n_phi)

R = x_proj - c_proj[:, None, :]        # (B, n, n_phi)
```

Массив `X[None, :, :]` обычно представляет view. Его не требуется физически копировать (B_J) раз.

---

## 3. Совместное вычисление (I), (S) и (U)

Пусть

[
w_j=
\begin{bmatrix}
w_{1j}&\cdots&w_{nj}
\end{bmatrix}^\top.
]

Введём взвешенные направленные проекции

[
Z_j
===

\operatorname{diag}(w*j)R_j
\in\mathbb R^{n\times n*\Phi}.
]

Матрицу (\operatorname{diag}(w_j)) строить нельзя. Умножение выполняется поэлементно:

[
(Z_j)*{i\ell}=w*{ij}(R_j)_{i\ell}.
]

Тогда

[
S_j=Z_j^\top\mathbf1_n,
]

[
I_j=Z_j^\top Y.
]

Для (U_j):

[
U_j
===

\sum*{i=1}^{n}
w*{ij}
\begin{bmatrix}
(R*j)*{i1}\
\vdots\
(R_j)*{in*\Phi}
\end{bmatrix}
(X_i-x_j)^\top.
]

Раскрывая разность, получаем

[
U_j
===

Z_j^\top X-S_jx_j^\top.
]

Это наиболее полезная формула. Она заменяет суммирование (d)-мерного вектора для каждой тройки ((j,i,\ell)) одним матричным умножением:

[
Z_j^\top X:
\quad
(n_\Phi\times n)(n\times d)
\longrightarrow
n_\Phi\times d.
]

Для блока центров:

```python
# R: (B, n, n_phi)
# W: (B, n)

Z = R * W[:, :, None]                  # (B, n, n_phi)

S = np.sum(Z, axis=1)                  # (B, n_phi)

I = np.einsum(
    "bnp,n->bp",
    Z,
    Y,
)                                     # (B, n_phi)

U = np.matmul(
    Z.transpose(0, 2, 1),              # (B, n_phi, n)
    X[None, :, :],                     # (1, n, d)
)                                     # (B, n_phi, d)

U -= S[:, :, None] * C[:, None, :]
```

Или полностью через `einsum`:

```python
I = np.einsum("bnp,n->bp", Z, Y)
S = np.einsum("bnp->bp", Z)
U = np.einsum("bnp,nd->bpd", Z, X)
U -= S[:, :, None] * C[:, None, :]
```

Первый вариант с `matmul` чаще напрямую отображается на batched GEMM. Конкретный победитель определяется профилированием используемой библиотеки.

### Уменьшение числа промежуточных массивов

`R` можно преобразовать в `Z` на месте:

```python
R *= W[:, :, None]
Z = R
```

После этого исходные невзвешенные проекции уже не доступны, но для построения статистик они больше не нужны.

Тогда один блок использует:

- (Q) или (W): (B_J\times n);
- (Z): (B*J\times n\times n*\Phi);
- (\Phi): (B*J\times n*\Phi\times d);
- (U): (B*J\times n*\Phi\times d).

Полный тензор (B_J\times n\times d) не создаётся.

---

## 4. Что кэшировать

### Один раз для всего запуска

Если центры не меняются:

[
|X_i|_2^2,
\qquad
|x_j|_2^2.
]

Если (x_j=X_j), это один и тот же вектор норм.

Матрицу евклидовых расстояний (D^2) можно кэшировать, если она помещается в память:

[
D^2\in\mathbb R^{J\times n}.
]

Для (J=n=1000) в `float64`:

[
1000^2\cdot8=8\text{ МБ}.
]

Для (J=n=10,000):

[
10^8\cdot8=800\text{ МБ}.
]

Во втором случае лучше пересчитывать расстояния блоками или хранить только локальных соседей.

### Один раз на внешнюю итерацию

При фиксированных

[
h_k,\quad \rho_k,\quad \beta_{k-1},\quad \Phi_k
]

фиксированы:

[
W,\qquad I,\qquad S,\qquad U.
]

Их нельзя пересчитывать внутри чередования по (c,a,\beta).

Это, вероятно, наиболее критичное правило реализации:

[
\boxed{
\text{одна внешняя итерация}
\Rightarrow
\text{один расчёт }I,S,U.
}
]

Внутренний оптимизатор должен работать только с уже построенными массивами

[
I\in\mathbb R^{J\times n_\Phi},
\quad
S\in\mathbb R^{J\times n_\Phi},
\quad
U\in\mathbb R^{J\times n_\Phi\times d}.
]

### Дополнительные кэши

Для локальных least-squares обновлений полезны

[
s^{(2)}_j=S_j^\top S_j,
\qquad
s^{(I)}_j=S_j^\top I_j,
]

[
g^{(S)}_j=U_j^\top S_j\in\mathbb R^d,
\qquad
g^{(I)}_j=U_j^\top I_j\in\mathbb R^d.
]

Для диагонального предобуславливателя:

[
d_j
===

# \operatorname{diag}(U_j^\top U_j)

\sum*{\ell=1}^{n*\Phi}U*{j\ell}\odot U*{j\ell}.
]

Векторизованно:

```python
SS = np.einsum("jp,jp->j", S, S)
SI = np.einsum("jp,jp->j", S, I)

UTS = np.einsum("jpd,jp->jd", U, S)
UTI = np.einsum("jpd,jp->jd", U, I)

diag_UTU = np.einsum("jpd,jpd->jd", U, U)
```

Кэшировать полные матрицы

[
U_j^\top U_j\in\mathbb R^{d\times d}
]

для каждого центра обычно не следует. Память составит

[
O(Jd^2),
]

а построение потребует

[
O(Jn_\Phi d^2).
]

---

## 5. Батчевое обновление локальных параметров

При фиксированном (\beta) вычислим

[
q_j=U_j\beta\in\mathbb R^{n_\Phi}.
]

Для всех центров сразу:

```python
Q_beta = np.einsum(
    "jpd,d->jp",
    U,
    beta,
)
```

Локальная задача:

[
\min_{c_j,a_j}
|I_j-c_jS_j-a_jq_j|_2^2.
]

Её нормальная система имеет размер (2\times2):

[
\begin{bmatrix}
S_j^\top S_j&S_j^\top q_j\
q_j^\top S_j&q_j^\top q_j
\end{bmatrix}
\begin{bmatrix}
c_j\a_j
\end{bmatrix}
=============

\begin{bmatrix}
S_j^\top I_j\
q_j^\top I_j
\end{bmatrix}.
]

Все коэффициенты для (J) центров вычисляются одновременно:

```python
ss = np.einsum("jp,jp->j", S, S)
sq = np.einsum("jp,jp->j", S, Q_beta)
qq = np.einsum("jp,jp->j", Q_beta, Q_beta)

si = np.einsum("jp,jp->j", S, I)
qi = np.einsum("jp,jp->j", Q_beta, I)
```

При ridge-регуляризации:

[
G_j=
\begin{bmatrix}
ss_j+\eta&sq_j\
sq_j&qq_j+\eta
\end{bmatrix}.
]

Определитель:

[
\Delta_j
========

(ss_j+\eta)(qq_j+\eta)-sq_j^2.
]

Решение:

[
c_j
===

\frac{(qq_j+\eta)si_j-sq_jqi_j}{\Delta_j},
]

[
a_j
===

\frac{(ss_j+\eta)qi_j-sq_jsi_j}{\Delta_j}.
]

Это позволяет вообще не вызывать `lstsq` (J) раз:

```python
g00 = ss + eta
g01 = sq
g11 = qq + eta

det = g00 * g11 - g01 * g01
det = np.maximum(det, det_floor)

c = (g11 * si - g01 * qi) / det
a = (g00 * qi - g01 * si) / det
```

Но простое ограничение `det_floor` меняет решение в вырожденных случаях. Корректнее адаптивно увеличивать (\eta_j) либо помечать центры с малой эффективной массой как невалидные.

Для максимальной устойчивости можно собрать массив

[
G\in\mathbb R^{J\times2\times2}
]

и использовать batched Cholesky или batched `solve`.

---

## 6. Матричная реализация глобального оператора

При фиксированных (a_j,c_j) глобальная задача использует матрицу

[
A=
\begin{bmatrix}
a_1U_1\
\vdots\
a_JU_J
\end{bmatrix}
\in\mathbb R^{Jn_\Phi\times d}.
]

Физически собирать (A) не требуется.

### Оператор (v\mapsto Av)

[
(Av)_j=a_jU_jv.
]

```python
def A_matvec(U, a, v):
    uv = np.einsum("jpd,d->jp", U, v)
    return a[:, None] * uv
```

### Оператор (z\mapsto A^\top z)

Пусть (z\in\mathbb R^{J\times n\_\Phi}). Тогда

[
A^\top z
========

\sum\_{j=1}^{J}a_jU_j^\top z_j.
]

```python
def AT_matvec(U, a, z):
    return np.einsum(
        "jpd,jp->d",
        U,
        a[:, None] * z,
    )
```

### Оператор нормальных уравнений

[
Hv
==

\sum\_{j=1}^{J}
a_j^2U_j^\top U_jv+\lambda v.
]

```python
def normal_matvec(U, a, v, reg):
    uv = np.einsum("jpd,d->jp", U, v)
    return (
        np.einsum(
            "jpd,jp->d",
            U,
            (a * a)[:, None] * uv,
        )
        + reg * v
    )
```

Стоимость одного вызова:

[
O(Jn_\Phi d).
]

Память сверх (U):

[
O(Jn_\Phi+d).
]

Эта запись подходит для CG, LSQR и LSMR.

---

## 8. Выбор размера блока центров

Пусть одновременно обрабатываются (B_J) центров. Главный временный массив:

[
Z\in\mathbb R^{B_J\times n\times n_\Phi}.
]

В `float64` он занимает

[
8B_Jn n_\Phi\ \text{байт}.
]

Например, при

[
B_J=128,\qquad n=1000,\qquad n_\Phi=32
]

получаем

[
128\cdot1000\cdot32\cdot8
=========================

32.8\text{ МБ}.
]

Массив (U) для блока:

[
128\cdot32\cdot100\cdot8
========================

3.28\text{ МБ}.
]

Если одновременно хранить `R` и `Z`, расход на направленные проекции удвоится. Поэтому выгодно умножать `R` на веса на месте.

Приближённая оценка памяти на блок:

[
M_{\mathrm{block}}
\approx
sB_J
\left(
n n_\Phi+n+2n_\Phi d
\right),
]

где (s=8) для `float64` и (s=4) для `float32`.

Отсюда:

[
B_J
\lesssim
\frac{M_{\mathrm{budget}}}
{s(nn_\Phi+n+2n_\Phi d)}.
]

Не следует занимать всю доступную память. Нужны буферы BLAS, данные алгоритма и память интерпретатора.

Разумные стартовые значения для профилирования:

- CPU: (B_J\in{16,32,64,128,256});
- GPU: (B_J\in{64,128,256,512}).

Это не параметры статистической модели. Они выбираются только по времени и памяти.

---

## 10. CPU-реализация

На CPU основной выигрыш дают:

- GEMM через MKL, OpenBLAS или BLIS;
- непрерывное хранение последней оси (d);
- блоки центров;
- отсутствие циклов Python по (i) и (\ell);
- повторное использование (I,S,U);
- отсутствие конкурирующих уровней многопоточности.

Последнее означает: не следует одновременно запускать много процессов по центрам и оставлять каждому процессу многопоточный BLAS. Например, 12 процессов по 12 BLAS-потоков создадут 144 вычислительных потока на 12-ядерном CPU.

Следует выбрать один вариант:

[
\text{один процесс + многопоточный BLAS}
]

или

[
\text{несколько процессов + один BLAS-поток на процесс}.
]

Для плотной батчевой математики первый вариант обычно проще.

---

---

## 12. Рекомендуемая структура вычислений

Одна внешняя итерация:

```text
1. Вычислить beta-проекции X beta и C beta.
2. Для каждого блока центров:
   2.1. Вычислить D² через GEMM.
   2.2. Вычислить анизотропный аргумент Q.
   2.3. Вычислить веса W.
   2.4. Построить направленные проекции R.
   2.5. Умножить R на W на месте.
   2.6. Получить I и S суммированием.
   2.7. Получить U через batched GEMM.
   2.8. Записать I, S, U в итоговые массивы.
3. Освободить W, Q и временные блоковые буферы.
4. Запустить внутреннюю оптимизацию только на I, S, U.
5. Все U beta считать батчево.
6. Все локальные системы 2×2 решать батчево.
7. Глобальную систему решать matrix-free.
```

Главный принцип:

[
\boxed{
\text{дорогая сумма по наблюдениям выполняется один раз на внешний шаг;}
}
]

[
\boxed{
\text{внутренние итерации больше не обращаются к исходным }X_i.
}
]

---

## 13. Что внедрять первым

1. **Убедиться, что (I,S,U) не пересчитываются внутри inner loop.**
2. **Заменить построение (X_i-x_j) на формулу расстояний через GEMM.**
3. **Реализовать блоковую формулу**
   [
   U_j=Z_j^\top X-S_jx_j^\top.
   ]
4. **Заменить циклы по центрам при вычислении (U_j\beta) на `einsum`.**
5. **Заменить (J) вызовов `lstsq` батчевым решением систем (2\times2).**
6. **Добавить компактные локальные окрестности, если доля ненулевых весов мала.**
7. **Только после профилирования переносить вычислительное ядро на GPU.**

## 14. Как проверить корректность

Нужна медленная эталонная реализация на малом наборе:

[
n=30,\qquad d=5,\qquad J=10,\qquad n_\Phi=4.
]

Для одинаковых весов и направлений сравнить:

[
\frac{|I_{\mathrm{batch}}-I_{\mathrm{ref}}|*F}
{\max(1,|I*{\mathrm{ref}}|_F)},
]

[
\frac{|S_{\mathrm{batch}}-S_{\mathrm{ref}}|*F}
{\max(1,|S*{\mathrm{ref}}|_F)},
]

[
\frac{|U_{\mathrm{batch}}-U_{\mathrm{ref}}|*F}
{\max(1,|U*{\mathrm{ref}}|_F)}.
]

В `float64` разумно ожидать расхождение порядка (10^{-12})–(10^{-10}), но точный уровень зависит от масштаба данных и порядка суммирования.

После этого сравниваются:

- значение целевой функции;
- последовательность внешних (h_k,\rho_k);
- итоговое направление через
  [
  |\widehat\beta_{\mathrm{batch}}^\top
  \widehat\beta_{\mathrm{ref}}|;
  ]
- время отдельно для построения весов, статистик и внутреннего решателя.
