Да. Но я бы **не распараллеливал текущие greedy-компоненты как \(r\) независимых задач**. Это меняет алгоритм в плохую сторону. Для GPU лучше заменить rank-1 шаг на **block rank-\(q\) шаг**, в котором \(q\) направлений оптимизируются совместно.

### Почему нельзя просто считать \(v_1,\ldots,v_r\) независимо

В текущем алгоритме

\[
Q_k
=

\sum_j N_jg_j(U_j^\top e_j^{(k)})^\top
+\lambda(P-B_{k-1}),
\]

где

\[
e_j^{(k)}
=

I_j-U_jB_{k-1}^{\top}g_j.
\]

Поэтому

\[
Q_1\neq Q_2\neq\cdots\neq Q_r.
\]

Например,

\[
v_2
=

v_1\!\left(
Q_2(I-v_1v_1^\top)
\right),
\]

а сам \(Q_2\) можно получить только после вычисления первой компоненты и обновления \(B_1\).

Если запустить

\[
v_1,\ldots,v_r
\]

параллельно от одного \(Q_1\), это уже не текущий greedy solver. Более того, все процессы склонны находить близкие направления максимального gain.

Для обычного truncated SVD фиксированной матрицы это решается block Lanczos/randomized SVD. Такие blocked методы хорошо подходят GPU именно потому, что вместо большого числа matvec выполняют матричные операции. :chatgpt-content-reference{index="0"}

---

# 1. Минимальная модификация: block-greedy solver

Вместо

\[
B_k=B_{k-1}+\sigma av^\top
\]

добавляем сразу rank-\(q\) поправку

\[
\boxed{
B_{\text{new}}
=

B+C V^\top
}
\]

с

\[
C\in\mathbb R^{m\times q},
\qquad
V\in\mathbb R^{d\times q}.
\]

Пусть

\[
E_j=I_j-U_jB^\top g_j,
\qquad
Z=P-B.
\]

Тогда новый внутренний шаг решает

\[
\boxed{
\min_{C,V}
\sum_jN_j
\left\|
E_j-U_jVC^\top g_j
\right\|^2 +
\lambda
\left\|
CV^\top-Z
\right\|_F^2.
}
\]

Это точный аналог вашего rank-1 refinement, но сразу для \(q\) направлений.

Если \(q=1\), получаем почти вашу текущую задачу.

---

## 2. Block spectral initialization

Сначала вычисляем тот же

\[
Q=-\frac12\nabla F(B).
\]

Удаляем уже найденное пространство:

\[
Q_\perp=Q(I-V_{\rm old}V_{\rm old}^\top).
\]

Но теперь берём не один leading singular vector, а сразу \(q\):

\[
Q_\perp
\approx
A_q\Sigma_qV_q^\top.
\]

Поскольку \(m\ll d\), всё так же достаточно построить

\[
H=Q_\perp Q_\perp^\top
\in\mathbb R^{m\times m}
\]

и найти

\[
H A_q=A_q\Lambda_q.
\]

После этого

\[
\boxed{
V_q
=

Q_\perp^\top A_q
\Lambda_q^{-1/2}.
}
\]

То есть сразу получаем

\[
[v_1,\ldots,v_q].
\]

Сам eigensolve маленький и большого GPU-ускорения не даст. Главное изменение происходит дальше.

Block eigensolvers в принципе хорошо параллелятся; LOBPCG и близкие методы используют именно одновременную работу с несколькими eigenvectors. GPU-реализации такого подхода существуют. :chatgpt-content-reference{index="1"}

---

# 3. Главное изменение: заменить \(a\leftrightarrow v\) на \(C\leftrightarrow V\)

### Фиксируем \(V\)

Пусть

\[
Y_j=U_jV\in\mathbb R^{n_j\times q}.
\]

Вместо

\[
U_jv
\]

теперь GPU считает сразу

\[
\boxed{U_jV}.
\]

Это уже matrix-matrix multiplication вместо matrix-vector multiplication.

Для ортонормированного \(V\) задача по \(C\):

\[
\min_C
\sum_jN_j
\left\|
E_j-Y_jC^\top g_j
\right\|^2 +
\lambda
\left\|
CV^\top-Z
\right\|_F^2.
\]

Её normal operator:

\[
\boxed{
\mathcal H_C(D)
=

\sum_j
N_j
g_jg_j^\top
D
(V^\top U_j^\top U_jV) +
\lambda D.
}
\]

Правая часть:

\[
\boxed{
R_C
=

\sum_j
N_j
g_jE_j^\top U_jV +
\lambda ZV.
}
\]

То есть

\[
\mathcal H_C(C)=R_C.
\]

Размер неизвестной всего

\[
m\times q.
\]

При малых \(m,q\) её вообще можно решать плотным Cholesky.

---

# 4. Фиксируем \(C\)

Определим

\[
\alpha_j=C^\top g_j\in\mathbb R^q.
\]

Теперь требуется решить

\[
\boxed{
\min_X
\sum_jN_j
\left\|
E_j-U_jX\alpha_j
\right\|^2 +
\lambda
\left\|
XC^\top-Z^\top
\right\|_F^2.
}
\]

Здесь

\[
X\in\mathbb R^{d\times q}.
\]

Normal operator имеет вид

\[
\boxed{
\mathcal H_V(X)
=

\sum_j
N_j
U_j^\top U_j
X
\alpha_j\alpha_j^\top +
\lambda X(C^\top C).
}
\]

Правая часть:

\[
\boxed{
R_V
=

\sum_j
N_j
U_j^\top E_j\alpha_j^\top +
\lambda Z^\top C.
}
\]

Поэтому

\[
\mathcal H_V(V)=R_V.
\]

Вот эта операция очень подходит GPU.

Вместо текущего

\[
U_jv
\]

на каждой итерации LSMR выполняется

\[
\boxed{
U_jV
}
\]

для \(q\) направлений одновременно.

А обратная операция:

\[
U_j^\top Y_j.
\]

Это уже batched GEMM.

---

# 5. Даже LSMR можно оставить

Необязательно переходить к normal equations.

Определим оператор

\[
\mathcal A_C(X)_j
=

\sqrt{N_j}\,U_jX\alpha_j.
\]

Тогда можно оставить least-squares постановку

\[
\min_X
\left\|
\mathcal A_C(X)-E
\right\|^2 +
\lambda
\left\|
XC^\top-Z^\top
\right\|^2.
\]

Внутри solver видит \(X\) как вектор длины \(dq\), но `matvec` фактически делает операции с

\[
X\in\mathbb R^{d\times q}.
\]

То есть математически это один Krylov solve, а физически GPU выполняет matrix-matrix операции.

Это лучше, чем запускать \(q\) независимых LSMR.

---

# 6. Почему GPU здесь существенно удобнее

Текущий rank-1 solver в основном делает операции типа

\[
U_jv.
\]

Это GEMV. Для GPU GEMV обычно ограничен пропускной способностью памяти: матрица \(U_j\) читается целиком ради одного вектора.

Block solver делает

\[
U_j
\underbrace{
\begin{pmatrix}
v_1&\cdots&v_q
\end{pmatrix}
}_{V}.
\]

Асимптотика остаётся

\[
O\left(q\sum_j\operatorname{cost}(U_jv)\right),
\]

но одна и та же \(U_j\) используется сразу для \(q\) направлений.

Получаем примерно такую разницу:

| текущий solver                   | block solver                   |
| -------------------------------- | ------------------------------ |
| `U @ v`                          | `U @ V`                        |
| GEMV                             | GEMM                           |
| rank-1 Krylov                    | rank-\(q\) Krylov              |
| один \(v\)                       | \(q\) столбцов                 |
| последовательный \(k=1,\dots,r\) | \(k\to k+q\)                   |
| плохо насыщает GPU               | значительно лучше насыщает GPU |

Именно blocked Lanczos/randomized SVD используют похожий переход от vector operations к matrix operations для GPU. :chatgpt-content-reference{index="2"}

---

# 7. Ещё сильнее: вообще убрать greedy rank growth

Для GPU мне этот вариант кажется ещё интереснее.

Сразу параметризовать

\[
\boxed{
B=LR^\top,
\qquad
L\in\mathbb R^{m\times r},
\quad
R\in\mathbb R^{d\times r}.
}
\]

Тогда

\[
B^\top g_j
=

R(L^\top g_j).
\]

Objective:

\[
F(L,R)
=

\sum_jN_j
\left\|
I_j-U_jR(L^\top g_j)
\right\|^2 +
\lambda\|LR^\top-P\|_F^2.
\]

Для residual

\[
e_j=I_j-U_jR(L^\top g_j)
\]

градиенты имеют очень удобную форму:

\[
\boxed{
\nabla_LF
=

-2\sum_jN_j
g_j
(e_j^\top U_jR) +
2\lambda
\left[
L(R^\top R)-PR
\right]
}
\]

и

\[
\boxed{
\nabla_RF
=

-2\sum_jN_j
U_j^\top e_j
(g_j^\top L) +
2\lambda
\left[
R(L^\top L)-P^\top L
\right].
}
\]

Вся тяжёлая часть здесь:

\[
U_jR.
\]

Опять одна операция сразу для всех \(r\) направлений.

Дальше можно использовать Riemannian CG/L-BFGS/trust-region на fixed-rank manifold. Оптимизация непосредственно по многообразию матриц фиксированного ранга давно используется для крупномасштабных low-rank задач; существуют также rank-adaptive варианты. :chatgpt-content-reference{index="3"}

Это уже решает непосредственно

\[
\boxed{
\min_{\operatorname{rank}(B)\le r}F(B)
}
\]

вместо последовательного приближения greedy rank-1 updates.

---

# Что я бы сделал с вашим `SVD.py`

Я бы не переписывал сразу всё. Наиболее безопасная последовательность:

\[
\boxed{
\text{rank-1 greedy}
\longrightarrow
\text{block-}q\text{ greedy}
}
\]

с

\[
q=\min(r,4\text{ или }8).
\]

Архитектура станет

\[
B_k
\rightarrow Q
\rightarrow
\operatorname{TopSVD}_q(Q_\perp)
\rightarrow
[C\leftrightarrow V]
\rightarrow
B_k+CV^\top
\]

\[
\rightarrow
\text{compact SVD}
\rightarrow
\text{joint scale refit}.
\]

При этом сохраняются почти все хорошие свойства нынешнего solver:

\[
F(B_{\rm new})\le F(B)
\]

можно проверять точно;

\[
\operatorname{rank}(B_{\rm new})
\le k+q;
\]

compact QR+SVD остаётся маленьким;

joint scale refit остаётся без изменений;

rank tolerance можно заменить на gain всего блока.

И главное, наиболее дорогая операция меняется с

\[
U_jv
\]

на

\[
\boxed{U_jV}.
\]

Именно это изменение я бы считал основным для GPU.

Отдельно параллельно надо считать индекс \(j\): все \(U_jV\), residual, `pulled` и вклады в \(Q\) независимы до reduction. Если размеры \(U_j\) одинаковы, их можно хранить как батч

\[
U\in\mathbb R^{J\times n_{\rm loc}\times d}
\]

и почти весь inner loop свести к `bmm/einsum`. Если размеры разные, лучше bucket по \(n_j\) или sparse/block-sparse представление.

Для небольшого \(r\), например \(r=2\), ускорение от block-\(r\) само по себе будет ограниченным. Тогда главный GPU-параллелизм идёт по \(J\). При \(r=4\!-\!16\) сочетание **параллелизма по \(J\) + block \(V\)** уже гораздо лучше соответствует GPU.

Если цель состоит именно в переносе существующего алгоритма без сильного изменения его математического поведения, я бы выбрал **block-greedy \(C,V\)**, а не прямой \(LR^\top\)-solver. Он ближе всего к вашему нынешнему `SVD.py` и позволяет сравнить оба solver на одной и той же сетке экспериментов.
