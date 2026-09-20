## 1. Строго доказуемый результат

В `sin-mul.tex` есть структура, которая позволяет построить низкоранговую реализацию ADP без приближения исходной finite-sketch задачи.

Для центра (j) обозначим через

[
\mathcal N_j={i:w_{ij}>0},\qquad k_j=|\mathcal N_j|
]

его ненулевую окрестность. Для Epanechnikov kernel она разрежена из-за компактного носителя.

Пусть

[
a_{ji}=\frac{w_{ij}}{\sum_{\ell\in\mathcal N_j}w_{\ell j}},
\qquad
\bar X_j=\sum_{i\in\mathcal N_j}a_{ji}X_i,
]

и введём матрицу

[
R_j
===

\operatorname{diag}(\sqrt{w_j})
\begin{bmatrix}
(X_{i_1}-\bar X_j)^\top\
\vdots\
(X_{i_{k_j}}-\bar X_j)^\top
\end{bmatrix}
\in\mathbb R^{k_j\times d}.
]

Тогда локальная матрица второго момента из алгоритма равна

[
C_j
===

\sum_i
w_{ij}(X_i-\bar X_j)(X_i-\bar X_j)^\top
=======================================

R_j^\top R_j.
]

Для направлений

[
\Phi_j=
\begin{pmatrix}
\phi_{j1}^\top\
\vdots\
\phi_{jp}^\top
\end{pmatrix}
\in\mathbb R^{p\times d},
\qquad p=n_\Phi,
]

формулы из файла можно записать как

[
\boxed{
U_j=\Phi_j C_j
}
]

и

[
\boxed{
I_j=\Phi_j b_j,
\qquad
b_j=R_j^\top y_j^{(w)}.
}
]

Это непосредственно следует из определений (I_{j,\phi}) и (U_{j,\phi}) в файле [sin-mul.tex](sandbox:/mnt/data/sin-mul.tex).

Теперь положим

[
Q_j=\Phi_jR_j^\top\in\mathbb R^{p\times k_j}.
]

Получаем точную факторизацию

[
\boxed{
U_j=Q_jR_j,
\qquad
I_j=Q_jy_j^{(w)}.
}
]

Матрица (U_j\in\mathbb R^{p\times d}) больше не нужна вообще.

Есть ещё более сильный факт:

[
R_j^\top\sqrt{w_j}
==================

# \sum_iw_{ij}(X_i-\bar X_j)

0.

]

Следовательно,

[
\boxed{
\operatorname{rank}R_j
======================

\operatorname{rank}C_j
\le k_j-1.
}
]

Именно (k_j-1), а не (d), представляет истинную локальную размерность ADP.

Это центральная низкоранговая структура, которую исходная запись скрывает.

---

# 2. Независимые семейства подходов

Я рассматривал пять разных механизмов.

| Семейство                              | Основа                            | Результат                                            | Оставшийся пробел                                         |
| -------------------------------------- | --------------------------------- | ---------------------------------------------------- | --------------------------------------------------------- |
| Локальная точная факторизация          | (\operatorname{rank}C_j\le k_j-1) | не хранить (U_j), сохранить исходный objective точно | matvec может быть дороже явного (U_j), если (k_j\gg p)    |
| Полная локальная row-space задача      | исключить случайный sketch        | задача живёт в (r_j\le k_j-1) координатах            | меняется finite-sketch objective                          |
| Низкоранговая структура (T_k)          | rank-1/rank-(m) update к (I)      | не строить (d\times d) tensor, сильно ускорить веса  | первая изотропная итерация всё ещё дорогая                |
| Variable projection / Grassmann        | исключить (f_j) аналитически      | меньше неизвестных, корректная геометрия multi-index | нет универсального доказательства меньшего числа итераций |
| Глобальный randomized low-rank Hessian | спектральный preconditioner       | сильное ускорение Krylov при spectral decay          | без spectral decay выигрыша гарантировать нельзя          |

Глобальный Nyström для (W), CP/Tucker для всего тензора (U) и обычный randomized SVD (U_j) я бы не выбирал. Для адаптивного компактного kernel нет предположения о глобальном спектральном затухании (W), а локальный ранг (U_j) уже известен аналитически. Randomized SVD в таком случае сначала вычисляет объект, структуру которого можно получить без его построения.

---

# 3. Основной алгоритм: Factorized Low-Rank ADP

Я бы сделал именно эту версию основной, поскольку она **не меняет finite-direction objective из файла**.

## 3.1. Веса без матрицы (T_k)

### Single-index

В файле

[
T_k^2
=====

h_k^{-2}
\left(
\alpha_k^2I+\beta_k\beta_k^\top
\right).
]

Для

[
\Delta_{ij}=X_i-x_j
]

имеем точно

[
\boxed{
|T_k\Delta_{ij}|^2
==================

h_k^{-2}
\left[
\alpha_k^2|\Delta_{ij}|^2 +
(\beta_k^\top\Delta_{ij})^2
\right].
}
]

Не нужны ни (T_k), ни (T_k^2).

Сначала вычисляется один вектор

[
z_i=\beta_k^\top X_i,
\qquad
z_j^c=\beta_k^\top x_j.
]

Стоимость:

[
O((n+J)d).
]

После этого

[
\beta_k^\top(X_i-x_j)=z_i-z_j^c
]

стоит (O(1)) на пару.

### Точное предварительное отсечение

Kernel из файла ненулевой только при

[
|T_k\Delta_{ij}|^2<1.
]

Но

[
|T_k\Delta_{ij}|^2
\ge
h_k^{-2}(z_i-z_j^c)^2.
]

Поэтому

[
\boxed{
|z_i-z_j^c|\ge h_k
\quad\Longrightarrow\quad
w_{ij}=0.
}
]

Это позволяет:

1. отсортировать (z_i) за (O(n\log n));
2. для каждого центра двумя binary search найти интервал
   [
   z_i\in(z_j^c-h_k,z_j^c+h_k);
   ]
3. полный (d)-мерный квадрат расстояния считать только для этих кандидатов.

Если число кандидатов равно (C_k), стоимость становится

[
\boxed{
O((n+J)d+n\log n+J\log n+C_kd)
}
]

вместо

[
O(Jnd).
]

Это **точный filter**, false negatives отсутствуют.

---

## 3.2. Multi-index

Пусть

[
P_k\in\mathbb R^{m\times d},
\qquad
P_kP_k^\top=I_m.
]

Из файла:

[
T_k^2
=====

h_k^{-2}
\alpha_k^2(I-P_k^\top P_k) +
h_k^{-2}P_k^\top\Lambda_kP_k.
]

Поэтому

[
\boxed{
|T_k\Delta|^2
=============

h_k^{-2}
\left[
\alpha_k^2
\left(
|\Delta|^2-|P_k\Delta|^2
\right) +
(P_k\Delta)^\top\Lambda_k(P_k\Delta)
\right].
}
]

Сначала вычисляем

[
Z=XP_k^\top\in\mathbb R^{n\times m},
\qquad
Z_c=CP_k^\top\in\mathbb R^{J\times m}.
]

Стоимость:

[
O((n+J)dm).
]

После этого активная часть расстояния стоит (O(m)), а не (O(dm)).

Ещё раз работает точное отсечение:

[
(P_k\Delta)^\top\Lambda_k(P_k\Delta)\ge h_k^2
\quad\Longrightarrow\quad
w_{ij}=0.
]

Значит, кандидатов можно искать в (m)-мерном пространстве

[
\Lambda_k^{1/2}P_kX_i
]

обычным radius search радиуса (h_k).

При (m=2,3,5) это существенно лучше поиска в (d=1000).

Это похоже по вычислительной идее на использование разреженной локализации в низкоразмерном projected space в недавней SMAVE, хотя criterion и статистическая процедура там другие. ([arXiv][1])

---

# 4. Низкоранговые статистики без (U_j)

После определения соседей не надо создавать

[
X_i-\bar X_j
]

для всех (i,j).

Для произвольного (v\in\mathbb R^d):

[
R_jv
====

\sqrt{w_j}\odot
\left(
X_{\mathcal N_j}v
-----------------

a_j^\top X_{\mathcal N_j}v,\mathbf 1
\right).
]

Следовательно,

[
R_jv
]

вычисляется за

[
O(k_jd)
]

без хранения (R_j).

Сопряжённая операция тоже не требует (R_j). Если

[
c=\sqrt{w_j}\odot u,
]

то

[
\boxed{
R_j^\top u
==========

X_{\mathcal N_j}^\top
\left[
c-a_j(\mathbf1^\top c)
\right].
}
]

То есть достаточно хранить:

[
\mathcal N_j,\quad w_j,\quad Q_j.
]

Ни (\bar X_j\in\mathbb R^d), ни (R_j\in\mathbb R^{k_j\times d}), ни (U_j\in\mathbb R^{p\times d}) постоянно хранить не обязательно.

---

# 5. Single-index solver

Исходная задача:

[
\min_{\beta,{f_j}}
\sum_j
|I_j-f_jU_j\beta|^2.
]

После точной факторизации:

[
\boxed{
\min_{\beta,{f_j}}
\sum_j
\left|
I_j-f_jQ_jR_j\beta
\right|^2.
}
]

## Обновление (f_j)

Сначала

[
t_j=R_j\beta\in\mathbb R^{k_j},
\qquad
u_j=Q_jt_j\in\mathbb R^p.
]

Затем

[
\boxed{
f_j=
\frac{I_j^\top u_j}
{u_j^\top u_j}.
}
]

Стоимость всех обновлений:

[
O(Ed+pE),
\qquad
E=\sum_jk_j.
]

При почти нулевом (u_j) локальная производная неидентифицируема. Такой центр нужно либо пропустить, либо использовать

[
u_j^\top u_j+\eta.
]

---

## Обновление (\beta)

Не следует строить формулу из файла

[
\sum_jf_j^2U_j^\top U_j+\lambda I.
]

Вместо неё задаём LinearOperator

[
A\beta=
\begin{pmatrix}
f_1Q_1R_1\beta\
\vdots\
f_JQ_JR_J\beta
\end{pmatrix}
]

и

[
A^\top
\begin{pmatrix}
v_1\
\vdots\
v_J
\end{pmatrix}
=============

\sum_j
f_jR_j^\top Q_j^\top v_j.
]

Ridge записывается стандартным augmented system:

[
\min_\beta
\left|
\begin{pmatrix}
A\
\sqrt\lambda I
\end{pmatrix}
\beta
-----

\begin{pmatrix}
I\
\sqrt\lambda\beta_{\rm init}
\end{pmatrix}
\right|^2.
]

Для него я бы использовал **LSMR**.

LSMR предназначен именно для sparse/fast linear operators и использует Golub-Kahan bidiagonalization, поэтому матрицу (A) формировать не требуется. ([SIAM][2])

Это также устраняет явное формирование normal matrix. Для обычных нормальных уравнений

[
\kappa(A^\top A)=\kappa(A)^2.
]

---

# 6. Ещё более низкоранговая версия: полное локальное row space

Здесь появляется более интересная модификация самого ADP.

Случайные направления в файле измеряют один и тот же вектор

[
r_j=b_j-f_jC_j\beta
]

через

[
\Phi_jr_j.
]

Но

[
r_j\in\operatorname{range}(R_j^\top)
]

и это пространство имеет размерность не более (k_j-1).

Возьмём thin SVD

[
R_j
===

A_j\Sigma_jV_j^\top,
\qquad
r_j=\operatorname{rank}R_j.
]

Определим

[
D_j=\Sigma_jA_j^\top
\in\mathbb R^{r_j\times k_j}.
]

Тогда

[
c_j=D_jy_j^{(w)},
]

[
L_j=D_jR_j.
]

Причём

[
b_j=V_jc_j,
]

[
C_jv=V_jL_jv.
]

Поскольку (V_j^\top V_j=I),

[
\boxed{
|b_j-C_jv|^2
============

|c_j-L_jv|^2.
}
]

Это **точное равенство**.

Следовательно, полная локальная derivative equation

[
b_j\approx f_jC_j\beta
]

может быть решена в

[
r_j\le k_j-1
]

измерениях без единого (d\times d) объекта.

При этом (V_j) хранить тоже не нужно:

[
L_jv
====

\Sigma_jA_j^\top(R_jv),
]

[
L_j^\top z
==========

R_j^\top A_j\Sigma_jz.
]

Достаточно вычислить спектральное разложение маленькой матрицы

[
R_jR_j^\top\in\mathbb R^{k_j\times k_j}.
]

### Что меняется

Это уже не буквально finite-(n_\Phi) objective из файла. Вместо

[
|\Phi_jr_j|^2
]

используется

[
|r_j|^2.
]

То есть исчезает Monte Carlo distortion от случайных направлений.

Я считаю эту версию математически более чистой, но не называю её эквивалентной исходному estimator.

---

# 7. Когда случайные направления вообще избыточны

Есть ещё один точный результат.

Пусть

[
S_j=\Phi_jV_j\in\mathbb R^{p\times r_j}.
]

Тогда

[
I_j=S_jc_j,
\qquad
U_j=S_jL_j.
]

Если

[
p\ge r_j
]

и (S_j) имеет полный столбцовый ранг, то

[
c_j=S_j^\dagger I_j,
\qquad
L_j=S_j^\dagger U_j.
]

Для непрерывного распределения направлений и положительно определённой covariance это происходит с вероятностью (1).

То есть при

[
n_\Phi\ge r_j
]

случайные направления не уменьшают информационную размерность локальной задачи. Они лишь задают случайную метрику

[
S_j^\top S_j
]

на уже (r_j)-мерном пространстве.

Это сильный аргумент в пользу row-space ADP при небольших окрестностях.

---

# 8. Multi-index solver

Пусть

[
B\in\mathbb R^{m\times d}.
]

Finite-sketch версия после факторизации имеет вид

[
\boxed{
\min_{B,{\ell_j}}
\sum_j
|I_j-Q_jR_jB^\top\ell_j|^2 +
\lambda|B-B_0|_F^2.
}
]

## Локальное обновление

Положим

[
A_j=Q_jR_jB^\top
\in\mathbb R^{p\times m}.
]

Тогда

[
\ell_j
======

\arg\min_\ell|I_j-A_j\ell|^2.
]

Не следует вычислять

[
(A_j^\top A_j)^{-1}A_j^\top I_j
]

так, как написано в файле.

При (m) малом нужно использовать QR или SVD:

[
A_j=Q_j^{(A)}R_j^{(A)}.
]

Это избегает квадрата числа обусловленности.

---

## Глобальное обновление (B)

Положим

[
X=B^\top\in\mathbb R^{d\times m}.
]

Каждый residual равен

[
I_j-Q_jR_jX\ell_j.
]

Forward operation для LSMR:

[
X
\longmapsto
Q_jR_j(X\ell_j).
]

Сначала

[
v=X\ell_j
]

за (O(dm)), затем

[
R_jv
]

за (O(k_jd)).

То есть не требуется операция порядка

[
O(k_jdm).
]

Для одного центра:

[
\boxed{
O(d(m+k_j)+pk_j).
}
]

Adjoint:

[
z_j
\longmapsto
\left(R_j^\top Q_j^\top z_j\right)\ell_j^\top.
]

Это даёт полноценный matrix-free LSMR над (dm) неизвестными.

---

# 9. Матрицу (\mathcal J) в multi-index собирать нельзя

В файле:

[
\mathcal J
==========

B^\top
\left(
\sum_j\ell_j\ell_j^\top
\right)
B.
]

Обозначим

[
M=\sum_j\ell_j\ell_j^\top
\in\mathbb R^{m\times m}.
]

Тогда

[
\mathcal J=B^\top MB.
]

Её ранг удовлетворяет

[
\operatorname{rank}\mathcal J\le m.
]

Следовательно, (d\times d) матрицу строить бессмысленно.

Вычисляем

[
C=M^{1/2}B
\in\mathbb R^{m\times d}.
]

Тогда

[
\boxed{
\mathcal J=C^\top C.
}
]

Нужные собственные векторы представляют правые singular vectors маленькой матрицы (C).

Стоимость:

[
O(dm^2+m^3)
]

вместо хранения

[
O(d^2).
]

Если строки (B) уже ортонормированы, ещё проще. При

[
M=Q\Lambda Q^\top
]

EDR-базис равен

[
P=Q^\top B.
]

Требуется eigendecomposition только (m\times m) матрицы.

---

# 10. (T_k) нельзя строить и для генерации направлений

### Single-index

Вместо

[
g\sim N(0,T_k^2)
]

можно точно генерировать

[
\boxed{
g=\alpha_kz+\beta_k\xi,
}
]

где

[
z\sim N(0,I_d),
\qquad
\xi\sim N(0,1)
]

независимы.

Тогда

[
\operatorname{Cov}g
===================

\alpha_k^2I+\beta_k\beta_k^\top.
]

Множитель (h_k^{-1}) вообще не нужен, поскольку после этого файл нормирует

[
\phi=\frac{g}{|g|}.
]

### Multi-index

Точно:

[
\boxed{
g
=

\alpha_k(I-P^\top P)z +
P^\top\Lambda^{1/2}\eta,
}
]

[
z\sim N(0,I_d),
\qquad
\eta\sim N(0,I_m).
]

Получаем именно

[
\operatorname{Cov}g
===================

\alpha_k^2(I-P^\top P) +
P^\top\Lambda P.
]

Стоимость одной генерации:

[
O(dm)
]

и память

[
O(d+m),
]

а не (O(d^2)).

В row-space версии этот шаг исчезает целиком.

---

# 11. Variable projection как второй solver

После локальной низкоранговой компрессии single-index objective можно профилировать.

Для

[
u_j(\beta)=L_j\beta
]

имеем

[
f_j^*(\beta)
============

\frac{c_j^\top u_j}
{u_j^\top u_j}.
]

Следовательно,

[
F(\beta)
========

\sum_j
\left[
|c_j|^2
-------

\frac{(c_j^\top L_j\beta)^2}
{|L_j\beta|^2}
\right],
\qquad
|\beta|=1.
]

Остаётся только (d-1) степеней свободы вместо

[
d+J.
]

По envelope theorem градиент вычисляется без дифференцирования (f_j^*(\beta)):

[
\nabla F(\beta)
===============

-2
\sum_j
f_j^*
L_j^\top
(c_j-f_j^*L_j\beta).
]

На сфере:

[
\operatorname{grad}F
====================

(I-\beta\beta^\top)\nabla F.
]

Это классическая структура separable nonlinear least squares, для которой variable projection был разработан Golub и Pereyra. ([SIAM][3])

Для multi-index аналогично исключаются все (\ell_j), после чего функция зависит только от EDR-подпространства. Объект естественно принадлежит Grassmann manifold; алгоритмы для Stiefel/Grassmann с ортогональными ограничениями подробно разработаны Edelman, Arias и Smith. ([SIAM][4])

Недавний препринт Pautrel и Portier применяет похожую комбинацию local profiling, Stiefel geometry и stochastic optimization к SDR/MAVE и сообщает существенное вычислительное снижение стоимости на их задачах. Это подтверждение практической жизнеспособности механизма, но не доказательство ускорения именно нашего ADP objective. ([arXiv][1])

Поэтому я бы оставил два solver backend:

[
\boxed{\text{ALS + matrix-free LSMR}}
]

как надёжную базу и

[
\boxed{\text{VarPro + Riemannian L-BFGS}}
]

как более агрессивную версию.

Универсального доказательства, что VarPro потребует меньше итераций, нет.

---

# 12. Низкоранговый preconditioner

Если LSMR или CG всё ещё делает много итераций, возникает ещё один настоящий low-rank объект.

Пусть ridge subproblem имеет Hessian

[
H=A^\top A+\lambda I
]

и

[
A=U\Sigma V^\top.
]

Возьмём первые (r) right singular vectors и определим

[
M
=

\lambda I+
V_r\Sigma_r^2V_r^\top.
]

В точном SVD-базисе собственные значения (M^{-1}H):

[
1,\ldots,1,
\quad
1+\frac{\sigma_{r+1}^2}{\lambda},
\ldots
]

поэтому

[
\boxed{
\kappa(M^{-1}H)
\le
1+\frac{\sigma_{r+1}^2}{\lambda}.
}
]

Если спектр после (r) быстро падает, число Krylov iterations резко уменьшается.

Это уже условный результат: без

[
\sigma_{r+1}^2\ll\lambda
]

низкоранговый preconditioner бессмыслен.

Если спектрального затухания нет, более общий randomized sketch-and-precondition подход можно строить по типу LSRN. LSRN использует Gaussian sketch, затем LSQR/Chebyshev на предобусловленной системе и даёт вероятностный контроль обусловленности. ([arXiv][5]) Для ADP я бы включал его только после профилирования: оператор меняется между outer/AO итерациями, поэтому построение нового preconditioner может оказаться дороже сэкономленных Krylov steps.

---

# 13. Initialization тоже надо переписать

Самое слабое вычислительное место файла до основного ADP:

[
\begin{pmatrix}
1\
X_i-x_j
\end{pmatrix}
\begin{pmatrix}
1\
X_i-x_j
\end{pmatrix}^{!T}
\in\mathbb R^{(d+1)\times(d+1)}
]

с последующим обращением.

Этого делать не надо.

Локальная задача

[
\min_{a,g}
\sum_i
w_i
\left[
Y_i-a-g^\top(X_i-x_j)
\right]^2
]

точно эквивалентна

[
\boxed{
\min_g
\left|
\sqrt w\odot
\left[
(Y-\bar Y)-Zg
\right]
\right|^2,
}
]

где

[
Z_i=X_i-\bar X.
]

После нахождения (g)

[
a=\bar Y-g^\top(\bar X-x_j).
]

Значит, каждый local linear gradient можно получить LSMR/LSQR непосредственно из матрицы размера

[
k_{\rm lin}\times d
]

без формирования Gram matrix.

Один matvec:

[
O(k_{\rm lin}d).
]

Формирование исходной normal matrix:

[
O(k_{\rm lin}d^2),
]

после чего ещё требуется факторизация порядка (O(d^3)).

Инициализационную

[
\mathcal J=\sum_jg_jg_j^\top
]

также не надо собирать. Если

[
G=
\begin{pmatrix}
g_1^\top\
\vdots\
g_J^\top
\end{pmatrix},
]

то

[
\mathcal J=G^\top G.
]

Нужны только первые (m) right singular vectors (G), которые можно получить Lanczos/partial SVD без (d\times d) матрицы.

Здесь остаётся настоящий пробел: локальная initialization covariance может иметь полный ранг (d). Без дополнительного предположения о данных **точного low-rank ускорения самого локального regression solve доказать нельзя**. Можно только убрать Gram matrix и использовать iterative LS.

---

# 14. Итоговая сложность

Пусть

[
p=n_\Phi,\qquad
\bar k=E/J,
]

а (C_k) представляет число пар после exact active-space pruning.

### Исходная реализация с явными (U_j)

Хранение:

[
\boxed{
\Theta(Jpd)
}
]

только для (U).

Статистики:

[
\Theta(Epd).
]

Прямое вычисление adaptive distances:

[
\Theta(Jnd)
]

для single-index и до

[
\Theta(Jndm)
]

при наивной multi-index реализации.

Если буквально реализовать inverse из single-index части файла, появляются ещё

[
O(d^2)
]

памяти и дорогая Gram matrix.

### Factorized LR-ADP

Хранение локальной структуры:

[
\boxed{
O(Ep+E)
}
]

плюс исходные (X).

Матрицы (d\times d) отсутствуют.

Adaptive weights, single-index:

[
\boxed{
O((n+J)d+C_kd)
}
]

с логарифмическими расходами на interval search.

Multi-index:

[
\boxed{
O((n+J)dm+C_kd)
}
]

плюс low-dimensional range search.

Один single-index solver matvec:

[
\boxed{
O(Ed+Ep).
}
]

Multi-index global matvec:

[
\boxed{
O(Jdm+Ed+Ep).
}
]

Если

[
\bar k,p,m=O(1),
\qquad
C_k=O(J),
]

то после initialization основная итерация становится линейной по (d):

[
O((n+J)d),
]

вместо all-pairs

[
O(Jnd).
]

Именно условие

[
C_k\ll Jn
]

нужно для строгого asymptotic ускорения weight stage. Без него этого ускорения нельзя утверждать.

---

# 15. Пример памяти

Возьмём для иллюстрации

[
d=1000,\quad
J=2000,\quad
p=10,\quad
\bar k=20.
]

Явный (U) содержит

[
2000\cdot10\cdot1000
====================

20,000,000
]

чисел.

Для `float32`:

[
80\text{ MB}.
]

Фактор (Q):

[
2000\cdot10\cdot20
==================

400,000
]

чисел:

[
1.6\text{ MB}.
]

Сокращение только этого объекта:

[
\boxed{50\times}.
]

Если также хранились directions (\Phi_j), это ещё примерно (80) MB, а в factorized варианте после построения (Q_j) они больше не нужны.

---

# 16. Численная устойчивость

Есть несколько мест, где низкий ранг легко превратить в нестабильную реализацию.

**Центрирование.** Лучше глобально сдвинуть (X), а затем вычислять локальные centered quantities. Для (Y) тоже использовать

[
Y_i-\bar Y_j,
]

поскольку

[
\sum_iw_{ij}(X_i-\bar X_j)=0
]

и математический результат не меняется.

**Локальный row-space factor.** (R_jR_j^\top) заведомо сингулярна как минимум в одном направлении. Поэтому Cholesky без rank handling применять нельзя. Нужен `eigh` или SVD маленькой (k_j\times k_j) матрицы.

**Численный ранг.** Удаление малых singular values меняет задачу. Если

[
\sigma_{r+1}(R_j)\le\tau,
]

то для truncated (C_{j,r})

[
|C_j-C_{j,r}|_2
===============

\sigma_{r+1}^2
\le\tau^2,
]

а для (b_j)

[
|b_j-b_{j,r}|
\le
\tau|y_j^{(w)}|.
]

То есть агрессивный truncation допустим только как явно контролируемое приближение.

**Multi-index local LS.** Если

[
\operatorname{rank}(Q_jR_jB^\top)<m,
]

все (m) компонент (\ell_j) локально неидентифицируемы. Нужно pseudoinverse/SVD либо ridge, а не обычный inverse.

**Kernel boundary.** При вычислении расстояний через Gram identity

[
|x-y|^2
=======

|x|^2+|y|^2-2x^\top y
]

для близких точек возможна cancellation. Для кандидатов около границы

[
q_{ij}\approx1
]

разумно пересчитать (|x-y|^2) напрямую.

---

# 17. Ошибки в самом `sin-mul.tex`, которые надо исправить до реализации

В single-index разделе после определения objective (f_j) представляет один скаляр на центр, но текст говорит о

[
(f_j)_{j\in J}\in\mathbb R^{n_\Phi},
]

что размерностно неверно.

Ещё существеннее: основной single-index objective записан без ridge, но следующий closed-form update содержит

[
\lambda I,\qquad \lambda\beta_{\rm init}.
]

Это две разные оптимизационные задачи. Нужно выбрать одну.

В конце multi-index structure-adaptive section снова появляется выражение вида

[
I_j-f_jU_j\beta,
]

хотя правильная multi-index модель, введённая выше, имеет вид

[
\boxed{
I_j-U_jP^\top f_j.
}
]

Это почти наверняка остаток single-index текста.

Есть также смешение двух трактовок (P): как (m\times d) coordinate map и как (d\times d) orthogonal projector. Для вычислительного алгоритма надо зафиксировать

[
P\in\mathbb R^{m\times d},
\qquad
PP^\top=I_m.
]

После этого все приведённые выше формулы размерностно согласованы.

---

# 18. Какую версию я считаю основной

Я бы строил библиотеку вокруг следующей схемы:

[
\boxed{
\begin{array}{c}
\text{low-rank }T_k
[2mm]
\downarrow
\
\text{exact active-space candidate pruning}
[2mm]
\downarrow
\
\text{sparse neighbor lists}
[2mm]
\downarrow
\
R_j\text{ как implicit operator}
[2mm]
\downarrow
\
Q_j=\Phi_jR_j^\top,\quad U_j\text{ не строится}
[2mm]
\downarrow
\
\text{ALS + matrix-free LSMR}
[2mm]
\downarrow
\
\mathcal J\text{ через }m\times m\text{ factor}
\end{array}
}
]

Это **точно сохраняет исходную finite-random-direction задачу**, кроме допуска linear solver.

Затем как отдельный эксперимент я бы добавил

[
\boxed{
\text{row-space ADP}
}
]

с полным устранением (\Phi_j). У него локальная размерность

[
r_j\le k_j-1,
]

и это, на мой взгляд, наиболее содержательная новая версия алгоритма.

Универсальное утверждение «новая реализация одновременно быстрее и требует меньше памяти для любого (n,d,p,k_j)» доказать нельзя. Если (p\ll k_j), уже материализованный (U_j) умножается на вектор за (O(pd)), тогда как factorized representation требует (O(k_jd)). Это фундаментальный time-memory tradeoff. Строго доказан более сильный и корректный результат: **новая реализация устраняет все (d^2)-объекты, сокращает локальную память с (Jpd) до (Ep), точно ускоряет построение adaptive weights при (C_k\ll Jn), а в режиме (k_j=O(p)) сохраняет или улучшает порядок стоимости Krylov-операций**.

Из исследованных направлений именно комбинация low-rank tensor + exact projected pruning + implicit local covariance + LSMR имеет меньше всего недоказанных предположений. Variable projection и spectral low-rank preconditioning стоит накладывать уже поверх неё, а не использовать вместо неё. ([SIAM][2])

[1]: https://arxiv.org/abs/2606.00413 "[2606.00413] Riemannian Stochastic Optimization for Sufficient Dimension Reduction"
[2]: https://epubs.siam.org/doi/10.1137/10079687X "LSMR: An Iterative Algorithm for Sparse Least-Squares Problems | SIAM Journal on Scientific Computing"
[3]: https://epubs.siam.org/doi/10.1137/0710036 "The Differentiation of Pseudo-Inverses and Nonlinear Least Squares Problems Whose Variables Separate | SIAM Journal on Numerical Analysis"
[4]: https://epubs.siam.org/doi/10.1137/S0895479895290954 "The Geometry of Algorithms with Orthogonality Constraints | SIAM Journal on Matrix Analysis and Applications"
[5]: https://arxiv.org/abs/1109.5981 "LSRN: A Parallel Iterative Solver for Strongly Over- or Under-Determined Systems"
