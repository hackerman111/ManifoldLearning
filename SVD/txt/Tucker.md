Да. Для твоей задачи это можно сделать довольно чисто: **сохранить тот же квадратичный функционал, но заменить постоянный \(B\) полем локальных \(B_j\), собранных в тензор**.

Сейчас multi-index предполагает одну матрицу

\[
B\in\mathbb R^{m\times d},
\]

и локально

\[
I_j\approx U_jB^\top g_j.
\]

То есть для всех центров \(j\) используется одно и то же подпространство

\[
\operatorname{span}(B^\top)\subset\mathbb R^d.
\]

Для manifold-варианта вводишь

\[
\mathcal B\in\mathbb R^{m\times d\times J},
\qquad
B_j=\mathcal B[:,:,j].
\]

И просто заменяешь

\[
U_jB^\top g_j
\quad\longrightarrow\quad
U_jB_j^\top g_j.
\]

Получается практически тот же функционал:

\[
\boxed{
F(\mathcal B)=
\sum\_{j=1}^J
N_j
\left\|
I_j-U_jB_j^\top g_j
\right\|^2

- \lambda
  \|\mathcal B-\mathcal P\|\_F^2.
  }
  \]

Если исходный prior \(P\) один, можно просто положить

\[
P_j=P,
\qquad
\mathcal P[:,:,j]=P.
\]

То есть algebra функционала действительно почти не меняется.

### Где появляется manifold

В multi-index

\[
B_1=B_2=\cdots=B_J=B.
\]

В manifold-варианте

\[
B_j=B(x_j),
\]

поэтому локальное EDR-пространство

\[
\mathcal S_j
=
\operatorname{span}(B_j^\top)
\]

может зависеть от \(x_j\).

Например,

\[
\mathcal S_1\neq\mathcal S_2,
\]

но для близких \(x_i,x_j\) пространства должны быть близкими.

Именно это превращает фиксированное линейное EDR-пространство в поле локальных подпространств.

---

Но есть критический момент.

Если просто оптимизировать все

\[
B_1,\ldots,B_J
\]

независимо, то получится не manifold, а \(J\) независимых локальных задач.

Действительно,

\[
F(\mathcal B)
=
\sum_jF_j(B_j),
\]

и Hessian становится block-diagonal:

\[
\mathcal H_j[Z]
=
2N_jg_jg_j^\top
Z
U_j^\top U_j

- 2\lambda Z.
  \]

Никакой информации между \(B_i\) и \(B_j\) тогда нет.

Поэтому нужна **связь между третьей модой тензора**. При этом сам data-fit менять необязательно. Связь можно внести через ограничение класса допустимых \(\mathcal B\).

## Самый естественный вариант: CP/Tucker low-rank

Например CP:

\[
\boxed{
\mathcal B
=
\sum\_{q=1}^{R}
a_q\otimes v_q\otimes w_q.
}
\]

Тогда

\[
B*j
=
\sum*{q=1}^{R}
w\_{jq}a_qv_q^\top.
\]

Это очень интересная форма именно для твоего алгоритма.

Вместо текущего

\[
B=
\sum\_{q=1}^{r}\sigma_q a_qv_q^\top
\]

получаешь

\[
\boxed{
B*j=
\sum*{q=1}^{R}
w_q(x_j)a_qv_q^\top.
}
\]

Разница только в коэффициентах:

multi-index:

\[
\sigma_q=\text{const},
\]

manifold:

\[
\sigma_q\longrightarrow w_q(x).
\]

То есть глобальные rank-1 направления \(a_qv_q^\top\) остаются общими, а их вклад меняется вдоль пространства.

Это буквально тензорное обобщение твоего текущего SVD-представления.

---

### Tucker даёт ещё больше свободы

Можно взять

\[
\boxed{
\mathcal B
=
\mathcal G
\times_1 A
\times_2 V
\times_3 W.
}
\]

Для точки \(j\)

\[
B_j
=
A M_j V^\top,
\]

где

\[
M_j
=
\mathcal G\times_3 w_j.
\]

То есть вместо одной матрицы

\[
M
\]

из нашего предыдущего block-SVD алгоритма появляется поле

\[
M_1,\ldots,M_J.
\]

Получается очень красивая цепочка:

\[
\boxed{
B=AMV^\top
}
\]

для multi-index,

и

\[
\boxed{
B(x)=A\,M(x)\,V^\top
}
\]

для manifold.

А дискретно:

\[
B_j=A M_jV^\top.
\]

Сам функционал:

\[
\boxed{
F(A,V,\mathcal G,W)
=
\sum_jN_j
\left\|
I_j-U_jV M_j^\top A^\top g_j
\right\|^2

- \lambda\|\mathcal B-\mathcal P\|\_F^2.
  }
  \]

То есть форма residual least squares сохранилась.

---

## Но есть тонкость с настоящим вращением EDR-пространства

Если взять

\[
A\in\mathbb R^{m\times r},
\qquad
V\in\mathbb R^{d\times r},
\]

то

\[
B_j=AM_jV^\top
\]

всегда удовлетворяет

\[
\operatorname{span}(B_j^\top)
\subseteq
\operatorname{span}(V).
\]

Если

\[
\dim V=r
\]

и каждый \(B_j\) имеет rank \(r\), то фактически

\[
\operatorname{span}(B_j^\top)
=
\operatorname{span}(V)
\]

для всех \(j\).

То есть manifold **вообще не появился**. Меняются коэффициенты внутри одного и того же пространства.

Это важное ограничение.

Чтобы локальное EDR-пространство действительно вращалось, надо взять глобальный envelope большей размерности:

\[
V\in\mathbb R^{d\times R},
\qquad
R>r.
\]

И

\[
M_j\in\mathbb R^{m\times R},
\qquad
\operatorname{rank}(M_j)\le r.
\]

Тогда

\[
B_j=M_jV^\top
\]

может иметь разные row spaces внутри

\[
\operatorname{span}(V),
\]

например

\[
\mathcal S_j
\subset\operatorname{span}(V),
\qquad
\dim\mathcal S_j=r,
\]

но

\[
\mathcal S_i\neq\mathcal S_j.
\]

Например при

\[
d=100,\qquad r=2,\qquad R=8
\]

мы ищем 8-мерное глобальное пространство, внутри которого в каждой точке находится своя 2-мерная касательная плоскость.

Это уже вполне естественная модель manifold.

---

## Тогда возникает очень интересное обобщение твоего SVD solver

Сейчас:

\[
B
=
\sum\_{q=1}^{r}
\sigma_q a_qv_q^\top.
\]

Можно перейти к

\[
\boxed{
B*j
=
\sum*{q=1}^{R}
c\_{jq}a_qv_q^\top
}
\]

или более общей форме

\[
\boxed{
B_j=A M_jV^\top.
}
\]

И искать **все \(j\) одновременно**.

Например:

\[
\mathcal B^{(0)}
\overset{\text{tensor SVD/HOSVD}}{\longrightarrow}
A_0,V_0,W_0,
\]

затем итеративно:

\[
M*j
\leftarrow
\arg\min*{M_j}
F,
\]

\[
A\leftarrow\text{joint update},
\]

\[
V\leftarrow\text{joint update},
\]

\[
W\leftarrow\text{joint update}.
\]

Причём операции по \(j\) хорошо батчатся на GPU.

---

### Что я считаю самым естественным математическим вариантом

Не делать буквально

\[
\mathcal B\in\mathbb R^{m\times d\times J}
\]

со свободными \(Jmd\) параметрами.

Лучше:

\[
\boxed{
B_j=C_jV^\top,
\qquad
V\in\mathbb R^{d\times R},
\quad
C_j\in\mathbb R^{m\times R},
\quad
\operatorname{rank}(C_j)=r,
}
\]

где

\[
R>r.
\]

А весь массив \(C_j\) тоже сделать low-rank по \(j\), например

\[
C*j
=
\sum*{\ell=1}^{s}w\_{j\ell}C^{(\ell)}.
\]

Тогда

\[
\boxed{
B(x)
=
\sum*{\ell=1}^{s}
w*\ell(x)\,
C^{(\ell)}V^\top.
}
\]

Это сохраняет твою исходную local-linear/ADP функцию потерь, но заменяет

\[
\text{одно фиксированное EDR-пространство}
\]

на

\[
\text{низкоразмерное поле локальных EDR-пространств}.
\]

И это уже действительно можно рассматривать как прямое tensor-обобщение multi-index модели на manifold. Главное: **тензор сам по себе недостаточен; третья мода должна кодировать зависимость локального подпространства от положения, а размер глобального envelope должен быть больше локальной размерности \(r\)**.
