## Hristache–Juditsky–Polzehl–Spokoiny, 2001: EDR / multi-index

Генеративная модель:

\[
Y_i=g(\theta_1^\top X_i,\ldots,\theta_m^\top X_i)+\varepsilon_i,
\qquad
\varepsilon_i\sim N(0,\sigma^2).
\]

Координаты \(X_i\) независимы и генерируются так, что

\[
\frac{X_{ij}+1}{2}\sim \mathrm{Beta}(1,\tau).
\]

При \(\tau=1\) это равномерное распределение на \([-1,1]\); дополнительно проверяются \(\tau=0.75\) и \(1.5\), то есть асимметричный дизайн. :chatgpt-content-reference{index="0"}

Для **\(m=1\)**:

\[
g(u)=u\sin(\sqrt5\,u),\qquad
\theta=\frac{(1,2,0,\ldots,0)}{\sqrt5}.
\]

Основная сетка:

\[
(d,n)=(3,200),(4,200),(6,200),
(10,100),(10,200),(10,400),(10,800),
\]

при \(\sigma=0.1,\tau=1\). Для каждой точки делают **250 независимых Monte Carlo реализаций**. :chatgpt-content-reference{index="1"}

Для **\(m=2\)**:

\[
g(u_1,u_2)=(u_1^3+u_2)(u_1-u_2^3),
\]

\[
\theta_1=\frac{(1,1,0,\ldots)}{\sqrt2},
\qquad
\theta_2=\frac{(1,-1,0,\ldots)}{\sqrt2}.
\]

Протокол разбит на несколько sweep:

- зависимость от \(n\): \(d=10,\ n=200,400,800,\ \sigma=0.1,\tau=1\);
- зависимость от \(d\): \((d,n)=(10,400),(10,800),(20,800),(50,800)\);
- зависимость от шума: \(d=10,n=400,\ \sigma\in\{0.05,0.1,0.2\}\);
- зависимость от распределения \(X\): \(d=10,n=400,\sigma=0.1,\ \tau\in\{0.75,1,1.5\}\).

Обычно делают 250 реализаций; для \(d>10\) в опубликованном benchmark использовалось 100. :chatgpt-content-reference{index="2"}

Для **\(m=3\)**:

\[
g(u_1,u_2,u_3)
=(u_1^3+u_2)(u_1-u_2^3)+u_3,
\]

с ортонормированными направлениями

\[
\theta_1=\frac{(1,1,1,0,\ldots)}{\sqrt3},\quad
\theta_2=\frac{(1,-1,0,\ldots)}{\sqrt2},\quad
\theta_3=\frac{(1,1,-2,0,\ldots)}{\sqrt6}.
\]

Используют

\[
d=10,\qquad n=800,\qquad \sigma=0.1,
\qquad \tau\in\{0.75,1,1.5\},
\]

по 250 реализаций. :chatgpt-content-reference{index="3"}

Для каждого режима записывают средний loss и interquartile range после **1, 2, 4, 8 и последней итерации**. В \(m=1,2\) также сравнивают с ADE, SIR II и PHD. :chatgpt-content-reference{index="4"}

Сам алгоритм во всех этих экспериментах запускают практически с фиксированными параметрами:

\[
\rho_1=1,\quad
\rho_{\min}=n^{-1/3},\quad
a_\rho=e^{-1/6},
\]

\[
h_1=n^{-1/(4\vee d)},\quad
h_{\max}=2\sqrt d,\quad
a_h=e^{1/[2(4\vee d)]},
\]
