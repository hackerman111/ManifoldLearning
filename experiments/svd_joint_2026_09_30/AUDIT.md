# Математика и независимые численные эталоны

Класс APPROXIMATE: новый оптимизатор того же fixed-g функционала SVD.
LSMR/HPAO использует другой proximal шаг; здесь сравнение именно с SVD greedy.

Обозначим Z=0 для matrix и Z=P для correction, Y=I-U Z.T g, T=P-Z.
Оптимизируется F(X)=sum_j mass_j||Y_j-U_j X.T g_j||²+lambda||X-T||²,
X=A M V.T, A.T A=V.T V=Id, rank(X)<=r. Все матрицы ранга <=r имеют
такое представление (с дополнением ортонормированных базисов).
Допускается потеря ранга M; это замкнутое ограничение rank<=r, а не обещание
оставаться на открытом многообразии точного ранга r.

alpha_j=g_j A, W_j=U_j V. Предсказание равно W_j M.T alpha_j.T.
Столбец (i,k) дизайна для row-major vec(M) равен alpha_ji W_jpk.
Все r² столбцов, включая i!=k, присутствуют. Ridge:
||A M V.T-T||²=||M-A.T T V||²+||T||²-||A.T T V||².
Последние два члена постоянны по M, но меняются при движении A,V:
line search считает исходную полную норму, а не один core penalty.
Augmented QR блоков [sqrt(mass) D, sqrt(mass) Y] сохраняет LS minimizer;
последний lstsq выбирает minimum-norm core при нулевом ridge. Редукция
не строит D.T D и не возводит condition number в квадрат при решении.
Normal residual после решения проверяется отдельно, порог 1e-8.

При R=Y-prediction:
G=-2 g.T [mass * (U_j.T R_j)] + 2 lambda (X-T).
Евклидовы производные: GA=G V M.T, GV=G.T A M, GM=A.T G V.
После точного core refit GM=0 с погрешностью решения. Вертикальные
компоненты A.T GA и V.T GV также нулевые, поэтому горизонтальные
проекции (Id-AA.T)GA, (Id-VV.T)GV дают производную reduced objective.
QR(A-step GA), QR(V-step GV) — retraction; каждый trial заново решает M.
Armijo требует F_new<=F_old-1e-4*step*(||GA||²+||GV||²).
SVD малого M=R S St вращает A->A R, V->V St.T, сохраняя X и U V.
Вращения A->A R,V->V S,M->R.T M S не меняют функционал.

Инициализация: Q=g.T[mass U.T Y]+lambda T=-grad F(0)/2,
берутся сразу r сингулярных направлений, затем full core refit.
Это first-order инициализация; Hessian-aware CG не реализован.
Минимум по M не выше F(0), поскольку M=0 допустима. Нет гарантии
глобального rank-r минимума, ускорения или качества адаптивного ADP fit.
Факторный градиент при вырожденном M может скрывать направления улучшения;
rank_loss и matrix_tangent_gradient_norm выводятся отдельно, rank loss
не помечается joint_converged. Даже при полном ранге joint_converged означает
только относительный факторный критерий, не глобальную оптимальность.

Память: X,G размера m*d; cached W размера J*p*r; рабочие строки дизайна
chunk*r². Консервативная цель scratch chunks — 16 MiB плюс O(r^4) для R
и SVD/lstsq workspace. Верхняя оценка трех core квадратов проверяется
до аллокации. Native LAPACK workspace не является жестко сертифицированным
лимитом RSS. Нет d*d, (md)^2 Hessian и полного (Jp,md) дизайна.
QR core имеет стоимость O(Jp*r^4), не O(Jp*r²); при r=10 core заметно
дороже, чем при r=2. Поэтому малое SVD само по себе не доказывает ускорение.

Независимый аудит формул выполняется в tests/test_svd_joint.py:
полный dense Kronecker design, явные матричные атомы (i,k), augmented lstsq,
проверка cross Hessian/off-diagonal M, finite differences reduced objective,
нулевого core gradient, gauge invariance, rotation covariance, collinearity,
неравномерных mass, lambda=0/.7/1e4, streamed QR, rank/finite guards.
Изотропный пример имеет известный глобальный ответ truncated SVD и проверяется
для r=1,2,3. Это независимо написанный вычислительный эталон, не внешний
экспертный аудит. Full ADP callback сохраняет basis completion/QR и refit.
