# Изменения grassman_optim.py

Классификация относительно сохранённого `ADP/solver/grassman.py`:
EXACT для перестановки вычислений и кэшей; NUMERICAL для локального QR.
Сам ограниченный Grassmann optimizer остаётся прежним APPROXIMATE методом.
Цель, допуски, default core_gn/max_steps=5, Armijo, damping и публичная
сигнатура не изменены. local_ridge>0 остаётся прежней явной ESTIMATOR опцией.

1. Локальный профиль. При m=1/2 и безопасной обусловленности используется
   Gram–Schmidt с повторной ортогонализацией вместо J отдельных LAPACK SVD.
   Ridge решается на том же расширенном [M;sqrt(tau)I]. Коэффициенты —
   triangular backsolve; действие нормальной обратной системы — два
   triangular solves, без формирования нормальной матрицы или inverse.
   Малое сингулярное число 2x2 triangular R вычисляется через hypot и
   произведение диагоналей, без вычитания почти равных корней. Если
   cond(R)^2*eps >= sqrt(eps), мал абсолютный масштаб, p<m или m>2,
   вызывается исходный SVD со всеми minimum-norm/rank/smooth guards.
2. Горизонтальный градиент. Алгебраическая перестановка:
   G = -2 sum_j (U_j.T r_j) (mass_j g_j).T; затем G -= Y(Y.T G).
   Вместо повторного прохода по U для каждого столбца сначала вычисляется
   локальный adjoint. Чанки ограничивают сумму adjoint и weightedcoefficients
   4MiB (при хотя бы одном центре); временный массив J*d не сохраняется.
3. Операторные действия. (Y.T @ U_flat.T).T и аналог для V используют wide
   GEMM. Flatten contiguous U — view; копируется только малый (J,p,k)
   результат для последующих локальных kernels. Одновременно может жить
   ещё один temporary размера результата, но не новый U-sized массив.
4. Полный GN Jacobian собирается батчами для всех q*m координат. Для
   T=W K остаётся весь rhs = T.T r - M.T T g; cross terms не отброшены.
   Столбец (k,l) имеет rhs_i = delta(i,l)*w_k.T r - (M.T w_k)_i*g_l.
   Порядок столбцов совпадает с K.ravel(). Core scratch освобождается перед
   следующей аллокацией; вес sqrt(mass) применяется in-place.
5. Кэши. sqrt(mass) и tan(max_angle) живут только в одном solve.
   GN design/RHS переиспользуются для damping trials в одном frozen шаге;
   меняется только нижний ridge-блок. AngleProfile хранит M@a и постоянный
   denominator guard; spectral backtracking хранит Y@R.T и M@R.T.
   При смене базиса создаются новые локальные объекты. Глобального кэша нет.

Сохранены augmented lstsq для GN, workspace fallback, rank-boundary fallback,
ортонормальность, диагностические статусы и окончательный пересчёт через
live U. Снижение числа итераций, tolerances или смена статистик не используются.
Численную эквивалентность и ускорение подтверждают тесты и исходные paired
артефакты, перечисленные в REPORT.md и VERIFICATION.md этой директории.
