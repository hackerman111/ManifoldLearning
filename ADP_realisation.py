import numpy as np
import tqdm

### Генерация данных
# Параметры для генерации
# d - размерность
# n - количество элементов в выборке
# eps - шум
# f - функция связи

rng = np.random.default_rng(42)
d = 10
n = 100
f = lambda X: np.sin(X)

# Пареметры для распределения
# sigma_x, mu_x - паремтры для X_i
# sigma_eps, sigma_eps - параметры для шума
mu_x = 0.0
sigma_x = 1.0

mu_eps = 0.0
sigma_eps = 1.0


mu_beta = 0.0
sigma_beta = 1.0

# Потом реализовать через эффективный размер где n = d/sigma
X = rng.normal(loc=mu_x, scale=sigma_x, size=(d, n))
eps = rng.normal(loc=mu_eps, scale=sigma_eps, size=(d))
beta = rng.normal(loc=mu_beta, scale=sigma_beta, size=(d))
Y = f(beta.T @ X) + eps

# Зафиксировать:
# N_loc - количество точек, которые должны быть в окрестности
# N_lin - параметр для
# N_J - количество окрестностей
# N_phi - кличество проекций
# K - тензор
# lambda - параметр ридж регрессии
# a - паремтр растяжения
# h_min - минимальное количество соседей
# J - выбранные компоненты
# phi - случайные проекции
# x_j - центры для наблюдения. Обычно x_j = X_j


def calculate_h_0(N_J, K, X, x, N_loc):
    pass


J = n
N_loc = 10
N_lin = n / N_loc  # Между 2d и n \ N_loc
N_J = n
N_phi = 5
K = lambda x: np.max(0, 1 - x**2)
lam = 0.1
a = np.sqrt(2)
phi = rng.normal(loc=0, scale=1, size=(d, N_phi))
x = X

### 0 Шаг
