from __future__ import annotations

import hashlib

import numpy as np
from scipy import linalg as scipy_linalg
from scipy import sparse
from scipy.spatial import cKDTree

from adp.characteristics import ADP_Characteristics
from adp.data import Data


class EDR:
    def __init__(
        self,
        n_samples=200,
        n_features=5,
        n_centers=40,
        n_directions=8,
        min_neighbors=10,
        h_decay=1.5,
        h_min=1e-2,
        max_outer_steps=4,
        rho_min=1e-2,
        ridge=1e-6,
        tol=1e-6,
        max_iter=50,
        random_starts=5,
        h_search_steps=35,
        rho_search_steps=35,
        dtype=np.float64,
        random_state=None,
        noise_scale=0.05,
    ) -> None:
        # Вход:
        # - параметры размера данных, локальных окон, регуляризации и случайности.
        #
        # Выход:
        # - инициализированный объект EDR с пустым состоянием данных.
        #
        # Что делает:
        # - сохраняет параметры average derivative procedure и готовит служебные объекты.
        #
        # Реализация:
        # - параметры алгоритма, данные и состояние итераций хранятся явно в полях объекта.

        self.n_samples = int(n_samples)
        self.n_features = int(n_features)
        self.n_centers = int(n_centers)
        self.n_directions = int(n_directions)
        self.min_neighbors = int(min_neighbors)
        self.max_outer_steps = int(max_outer_steps)

        self.h_decay = float(h_decay)
        self.h_min = float(h_min)
        self.rho_min = float(rho_min)
        self.ridge = float(ridge)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.random_starts = int(random_starts)
        self.h_search_steps = int(h_search_steps)
        self.rho_search_steps = int(rho_search_steps)
        self.noise_scale = float(noise_scale)

        self.dtype = np.dtype(dtype)
        self.eps = np.finfo(self.dtype).eps * 100
        self.rng = np.random.default_rng(random_state)

        self.Data = Data()
        self.Characteristics = ADP_Characteristics()

        self.X = None
        self.Y = None
        self.beta = None
        self.centers = None
        self.tree = None

        self.mean = None
        self.scale = None

        self.current_outer_step = 0
        self.rho_current = None
        self.last_objective = None
        self.last_local_slopes = None

        self._statistics_cache_key = None
        self._statistics_cache_value = None

    def _Call_Data_Method(self, method_name):
        # Вход:
        # - имя метода объекта Data.
        #
        # Выход:
        # - результат метода или None, если текущая заглушка Data ничего не дает.
        #
        # Что делает:
        # - безопасно вызывает будущий Data-контракт и текущие pass-заглушки.
        #
        # Реализация:
        # - сначала пробует bound method, затем fallback для текущих методов без self.

        data_method = getattr(self.Data, method_name, None)

        if data_method is None:
            return None

        try:
            return data_method()
        except TypeError:
            raw_method = getattr(type(self.Data), method_name, None)

            if raw_method is None:
                return None

            try:
                return raw_method()
            except TypeError:
                return None

    def _Ensure_Data_Is_Ready(self):
        # Вход:
        # - текущее состояние self.X, self.Y, self.centers.
        #
        # Выход:
        # - ничего не возвращает.
        #
        # Что делает:
        # - проверяет, что основные массивы и cKDTree готовы к локальным вычислениям.
        #
        # Реализация:
        # - ошибка возникает рано и с понятным сообщением, если данные еще не созданы.

        if self.X is None or self.Y is None or self.centers is None:
            raise ValueError("X, Y and centers must be generated before this calculation.")

        if self.tree is None:
            self.tree = cKDTree(self.X)

    def _Array_Fingerprint_Calculate(self, array):
        # Вход:
        # - числовой массив numpy или объект, приводимый к numpy array.
        #
        # Выход:
        # - короткий content hash с учетом dtype, shape и значений.
        #
        # Что делает:
        # - создает стабильную подпись массива для безопасного кеша статистик.
        #
        # Реализация:
        # - использует contiguous view и blake2b без создания Python-списков.

        numeric_array = np.asarray(array)
        contiguous_array = np.ascontiguousarray(numeric_array)

        hash_builder = hashlib.blake2b(digest_size=16)
        hash_builder.update(str(contiguous_array.dtype).encode("ascii"))
        hash_builder.update(np.asarray(contiguous_array.shape, dtype=np.int64).tobytes())
        hash_builder.update(contiguous_array.view(np.uint8))

        return hash_builder.hexdigest()

    def _Sparse_Fingerprint_Calculate(self, matrix):
        # Вход:
        # - sparse matrix, приводимая к CSR.
        #
        # Выход:
        # - content hash для структуры и значений CSR.
        #
        # Что делает:
        # - учитывает data, indices и indptr, чтобы in-place изменения инвалидировали кеш.
        #
        # Реализация:
        # - комбинирует shape, nnz и hash трех внутренних CSR-массивов.

        matrix = sparse.csr_matrix(matrix)

        return (
            matrix.shape,
            matrix.nnz,
            self._Array_Fingerprint_Calculate(matrix.data),
            self._Array_Fingerprint_Calculate(matrix.indices),
            self._Array_Fingerprint_Calculate(matrix.indptr),
        )

    def _Target_Neighbor_Count(self):
        # Вход:
        # - self.min_neighbors и число наблюдений в self.X.
        #
        # Выход:
        # - достижимый порог локальной массы.
        #
        # Что делает:
        # - не дает требовать больше соседей, чем есть наблюдений.
        #
        # Реализация:
        # - ограничивает min_neighbors отрезком [1, n_samples].

        if self.X is None:
            return max(1, self.min_neighbors)

        number_of_samples = self.X.shape[0]

        return max(1, min(self.min_neighbors, number_of_samples))

    def _Average_Kernel_Mass_Calculate(self, h, rho=None, beta=None):
        # Вход:
        # - bandwidth h;
        # - optional rho и beta для анизотропной метрики.
        #
        # Выход:
        # - средняя ненормированная kernel-масса по центрам.
        #
        # Что делает:
        # - считает массу локальных окон без row-normalization.
        #
        # Реализация:
        # - использует cKDTree radius search и обходит только точки внутри носителя.

        self._Ensure_Data_Is_Ready()

        bandwidth = float(h)

        if bandwidth <= 0:
            raise ValueError("h must be positive.")

        if rho is None and beta is None:
            normalized_beta = None
            rho_value = None
            search_radius = bandwidth
        else:
            if rho is None or beta is None:
                raise ValueError("rho and beta must be passed together.")

            normalized_beta = self.Beta_Normalize(beta)
            rho_value = max(float(rho), self.eps)
            search_radius = bandwidth / rho_value

        total_kernel_mass = 0.0

        # Считаем ненормированную массу отдельно для каждого центра.
        for center_index, center in enumerate(self.centers):
            neighbor_indices = self.tree.query_ball_point(center, r=search_radius)

            if not neighbor_indices:
                continue

            neighbor_indices = np.asarray(sorted(neighbor_indices), dtype=np.intp)
            centered_neighbors = self.X[neighbor_indices] - center
            squared_distance = np.einsum("nd,nd->n", centered_neighbors, centered_neighbors)

            if normalized_beta is None:
                normalized_distance = squared_distance / (bandwidth**2)
            else:
                projection_on_beta = centered_neighbors @ normalized_beta
                normalized_distance = (
                    rho_value**2 * squared_distance + projection_on_beta**2
                ) / (bandwidth**2)

            neighbor_weights = self.Kernel_Calculate(normalized_distance)
            total_kernel_mass += float(np.sum(neighbor_weights))

        return total_kernel_mass / max(1, self.centers.shape[0])

    def Mean_Calculate(self):
        # Вход:
        # - self.X формы (n_samples, n_features).
        #
        # Выход:
        # - feature_mean и feature_scale.
        #
        # Что делает:
        # - считает параметры стандартизации признаков.
        #
        # Реализация:
        # - маленькие scale заменяются на 1.0, чтобы не делить на ноль.

        if self.X is None:
            raise ValueError("X must be set before mean calculation.")

        feature_matrix = np.asarray(self.X, dtype=self.dtype)

        feature_mean = np.mean(feature_matrix, axis=0)
        feature_scale = np.std(feature_matrix, axis=0)

        small_scale_mask = feature_scale <= self.eps
        feature_scale = feature_scale.copy()
        feature_scale[small_scale_mask] = 1.0

        self.mean = feature_mean
        self.scale = feature_scale

        return feature_mean, feature_scale

    def Local_Mean_Calculate(self, weights):
        # Вход:
        # - weights: CSR или приводимая к CSR матрица формы (n_centers, n_samples).
        #
        # Выход:
        # - local_mean формы (n_centers, n_features).
        #
        # Что делает:
        # - считает Xbar_j = sum_i X_i w_ij.
        #
        # Реализация:
        # - основная часть считается sparse matrix multiplication.

        self._Ensure_Data_Is_Ready()

        weights = sparse.csr_matrix(weights, dtype=self.dtype)

        expected_shape = (self.centers.shape[0], self.X.shape[0])

        if weights.shape != expected_shape:
            raise ValueError(f"weights must have shape {expected_shape}.")

        local_mean = weights @ self.X
        local_mean = np.asarray(local_mean, dtype=self.dtype)

        row_sums = np.asarray(weights.sum(axis=1)).ravel()
        empty_row_indices = np.flatnonzero(row_sums <= self.eps)

        # Заменяем среднее центром там, где локальное окно оказалось пустым.
        for center_index in empty_row_indices:
            local_mean[center_index] = self.centers[center_index]

        return local_mean

    def Kernel_Calculate(self, distance):
        # Вход:
        # - distance: скаляр или массив нормированных квадратов расстояний.
        #
        # Выход:
        # - значения компактного quartic kernel той же формы.
        #
        # Что делает:
        # - возвращает (1 - distance)^2 внутри носителя distance < 1.
        #
        # Реализация:
        # - sqrt не используется, потому что distance уже квадрат нормы в масштабе h.

        normalized_distance = np.asarray(distance, dtype=self.dtype)

        kernel_values = np.zeros_like(normalized_distance, dtype=self.dtype)
        inside_kernel_support = normalized_distance < 1.0

        kernel_values[inside_kernel_support] = (
            1.0 - normalized_distance[inside_kernel_support]
        ) ** 2

        if np.isscalar(distance):
            return float(kernel_values)

        return kernel_values

    def Generate_Direction(self):
        # Вход:
        # - self.centers и self.X задают число центров и размерность.
        #
        # Выход:
        # - direction_vectors формы (n_centers, n_directions, n_features).
        #
        # Что делает:
        # - генерирует равномерные направления на единичной сфере для step 0.
        #
        # Реализация:
        # - нормирует независимые Gaussian-векторы.

        self._Ensure_Data_Is_Ready()

        number_of_centers = self.centers.shape[0]
        number_of_features = self.X.shape[1]

        direction_vectors = self.rng.normal(
            size=(number_of_centers, self.n_directions, number_of_features)
        ).astype(self.dtype)

        direction_norms = np.linalg.norm(direction_vectors, axis=-1, keepdims=True)
        small_norm_mask = direction_norms[..., 0] <= self.eps

        # Заменяем численно вырожденные направления первым базисным вектором.
        for center_index, direction_index in np.argwhere(small_norm_mask):
            direction_vectors[center_index, direction_index] = 0.0
            direction_vectors[center_index, direction_index, 0] = 1.0
            direction_norms[center_index, direction_index, 0] = 1.0

        direction_vectors = direction_vectors / direction_norms

        return direction_vectors

    def Generate_Anisotropic_Direction(self, beta_previous, h_k, rho_k):
        # Вход:
        # - beta_previous: направление прошлого шага;
        # - h_k: текущий bandwidth из теории;
        # - rho_k: текущая анизотропия.
        #
        # Выход:
        # - direction_vectors формы (n_centers, n_directions, n_features).
        #
        # Что делает:
        # - генерирует направления, вытянутые вдоль beta_previous.
        #
        # Реализация:
        # - h_k проверяется на корректность, но после нормировки направлений не входит явно.

        self._Ensure_Data_Is_Ready()

        if float(h_k) <= 0:
            raise ValueError("h_k must be positive.")

        normalized_beta = self.Beta_Normalize(beta_previous)
        rho_value = max(float(rho_k), self.eps)

        number_of_centers = self.centers.shape[0]
        number_of_features = self.X.shape[1]

        standard_noise = self.rng.normal(
            size=(number_of_centers, self.n_directions, number_of_features)
        ).astype(self.dtype)
        beta_direction_noise = self.rng.normal(
            size=(number_of_centers, self.n_directions, 1)
        ).astype(self.dtype)

        anisotropic_vectors = rho_value * standard_noise
        anisotropic_vectors = anisotropic_vectors + beta_direction_noise * normalized_beta

        direction_norms = np.linalg.norm(anisotropic_vectors, axis=-1, keepdims=True)
        small_norm_mask = direction_norms[..., 0] <= self.eps

        # Заменяем численно вырожденные анизотропные направления на beta_previous.
        for center_index, direction_index in np.argwhere(small_norm_mask):
            anisotropic_vectors[center_index, direction_index] = normalized_beta
            direction_norms[center_index, direction_index, 0] = 1.0

        direction_vectors = anisotropic_vectors / direction_norms

        return direction_vectors

    def Weight_Calculate(self, h, rho=None, beta=None):
        # Вход:
        # - h: bandwidth;
        # - rho и beta: optional параметры анизотропной метрики.
        #
        # Выход:
        # - CSR-матрица весов формы (n_centers, n_samples) с row-normalization.
        #
        # Что делает:
        # - строит локальные kernel-веса для step 0 или step k.
        #
        # Реализация:
        # - использует cKDTree radius search и хранит только ненулевые веса.

        self._Ensure_Data_Is_Ready()

        bandwidth = float(h)

        if bandwidth <= 0:
            raise ValueError("h must be positive.")

        if rho is None and beta is None:
            normalized_beta = None
            rho_value = None
            search_radius = bandwidth
        else:
            if rho is None or beta is None:
                raise ValueError("rho and beta must be passed together.")

            normalized_beta = self.Beta_Normalize(beta)
            rho_value = max(float(rho), self.eps)
            search_radius = bandwidth / rho_value

        data_values = []
        column_indices = []
        row_pointer = [0]

        # Собираем локальные ненулевые веса отдельно для каждой строки CSR.
        for center_index, center in enumerate(self.centers):
            neighbor_indices = self.tree.query_ball_point(center, r=search_radius)

            if neighbor_indices:
                neighbor_indices = np.asarray(sorted(neighbor_indices), dtype=np.intp)
                centered_neighbors = self.X[neighbor_indices] - center
                squared_distance = np.einsum("nd,nd->n", centered_neighbors, centered_neighbors)

                if normalized_beta is None:
                    normalized_distance = squared_distance / (bandwidth**2)
                else:
                    projection_on_beta = centered_neighbors @ normalized_beta
                    normalized_distance = (
                        rho_value**2 * squared_distance + projection_on_beta**2
                    ) / (bandwidth**2)

                neighbor_weights = self.Kernel_Calculate(normalized_distance)
                positive_weight_mask = neighbor_weights > self.eps

                data_values.extend(neighbor_weights[positive_weight_mask])
                column_indices.extend(neighbor_indices[positive_weight_mask])

            row_pointer.append(len(data_values))

        weights = sparse.csr_matrix(
            (
                np.asarray(data_values, dtype=self.dtype),
                np.asarray(column_indices, dtype=np.intp),
                np.asarray(row_pointer, dtype=np.intp),
            ),
            shape=(self.centers.shape[0], self.X.shape[0]),
            dtype=self.dtype,
        )

        # Нормируем каждую непустую строку, чтобы веса были локальными средними.
        for center_index in range(weights.shape[0]):
            row_start = weights.indptr[center_index]
            row_end = weights.indptr[center_index + 1]

            if row_start == row_end:
                continue

            row_sum = np.sum(weights.data[row_start:row_end])

            if row_sum > self.eps:
                weights.data[row_start:row_end] /= row_sum

        return weights

    def H0_Calculate(self):
        # Вход:
        # - self.X, self.centers и self.tree из состояния объекта.
        #
        # Выход:
        # - начальный bandwidth h_0.
        #
        # Что делает:
        # - ищет минимальный h_0 с достаточной средней kernel-массой.
        #
        # Реализация:
        # - сначала строит верхнюю границу, затем делает bisection.

        self._Ensure_Data_Is_Ready()

        target_neighbor_count = self._Target_Neighbor_Count()
        query_neighbor_count = min(self.X.shape[0], target_neighbor_count)

        nearest_distances, _ = self.tree.query(self.centers, k=query_neighbor_count)
        nearest_distances = np.asarray(nearest_distances, dtype=self.dtype)

        if nearest_distances.ndim == 2:
            kth_neighbor_distances = nearest_distances[:, -1]
        else:
            kth_neighbor_distances = nearest_distances

        positive_distances = kth_neighbor_distances[kth_neighbor_distances > self.eps]

        if positive_distances.size == 0:
            lower_bandwidth = self.eps
            upper_bandwidth = 1.0
        else:
            lower_bandwidth = max(float(np.quantile(positive_distances, 0.1)), self.eps)
            upper_bandwidth = max(
                float(np.quantile(positive_distances, 0.9)),
                lower_bandwidth * 2.0,
            )

        # Расширяем верхнюю границу, пока средняя масса не достигнет нужного порога.
        for _ in range(self.h_search_steps):
            average_kernel_mass = self._Average_Kernel_Mass_Calculate(upper_bandwidth)

            if average_kernel_mass >= target_neighbor_count:
                break

            upper_bandwidth *= 2.0

        # Сжимаем нижнюю границу, пока она точно не станет недостаточной.
        for _ in range(self.h_search_steps):
            average_kernel_mass = self._Average_Kernel_Mass_Calculate(lower_bandwidth)

            if average_kernel_mass < target_neighbor_count or lower_bandwidth <= self.eps:
                break

            lower_bandwidth *= 0.5

        # Bisection ищет минимальный bandwidth с достаточной средней массой.
        for _ in range(self.h_search_steps):
            candidate_bandwidth = 0.5 * (lower_bandwidth + upper_bandwidth)
            average_kernel_mass = self._Average_Kernel_Mass_Calculate(candidate_bandwidth)

            if average_kernel_mass >= target_neighbor_count:
                upper_bandwidth = candidate_bandwidth
            else:
                lower_bandwidth = candidate_bandwidth

        h_0 = float(upper_bandwidth)
        self.Characteristics.H0_Save(h_0)

        return h_0

    def H_Update(self, h_previous):
        # Вход:
        # - h_previous: bandwidth прошлого adaptive step.
        #
        # Выход:
        # - updated_bandwidth = h_previous / h_decay.
        #
        # Что делает:
        # - уменьшает bandwidth по правилу ADP.
        #
        # Реализация:
        # - дополнительно увеличивает счетчик outer-step.

        previous_bandwidth = float(h_previous)

        if previous_bandwidth <= 0:
            raise ValueError("h_previous must be positive.")

        if self.h_decay <= 1:
            raise ValueError("h_decay must be greater than 1.")

        updated_bandwidth = previous_bandwidth / self.h_decay

        self.current_outer_step += 1
        self.Characteristics.H_k_Save(updated_bandwidth)

        return updated_bandwidth

    def Step_k_Condition(self, h_k):
        # Вход:
        # - h_k: текущий bandwidth перед запуском следующего adaptive step.
        #
        # Выход:
        # - boolean, нужно ли запускать следующий step k.
        #
        # Что делает:
        # - объединяет лимит числа шагов и нижний порог bandwidth.
        #
        # Реализация:
        # - функция не меняет состояние, счетчик увеличивает H_Update.

        current_bandwidth = float(h_k)

        has_remaining_steps = self.current_outer_step < self.max_outer_steps
        next_bandwidth = current_bandwidth / self.h_decay
        has_large_enough_bandwidth = next_bandwidth >= self.h_min

        return bool(has_remaining_steps and has_large_enough_bandwidth)

    def Rho_Calculate(self, beta_previous, h_k):
        # Вход:
        # - beta_previous: направление прошлого шага;
        # - h_k: текущий bandwidth.
        #
        # Выход:
        # - rho_k из отрезка [rho_min, 1].
        #
        # Что делает:
        # - выбирает максимальную rho с достаточной средней kernel-массой.
        #
        # Реализация:
        # - bisection использует монотонное убывание массы при росте rho.

        self._Ensure_Data_Is_Ready()

        bandwidth = float(h_k)

        if bandwidth <= 0:
            raise ValueError("h_k must be positive.")

        normalized_beta = self.Beta_Normalize(beta_previous)
        target_neighbor_count = self._Target_Neighbor_Count()

        lower_rho = max(float(self.rho_min), self.eps)
        upper_rho = 1.0

        mass_at_upper_rho = self._Average_Kernel_Mass_Calculate(
            bandwidth,
            upper_rho,
            normalized_beta,
        )

        if mass_at_upper_rho >= target_neighbor_count:
            rho_k = upper_rho
        else:
            mass_at_lower_rho = self._Average_Kernel_Mass_Calculate(
                bandwidth,
                lower_rho,
                normalized_beta,
            )

            if mass_at_lower_rho < target_neighbor_count:
                rho_k = lower_rho
            else:
                # Bisection ищет максимальную допустимую rho, то есть минимально нужную анизотропию.
                for _ in range(self.rho_search_steps):
                    candidate_rho = 0.5 * (lower_rho + upper_rho)
                    average_kernel_mass = self._Average_Kernel_Mass_Calculate(
                        bandwidth,
                        candidate_rho,
                        normalized_beta,
                    )

                    if average_kernel_mass >= target_neighbor_count:
                        lower_rho = candidate_rho
                    else:
                        upper_rho = candidate_rho

                rho_k = lower_rho

        self.rho_current = float(rho_k)
        self.Characteristics.Rho_k_Save(self.rho_current)

        return self.rho_current

    def Generate_Data(self):
        # Вход:
        # - текущее состояние self.X/self.Y/self.beta/self.centers или будущий Data-контракт.
        #
        # Выход:
        # - ничего не возвращает, заполняет состояние объекта.
        #
        # Что делает:
        # - готовит стандартизованные X, Y, beta, centers и cKDTree.
        #
        # Реализация:
        # - сначала использует уже заданные массивы, затем Data, затем synthetic fallback.

        generated_X = self.X if self.X is not None else self._Call_Data_Method("Generate_X")
        generated_beta = (
            self.beta if self.beta is not None else self._Call_Data_Method("Generate_beta")
        )

        self._Call_Data_Method("Generate_func")
        self._Call_Data_Method("Generate_Noise")

        generated_Y = self.Y if self.Y is not None else self._Call_Data_Method("Generate_Y")
        generated_centers = (
            self.centers if self.centers is not None else self._Call_Data_Method("Generate_Centers")
        )

        if generated_X is None:
            generated_X = self.rng.normal(size=(self.n_samples, self.n_features))

        generated_X = np.asarray(generated_X, dtype=self.dtype)

        if generated_X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        if generated_beta is None:
            generated_beta = self.rng.normal(size=generated_X.shape[1])

        generated_beta = np.asarray(generated_beta, dtype=self.dtype).reshape(-1)

        if generated_beta.shape[0] != generated_X.shape[1]:
            raise ValueError("beta must have length n_features.")

        generated_beta = self.Beta_Normalize(generated_beta)

        if generated_Y is None:
            single_index = generated_X @ generated_beta
            generated_Y = np.sin(single_index)
            generated_Y = generated_Y + self.noise_scale * self.rng.normal(
                size=generated_X.shape[0]
            )

        generated_Y = np.asarray(generated_Y, dtype=self.dtype).reshape(-1)

        if generated_Y.shape[0] != generated_X.shape[0]:
            raise ValueError("Y must have length n_samples.")

        if generated_centers is None:
            number_of_samples = generated_X.shape[0]
            number_of_centers = min(max(1, self.n_centers), number_of_samples)

            # Выбираем центры из X, чтобы локальные окрестности точно были непустыми.
            center_indices = self.rng.choice(
                number_of_samples,
                size=number_of_centers,
                replace=False,
            )
            generated_centers = generated_X[center_indices]

        generated_centers = np.asarray(generated_centers, dtype=self.dtype)

        if generated_centers.ndim != 2 or generated_centers.shape[1] != generated_X.shape[1]:
            raise ValueError("centers must have shape (n_centers, n_features).")

        self.X = generated_X
        self.Y = generated_Y
        self.centers = generated_centers

        feature_mean, feature_scale = self.Mean_Calculate()

        standardized_X = (generated_X - feature_mean) / feature_scale
        standardized_centers = (generated_centers - feature_mean) / feature_scale
        standardized_beta = generated_beta * feature_scale

        self.X = np.asarray(standardized_X, dtype=self.dtype)
        self.Y = generated_Y
        self.beta = self.Beta_Normalize(standardized_beta)
        self.centers = np.asarray(standardized_centers, dtype=self.dtype)
        self.tree = cKDTree(self.X)

        self.current_outer_step = 0
        self._statistics_cache_key = None
        self._statistics_cache_value = None

    def Beta_Normalize(self, beta, beta_previous=None):
        # Вход:
        # - beta: произвольный вектор;
        # - beta_previous: optional вектор для выравнивания знака.
        #
        # Выход:
        # - normalized_beta единичной нормы.
        #
        # Что делает:
        # - стабилизирует направление beta и защищается от нулевой нормы.
        #
        # Реализация:
        # - нулевой или нечисловой вектор заменяется случайным направлением.

        beta_vector = np.asarray(beta, dtype=self.dtype).reshape(-1)

        if beta_vector.size == 0:
            raise ValueError("beta must be non-empty.")

        beta_norm = np.linalg.norm(beta_vector)

        if not np.isfinite(beta_norm) or beta_norm <= self.eps:
            beta_vector = self.rng.normal(size=beta_vector.shape[0]).astype(self.dtype)
            beta_norm = np.linalg.norm(beta_vector)

            if beta_norm <= self.eps:
                beta_vector = np.zeros_like(beta_vector)
                beta_vector[0] = 1.0
                beta_norm = 1.0

        normalized_beta = beta_vector / beta_norm

        if beta_previous is not None:
            previous_beta = np.asarray(beta_previous, dtype=self.dtype).reshape(-1)

            if previous_beta.shape == normalized_beta.shape:
                scalar_product = float(normalized_beta @ previous_beta)

                if scalar_product < 0:
                    normalized_beta = -normalized_beta

        return normalized_beta

    def I_Calculate(self, weights, local_mean, directions):
        # Вход:
        # - weights, local_mean, directions из текущего шага ADP.
        #
        # Выход:
        # - I_statistics формы (n_centers, n_directions).
        #
        # Что делает:
        # - возвращает только статистику I.
        #
        # Реализация:
        # - переиспользует общий расчет пары (I, U) и кеш последнего вызова.

        I_statistics, _ = self.Average_Derivative_Statistics_Calculate(
            weights,
            local_mean,
            directions,
        )

        return I_statistics

    def U_Calculate(self, weights, local_mean, directions):
        # Вход:
        # - weights, local_mean, directions из текущего шага ADP.
        #
        # Выход:
        # - U_statistics формы (n_centers, n_directions, n_features).
        #
        # Что делает:
        # - возвращает только статистику U.
        #
        # Реализация:
        # - переиспользует общий расчет пары (I, U) и кеш последнего вызова.

        _, U_statistics = self.Average_Derivative_Statistics_Calculate(
            weights,
            local_mean,
            directions,
        )

        return U_statistics

    def Average_Derivative_Statistics_Calculate(self, weights, local_mean, directions):
        # Вход:
        # - weights: CSR-веса;
        # - local_mean: локальные средние Xbar_j;
        # - directions: направления phi.
        #
        # Выход:
        # - пара (I_statistics, U_statistics).
        #
        # Что делает:
        # - считает статистики average derivative procedure по формулам из TeX.
        #
        # Реализация:
        # - обходит только ненулевые CSR-веса и не создает плотный тензор (m, n, d).

        self._Ensure_Data_Is_Ready()

        weights = sparse.csr_matrix(weights, dtype=self.dtype)
        local_mean = np.asarray(local_mean, dtype=self.dtype)
        directions = np.asarray(directions, dtype=self.dtype)

        cache_key = (
            self._Sparse_Fingerprint_Calculate(weights),
            self._Array_Fingerprint_Calculate(local_mean),
            self._Array_Fingerprint_Calculate(directions),
            self._Array_Fingerprint_Calculate(self.X),
            self._Array_Fingerprint_Calculate(self.Y),
        )

        if self._statistics_cache_key == cache_key and self._statistics_cache_value is not None:
            return self._statistics_cache_value

        number_of_centers = self.centers.shape[0]
        number_of_directions = directions.shape[1]
        number_of_features = self.X.shape[1]

        if weights.shape != (number_of_centers, self.X.shape[0]):
            raise ValueError("weights have incompatible shape.")

        if local_mean.shape != (number_of_centers, number_of_features):
            raise ValueError("local_mean has incompatible shape.")

        if directions.shape != (number_of_centers, number_of_directions, number_of_features):
            raise ValueError("directions have incompatible shape.")

        I_statistics = np.zeros((number_of_centers, number_of_directions), dtype=self.dtype)
        U_statistics = np.zeros(
            (number_of_centers, number_of_directions, number_of_features),
            dtype=self.dtype,
        )

        # Обходим центры по CSR-строкам, чтобы использовать только локальных соседей.
        for center_index in range(number_of_centers):
            row_start = weights.indptr[center_index]
            row_end = weights.indptr[center_index + 1]

            if row_start == row_end:
                continue

            neighbor_indices = weights.indices[row_start:row_end]
            neighbor_weights = weights.data[row_start:row_end]

            centered_neighbors = self.X[neighbor_indices] - local_mean[center_index]
            projection_values = centered_neighbors @ directions[center_index].T

            weighted_response = neighbor_weights * self.Y[neighbor_indices]
            I_statistics[center_index] = weighted_response @ projection_values

            weighted_centered_neighbors = neighbor_weights[:, None] * centered_neighbors
            U_statistics[center_index] = projection_values.T @ weighted_centered_neighbors

        self._statistics_cache_key = cache_key
        self._statistics_cache_value = (I_statistics, U_statistics)

        return I_statistics, U_statistics

    def L_Calculate(self, I, U, beta):
        # Вход:
        # - I: статистика формы (n_centers, n_directions);
        # - U: статистика формы (n_centers, n_directions, n_features);
        # - beta: текущее направление.
        #
        # Выход:
        # - local_slopes формы (n_centers,).
        #
        # Что делает:
        # - считает оптимальные l_j при фиксированной beta.
        #
        # Реализация:
        # - использует закрытую формулу через проекцию U_j beta.

        I = np.asarray(I, dtype=self.dtype)
        U = np.asarray(U, dtype=self.dtype)
        normalized_beta = self.Beta_Normalize(beta)

        projected_U = np.einsum("mqd,d->mq", U, normalized_beta)

        slope_numerator = np.einsum("mq,mq->m", I, projected_U)
        slope_denominator = np.einsum("mq,mq->m", projected_U, projected_U)

        local_slopes = np.zeros_like(slope_numerator, dtype=self.dtype)
        stable_denominator_mask = slope_denominator > self.eps

        local_slopes[stable_denominator_mask] = (
            slope_numerator[stable_denominator_mask]
            / slope_denominator[stable_denominator_mask]
        )

        return local_slopes

    def Beta_Calculate(self, I, U, l, beta_previous=None):
        # Вход:
        # - I, U: статистики average derivative;
        # - l: локальные наклоны;
        # - beta_previous: optional ridge-якорь.
        #
        # Выход:
        # - новая ненормированная beta.
        #
        # Что делает:
        # - решает регуляризованную нормальную систему при фиксированных l_j.
        #
        # Реализация:
        # - аккумулирует normal_matrix и right_hand_side по центрам.

        I = np.asarray(I, dtype=self.dtype)
        U = np.asarray(U, dtype=self.dtype)
        local_slopes = np.asarray(l, dtype=self.dtype).reshape(-1)

        number_of_centers, _, number_of_features = U.shape

        if I.shape[0] != number_of_centers or local_slopes.shape[0] != number_of_centers:
            raise ValueError("I, U and l have incompatible shapes.")

        identity_matrix = np.eye(number_of_features, dtype=self.dtype)
        normal_matrix = self.ridge * identity_matrix
        right_hand_side = np.zeros(number_of_features, dtype=self.dtype)

        if beta_previous is not None:
            normalized_previous_beta = self.Beta_Normalize(beta_previous)
            right_hand_side += self.ridge * normalized_previous_beta

        # Накопление по центрам собирает нормальную систему без больших промежуточных тензоров.
        for center_index in range(number_of_centers):
            U_for_center = U[center_index]
            local_slope = local_slopes[center_index]

            if abs(local_slope) <= self.eps:
                continue

            normal_matrix += local_slope**2 * (U_for_center.T @ U_for_center)
            right_hand_side += local_slope * (U_for_center.T @ I[center_index])

        try:
            beta_candidate = scipy_linalg.solve(
                normal_matrix,
                right_hand_side,
                assume_a="pos",
                check_finite=False,
            )
        except scipy_linalg.LinAlgError:
            beta_candidate = np.linalg.lstsq(normal_matrix, right_hand_side, rcond=None)[0]

        return np.asarray(beta_candidate, dtype=self.dtype)

    def Objective_Calculate(self, I, U, l, beta):
        # Вход:
        # - I, U, l и beta из alternating minimization.
        #
        # Выход:
        # - objective_value как обычный float.
        #
        # Что делает:
        # - считает сумму квадратов невязок I_j - l_j U_j beta.
        #
        # Реализация:
        # - использует einsum для проекции U beta.

        I = np.asarray(I, dtype=self.dtype)
        U = np.asarray(U, dtype=self.dtype)
        local_slopes = np.asarray(l, dtype=self.dtype).reshape(-1)
        normalized_beta = self.Beta_Normalize(beta)

        projected_U = np.einsum("mqd,d->mq", U, normalized_beta)
        predicted_I = local_slopes[:, None] * projected_U

        residual = I - predicted_I
        objective_value = float(np.sum(residual * residual))

        return objective_value

    def Alternating_Minimization(self, I, U, beta_initial=None):
        # Вход:
        # - I и U: статистики текущего ADP step;
        # - beta_initial: optional старт и ridge-якорь для adaptive step.
        #
        # Выход:
        # - beta_hat единичной нормы.
        #
        # Что делает:
        # - чередует расчет l_j и beta до сходимости.
        #
        # Реализация:
        # - step 0 использует несколько random starts, step k стартует от beta_initial.

        I = np.asarray(I, dtype=self.dtype)
        U = np.asarray(U, dtype=self.dtype)
        number_of_features = U.shape[2]

        initial_beta_candidates = []

        if beta_initial is not None:
            beta_anchor = self.Beta_Normalize(beta_initial)
            initial_beta_candidates.append(beta_anchor)
        else:
            beta_anchor = None

            # Несколько random starts защищают step 0 от плохой случайной инициализации.
            for _ in range(max(1, self.random_starts)):
                random_beta = self.rng.normal(size=number_of_features)
                initial_beta_candidates.append(self.Beta_Normalize(random_beta))

        best_beta = initial_beta_candidates[0]
        best_objective = np.inf
        best_local_slopes = None

        # Перебираем стартовые направления и оставляем решение с минимальным objective.
        for initial_beta in initial_beta_candidates:
            current_beta = self.Beta_Normalize(initial_beta)
            previous_objective = np.inf
            current_local_slopes = None

            # Внутренний цикл чередует расчет локальных наклонов и обновление beta.
            for _ in range(max(1, self.max_iter)):
                current_local_slopes = self.L_Calculate(I, U, current_beta)
                candidate_beta = self.Beta_Calculate(
                    I,
                    U,
                    current_local_slopes,
                    beta_anchor,
                )
                candidate_beta = self.Beta_Normalize(candidate_beta, current_beta)

                current_objective = self.Objective_Calculate(
                    I,
                    U,
                    current_local_slopes,
                    candidate_beta,
                )
                beta_delta = np.linalg.norm(candidate_beta - current_beta)

                current_beta = candidate_beta

                if (
                    abs(previous_objective - current_objective) <= self.tol
                    or beta_delta <= self.tol
                ):
                    previous_objective = current_objective
                    break

                previous_objective = current_objective

            current_local_slopes = self.L_Calculate(I, U, current_beta)

            final_objective = self.Objective_Calculate(
                I,
                U,
                current_local_slopes,
                current_beta,
            )

            if final_objective < best_objective:
                best_objective = final_objective
                best_beta = current_beta
                best_local_slopes = current_local_slopes

        self.last_objective = float(best_objective)
        self.last_local_slopes = best_local_slopes

        return self.Beta_Normalize(best_beta)
