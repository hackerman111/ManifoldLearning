import numpy as np


class Data:
    def __init__(
        self,
        n_samples=200,
        n_features=5,
        n_centers=40,
        noise_scale=0.05,
        link="sin",
        dtype=np.float64,
        random_state=None,
    ) -> None:
        # Вход:
        # - размеры синтетической выборки, уровень шума, link-функция и random_state.
        #
        # Выход:
        # - объект генератора с пустыми массивами данных.
        #
        # Что делает:
        # - подготавливает параметры single-index генерации для average derivative.
        #
        # Реализация:
        # - все сгенерированные объекты сохраняются в полях, чтобы методы работали по шагам.

        self.n_samples = int(n_samples)
        self.n_features = int(n_features)
        self.n_centers = int(n_centers)
        self.noise_scale = float(noise_scale)
        self.link = link
        self.dtype = np.dtype(dtype)
        self.eps = np.finfo(self.dtype).eps * 100
        self.rng = np.random.default_rng(random_state)

        self.X = None
        self.noise = None
        self.func = None
        self.beta = None
        self.Y = None
        self.centers = None
        self.center_indices = None

    def Generate_X(self):
        # Вход:
        # - параметры n_samples и n_features из состояния генератора.
        #
        # Выход:
        # - X формы (n_samples, n_features).
        #
        # Что делает:
        # - генерирует признаки из стандартного нормального распределения.
        #
        # Реализация:
        # - результат сохраняется в self.X и возвращается наружу.

        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        if self.n_features <= 0:
            raise ValueError("n_features must be positive.")

        self.X = self.rng.normal(size=(self.n_samples, self.n_features))
        self.X = np.asarray(self.X, dtype=self.dtype)

        self.Y = None
        self.centers = None
        self.center_indices = None

        return self.X

    def Generate_Noise(self):
        # Вход:
        # - n_samples и noise_scale из состояния генератора.
        #
        # Выход:
        # - noise формы (n_samples,).
        #
        # Что делает:
        # - генерирует аддитивный Gaussian noise для single-index модели.
        #
        # Реализация:
        # - при noise_scale = 0 возвращает нулевой массив без случайного сдвига.

        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        if self.noise_scale < 0:
            raise ValueError("noise_scale must be non-negative.")

        if self.noise_scale == 0:
            self.noise = np.zeros(self.n_samples, dtype=self.dtype)
        else:
            self.noise = self.rng.normal(scale=self.noise_scale, size=self.n_samples)
            self.noise = np.asarray(self.noise, dtype=self.dtype)

        self.Y = None

        return self.noise

    def Generate_func(self):
        # Вход:
        # - self.link: строка или callable link-функция.
        #
        # Выход:
        # - функция, применяемая к одномерному индексу X @ beta.
        #
        # Что делает:
        # - выбирает связь single-index модели.
        #
        # Реализация:
        # - поддерживает простые link-функции без внешних зависимостей.

        if callable(self.link):
            self.func = self.link

            return self.func

        link_name = str(self.link).lower()

        if link_name == "linear":
            self.func = lambda single_index: single_index
        elif link_name in {"sin", "sine"}:
            self.func = np.sin
        elif link_name == "cos":
            self.func = np.cos
        elif link_name == "quadratic":
            self.func = lambda single_index: single_index + 0.25 * single_index**2
        else:
            raise ValueError(f"Unknown link: {self.link}")

        self.Y = None

        return self.func

    def Generate_beta(self):
        # Вход:
        # - n_features из состояния генератора.
        #
        # Выход:
        # - beta формы (n_features,) единичной нормы.
        #
        # Что делает:
        # - генерирует истинное направление single-index модели.
        #
        # Реализация:
        # - нормирует Gaussian-вектор и заменяет вырожденный случай базисным вектором.

        if self.n_features <= 0:
            raise ValueError("n_features must be positive.")

        beta_vector = self.rng.normal(size=self.n_features)
        beta_vector = np.asarray(beta_vector, dtype=self.dtype)

        beta_norm = np.linalg.norm(beta_vector)

        if not np.isfinite(beta_norm) or beta_norm <= self.eps:
            beta_vector = np.zeros(self.n_features, dtype=self.dtype)
            beta_vector[0] = 1.0
            beta_norm = 1.0

        self.beta = beta_vector / beta_norm
        self.Y = None

        return self.beta

    def Generate_Y(self):
        # Вход:
        # - X, beta, func и noise из состояния генератора или параметры для их создания.
        #
        # Выход:
        # - Y формы (n_samples,).
        #
        # Что делает:
        # - собирает single-index ответы Y = f(X beta) + noise.
        #
        # Реализация:
        # - недостающие компоненты создаются лениво и сохраняются в объекте.

        if self.X is None:
            self.Generate_X()

        if self.beta is None:
            self.Generate_beta()

        if self.func is None:
            self.Generate_func()

        if self.noise is None:
            self.Generate_Noise()

        single_index = self.X @ self.beta
        signal = self.func(single_index)

        self.Y = np.asarray(signal, dtype=self.dtype) + self.noise
        self.Y = np.asarray(self.Y, dtype=self.dtype).reshape(-1)

        return self.Y

    def Generate_Centers(self):
        # Вход:
        # - X из состояния генератора или параметры для его создания.
        #
        # Выход:
        # - centers формы (n_centers, n_features).
        #
        # Что делает:
        # - выбирает локальные центры из уже сгенерированных наблюдений.
        #
        # Реализация:
        # - выбор без повторений гарантирует непустую окрестность каждого центра.

        if self.X is None:
            self.Generate_X()

        if self.n_centers <= 0:
            raise ValueError("n_centers must be positive.")

        number_of_centers = min(self.n_centers, self.X.shape[0])

        self.center_indices = self.rng.choice(
            self.X.shape[0],
            size=number_of_centers,
            replace=False,
        )
        self.centers = np.asarray(self.X[self.center_indices], dtype=self.dtype)

        return self.centers
