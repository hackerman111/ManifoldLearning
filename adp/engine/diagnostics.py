from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..common.plotting import (
    apply_adp_axis_style,
    configure_adp_matplotlib,
    format_adp_legend,
    prepare_adp_axis,
    save_figure,
    set_adp_figure_size,
    set_integer_x_ticks,
)
from ..common.types import ADPResult
from ..common.utils import unit_vector


class DiagnosticsMixin:
    """Оценка качества, summary и диагностические графики."""

    def score(
        self,
        beta_true: np.ndarray,  # Истинное EDR-направление.
    ) -> dict[str, float]:
        """Считает метрики восстановления направления beta.

        Вход:
            beta_true: истинное направление beta.
        Выход:
            Словарь cosine, cosine_abs, angle_deg и signed_l2.
        """

        result = self._require_result()
        expected = unit_vector(beta_true)
        estimated = unit_vector(result.beta)
        cosine = float(np.clip(expected @ estimated, -1.0, 1.0))
        cosine_abs = abs(cosine)
        signed_l2 = min(np.linalg.norm(estimated - expected), np.linalg.norm(estimated + expected))
        return {
            "cosine": cosine,
            "cosine_abs": cosine_abs,
            "angle_deg": float(np.degrees(np.arccos(np.clip(cosine_abs, -1.0, 1.0)))),
            "signed_l2": float(signed_l2),
        }

    def summary(
        self,  # Обученная модель ADP.
    ) -> dict[str, Any]:
        """Возвращает краткую сводку обученной модели.

        Вход:
            self: модель после fit(...).
        Выход:
            Словарь параметров, прогресса, timings и путей графиков.
        """

        result = self._require_result()
        n_directions = result.statistics.n_directions
        if n_directions is None and result.statistics.directions is not None:
            n_directions = int(result.statistics.directions.shape[1])
        return {
            "variant": self.variant,
            "backend": result.backend,
            "n_centers": int(result.statistics.centers.shape[0]),
            "n_directions": n_directions,
            "h": float(result.statistics.h),
            "weights_mean": float(result.statistics.weights_mean),
            "objective": float(result.objective),
            "progress_last": dict(result.progress[-1]) if result.progress else None,
            "diagnostic_plots": {name: str(path) for name, path in self.diagnostic_plots_.items()},
            "timings": dict(result.timings),
        }

    def plot_history(
        self,
        ax: Any = None,  # Существующая ось графика или None.
        *,
        save_path: str | Path | None = None,  # Путь сохранения или None.
        dpi: int = 150,  # Разрешение изображения.
        close: bool = False,  # Закрыть рисунок после сохранения.
    ) -> Any:
        """Рисует историю objective.

        Вход:
            ax: axis для рисования или None.
            save_path: путь сохранения или None.
            dpi: разрешение.
            close: закрывать ли figure.
        Выход:
            Axis с графиком.
        """

        result = self._require_result()
        configure_adp_matplotlib()
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
            set_adp_figure_size(fig)
        prepare_adp_axis(ax)
        ax.plot([step.objective for step in result.history], marker="o", linewidth=2.1, markersize=5)
        apply_adp_axis_style(
            ax,
            xlabel="итерация",
            ylabel="целевая функция",
            title="История сходимости ADP",
        )
        if save_path is not None:
            saved_path = save_figure(ax.figure, save_path, dpi=dpi, close=close)
            self._remember_diagnostic_plot("history", saved_path)
        return ax

    def save_diagnostics(
        self,
        output_dir: str | Path,  # Каталог для изображений.
        *,
        beta_true: np.ndarray | None = None,  # Истинное beta для сравнения.
        prefix: str = "adp",  # Префикс имен файлов.
        dpi: int = 150,  # Разрешение изображения.
        close: bool = True,  # Закрывать рисунки после сохранения.
    ) -> dict[str, Path]:
        """Строит и сохраняет стандартные диагностические графики.

        Вход:
            output_dir: каталог для графиков.
            beta_true: истинное направление beta или None.
            prefix: префикс файлов.
            dpi: разрешение.
            close: закрывать ли figures.
        Выход:
            Словарь имя_графика -> Path.
        """

        result = self._require_result()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        configure_adp_matplotlib()
        import matplotlib.pyplot as plt

        saved: dict[str, Path] = {}
        iterations = np.arange(len(result.history))

        # График цели показывает, как попеременный решатель уменьшал целевую
        # функцию после исключения наблюдаемых величин в manifold_new.tex.
        fig, ax = plt.subplots()
        set_adp_figure_size(fig)
        prepare_adp_axis(ax)
        ax.plot(iterations, [step.objective for step in result.history], marker="o", linewidth=2.1, markersize=5)
        apply_adp_axis_style(
            ax,
            xlabel="итерация",
            ylabel="целевая функция",
            title="Целевая функция ADP",
        )
        saved["objective"] = save_figure(fig, output_path / f"{prefix}_objective.png", dpi=dpi, close=close)

        # Изменение показывает стабилизацию направления beta между внутренними шагами.
        fig, ax = plt.subplots()
        set_adp_figure_size(fig)
        prepare_adp_axis(ax)
        ax.plot(iterations, [step.beta_delta for step in result.history], marker="o", linewidth=2.1, markersize=5)
        ax.set_yscale("log")
        apply_adp_axis_style(
            ax,
            xlabel="итерация",
            ylabel="||beta_k - beta_{k-1}||",
            title="Стабилизация направления beta",
        )
        saved["delta"] = save_figure(fig, output_path / f"{prefix}_delta.png", dpi=dpi, close=close)

        # График масштабов полезен для проверки адаптивной локализации через rho.
        outer = np.arange(1, len(result.progress) + 1)
        # Средняя масса весов должна оставаться около min_neighbors, иначе
        # локальные задачи наименьших квадратов становятся плохо обусловленными.
        fig, ax = plt.subplots()
        set_adp_figure_size(fig)
        prepare_adp_axis(ax)
        ax.plot(outer, [record["h"] for record in result.progress], marker="o", linewidth=2.1, markersize=5, label="h")
        if any("rho" in record for record in result.progress):
            ax.plot(
                outer,
                [record.get("rho", np.nan) for record in result.progress],
                marker="s",
                linewidth=2.1,
                markersize=5,
                label="rho",
            )
        ax.legend()
        apply_adp_axis_style(
            ax,
            xlabel="внешний шаг",
            ylabel="масштаб",
            title="Масштабы локализации ADP",
            legend_title="параметр",
        )
        saved["bandwidth"] = save_figure(fig, output_path / f"{prefix}_bandwidth.png", dpi=dpi, close=close)

        fig, ax = plt.subplots()
        set_adp_figure_size(fig)
        prepare_adp_axis(ax)
        ax.plot(outer, [record["weights"] for record in result.progress], marker="o", linewidth=2.1, markersize=5)
        ax.axhline(self.config.min_neighbors, color="#dc2626", linestyle="--", linewidth=1.4, label="min_neighbors")
        ax.legend()
        apply_adp_axis_style(
            ax,
            xlabel="внешний шаг",
            ylabel="средняя локальная масса",
            title="Локальная масса весов ADP",
            legend_title="ориентир",
        )
        saved["weights"] = save_figure(fig, output_path / f"{prefix}_weights.png", dpi=dpi, close=close)

        if beta_true is not None:
            # Знак EDR-направления не идентифицируется, поэтому перед сравнением
            # разворачиваем оценку к истинному beta.
            expected = unit_vector(beta_true)
            estimated = unit_vector(result.beta)
            if expected @ estimated < 0:
                estimated = -estimated
            x = np.arange(estimated.size)
            fig, ax = plt.subplots()
            set_adp_figure_size(fig, width=max(8.0, 0.35 * estimated.size), height=4.8)
            prepare_adp_axis(ax)
            width = 0.4
            ax.bar(x - width / 2, expected, width=width, label="истинное", edgecolor="#ffffff", linewidth=0.8)
            ax.bar(x + width / 2, estimated, width=width, label="оценка", edgecolor="#ffffff", linewidth=0.8)
            set_integer_x_ticks(ax, count=estimated.size)
            ax.legend()
            apply_adp_axis_style(
                ax,
                xlabel="компонента",
                ylabel="значение",
                title="Сравнение направления beta",
                legend_title="направление",
            )
            format_adp_legend(ax, title="направление")
            saved["beta_compare"] = save_figure(fig, output_path / f"{prefix}_beta_compare.png", dpi=dpi, close=close)

        for name, path in saved.items():
            self._remember_diagnostic_plot(name, path)
        return saved

    def _remember_diagnostic_plot(
        self,
        name: str,  # Логическое имя графика.
        path: Path,  # Путь к сохраненному изображению.
    ) -> None:
        """Запоминает путь к диагностическому графику.

        Вход:
            name: имя графика.
            path: путь к файлу.
        Выход:
            None; обновляет модель и ADPResult.
        """

        self.diagnostic_plots_[name] = path
        if self.result_ is not None:
            self.result_.diagnostic_plots[name] = path

    def _require_result(
        self,  # Модель ADP.
    ) -> ADPResult:
        """Возвращает обученный результат или выбрасывает ошибку.

        Вход:
            self: модель ADP.
        Выход:
            Последний ADPResult.
        """

        if self.result_ is None:
            raise RuntimeError("Сначала вызовите fit(...)")
        return self.result_
