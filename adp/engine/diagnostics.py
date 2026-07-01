from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..common.plotting import ensure_matplotlib_config_dir, save_figure
from ..common.types import ADPResult
from ..common.utils import unit_vector

class DiagnosticsMixin:
    """Методы оценки результата и сохранения диагностик."""

    def score(self, beta_true: np.ndarray) -> dict[str, float]:
        """Считает метрики восстановления направления beta.

        Вход:
            beta_true: истинное EDR-направление.
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

    def summary(self) -> dict[str, Any]:
        """Возвращает краткую сводку обученной модели.

        Вход:
            self: обученная модель ADP.
        Выход:
            Словарь с параметрами варианта, прогрессом и путями диагностик.
        """

        result = self._require_result()
        return {
            "variant": self.variant,
            "backend": result.backend,
            "n_centers": int(result.statistics.centers.shape[0]),
            "n_directions": None if result.statistics.directions is None else int(result.statistics.directions.shape[1]),
            "h": float(result.statistics.h),
            "weights_mean": float(result.statistics.weights_mean),
            "objective": float(result.objective),
            "progress_last": dict(result.progress[-1]) if result.progress else None,
            "diagnostic_plots": {name: str(path) for name, path in self.diagnostic_plots_.items()},
            "timings": dict(result.timings),
        }

    def plot_history(self, ax: Any = None, *, save_path: str | Path | None = None, dpi: int = 150, close: bool = False) -> Any:
        """Рисует историю objective.

        Вход:
            ax: существующая matplotlib axis или None.
            save_path: путь сохранения графика или None.
            dpi: разрешение сохраняемого изображения.
            close: закрыть figure после сохранения.
        Выход:
            Axis с нарисованной историей.
        """

        result = self._require_result()
        ensure_matplotlib_config_dir()
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ax.plot([step.objective for step in result.history], marker="o")
        ax.set_xlabel("iteration")
        ax.set_ylabel("objective")
        ax.set_title(f"ADP {self.variant}")
        if save_path is not None:
            saved_path = save_figure(ax.figure, save_path, dpi=dpi, close=close)
            self._remember_diagnostic_plot("history", saved_path)
        return ax

    def save_diagnostics(
        self,
        output_dir: str | Path,
        *,
        beta_true: np.ndarray | None = None,
        prefix: str = "adp",
        dpi: int = 150,
        close: bool = True,
    ) -> dict[str, Path]:
        """Строит и сохраняет стандартные диагностические графики.

        Вход:
            output_dir: каталог для PNG-файлов.
            beta_true: истинное направление beta для сравнительного графика.
            prefix: префикс имён файлов.
            dpi: разрешение изображений.
            close: закрывать figures после сохранения.
        Выход:
            Словарь имя_графика -> путь.
        """

        result = self._require_result()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        ensure_matplotlib_config_dir()
        import matplotlib.pyplot as plt

        saved: dict[str, Path] = {}
        iterations = np.arange(len(result.history))

        fig, ax = plt.subplots()
        ax.plot(iterations, [step.objective for step in result.history], marker="o")
        ax.set_xlabel("iteration")
        ax.set_ylabel("objective")
        ax.set_title("ADP objective")
        saved["objective"] = save_figure(fig, output_path / f"{prefix}_objective.png", dpi=dpi, close=close)

        fig, ax = plt.subplots()
        ax.plot(iterations, [step.beta_delta for step in result.history], marker="o")
        ax.set_xlabel("iteration")
        ax.set_ylabel("||beta_k - beta_{k-1}||")
        ax.set_title("ADP beta update")
        ax.set_yscale("log")
        saved["delta"] = save_figure(fig, output_path / f"{prefix}_delta.png", dpi=dpi, close=close)

        outer = np.arange(1, len(result.progress) + 1)
        fig, ax = plt.subplots()
        ax.plot(outer, [record["h"] for record in result.progress], marker="o", label="h")
        if any("rho" in record for record in result.progress):
            ax.plot(outer, [record.get("rho", np.nan) for record in result.progress], marker="s", label="rho")
        if any("b" in record for record in result.progress):
            ax.plot(outer, [record.get("b", np.nan) for record in result.progress], marker="s", label="b")
        ax.set_xlabel("outer step")
        ax.set_ylabel("scale")
        ax.set_title("ADP localization scales")
        ax.legend()
        saved["bandwidth"] = save_figure(fig, output_path / f"{prefix}_bandwidth.png", dpi=dpi, close=close)

        fig, ax = plt.subplots()
        ax.plot(outer, [record["weights"] for record in result.progress], marker="o")
        ax.axhline(self.config.min_neighbors, color="tab:red", linestyle="--", linewidth=1, label="min_neighbors")
        ax.set_xlabel("outer step")
        ax.set_ylabel("average local weight")
        ax.set_title("ADP local mass")
        ax.legend()
        saved["weights"] = save_figure(fig, output_path / f"{prefix}_weights.png", dpi=dpi, close=close)

        if beta_true is not None:
            expected = unit_vector(beta_true)
            estimated = unit_vector(result.beta)
            if expected @ estimated < 0:
                estimated = -estimated
            x = np.arange(estimated.size)
            fig, ax = plt.subplots()
            width = 0.4
            ax.bar(x - width / 2, expected, width=width, label="true")
            ax.bar(x + width / 2, estimated, width=width, label="estimated")
            ax.set_xlabel("component")
            ax.set_ylabel("value")
            ax.set_title("ADP beta comparison")
            ax.legend()
            saved["beta_compare"] = save_figure(fig, output_path / f"{prefix}_beta_compare.png", dpi=dpi, close=close)

        for name, path in saved.items():
            self._remember_diagnostic_plot(name, path)
        return saved

    def _remember_diagnostic_plot(self, name: str, path: Path) -> None:
        """Запоминает путь к диагностическому графику.

        Вход:
            name: логическое имя графика.
            path: путь к сохранённому файлу.
        Выход:
            None; обновляет модель и текущий ADPResult.
        """

        self.diagnostic_plots_[name] = path
        if self.result_ is not None:
            self.result_.diagnostic_plots[name] = path

    def _require_result(self) -> ADPResult:
        """Возвращает обученный результат или ошибку.

        Вход:
            self: модель ADP.
        Выход:
            ADPResult последнего fit.
        """

        if self.result_ is None:
            raise RuntimeError("Сначала вызовите fit(...)")
        return self.result_
