from pathlib import Path

import numpy as np
import pandas as pd

from adp import ADP, ADPConfig
from adp.evaluation.reports import plot_grouped_bars


def test_plot_history_uses_russian_labels_and_polished_style():
    model = ADP.create(
        "new",
        ADPConfig(
            n_centers=14,
            n_directions=3,
            min_neighbors=5,
            outer_steps=1,
            inner_steps=3,
            backend="numpy",
            show_progress=False,
            random_state=11,
        ),
    )
    data = model.generate_data(n=70, d=4, noise=0.01, link="linear")
    model.fit(data.X, data.y, beta0=data.beta)

    ax = model.plot_history()

    assert ax.get_xlabel() == "итерация"
    assert ax.get_ylabel() == "целевая функция"
    assert ax.get_title() == "История сходимости ADP"
    assert any(line.get_marker() == "o" and line.get_linewidth() >= 2.0 for line in ax.lines)
    assert ax.get_facecolor() != (1.0, 1.0, 1.0, 1.0)


def test_benchmark_grouped_bars_use_russian_legend_and_soft_grid():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(
        [
            {"scenario": "linear", "method": "adp_new", "cosine_abs": 0.92},
            {"scenario": "linear", "method": "sklearn_pls", "cosine_abs": 0.88},
            {"scenario": "sin", "method": "adp_new", "cosine_abs": 0.81},
            {"scenario": "sin", "method": "sklearn_pls", "cosine_abs": 0.74},
        ]
    )

    fig, ax = plt.subplots()
    plot_grouped_bars(
        ax,
        frame,
        value="cosine_abs",
        ylabel="среднее |cos(beta, beta_hat)|",
        title="Качество восстановления EDR",
    )

    assert ax.get_legend().get_title().get_text() == "метод"
    assert ax.get_facecolor() != (1.0, 1.0, 1.0, 1.0)
    assert any(line.get_visible() for line in ax.get_ygridlines())
    assert fig.get_size_inches()[0] >= 7.0
    plt.close(fig)


def test_common_plot_style_helper_configures_axis():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from adp.common.plotting import apply_adp_axis_style, set_adp_figure_size, set_integer_x_ticks

    fig, ax = plt.subplots()
    set_adp_figure_size(fig, width=7.5, height=4.5)
    set_integer_x_ticks(ax, count=4)
    apply_adp_axis_style(ax, xlabel="ось x", ylabel="ось y", title="Заголовок")

    assert ax.get_xlabel() == "ось x"
    assert ax.get_ylabel() == "ось y"
    assert ax.get_title() == "Заголовок"
    assert ax.get_facecolor() != (1.0, 1.0, 1.0, 1.0)
    assert tuple(np.round(fig.get_size_inches(), 1)) == (7.5, 4.5)
    assert list(ax.get_xticks()) == [0, 1, 2, 3]
    assert any(line.get_visible() for line in ax.get_ygridlines())
    plt.close(fig)
