"""Publication-size figures for the paired recovery grid; PDFs and PNGs."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from experiments.mave_grassmann_grid import OUT, C, G, M

COLORS = {G: "#167b63", M: "#c65d16", C: "#656565"}


def main():
    folder = OUT / "main"
    records = [json.loads(p.read_text()) for p in (folder / "fits").glob("*/result.json")]
    data = pd.DataFrame(records)
    data = data[data.status == "ok"]
    gkeys = set(zip(data[data.method == G]["case"], data[data.method == G].seed, strict=True))
    mkeys = set(zip(data[data.method == M]["case"], data[data.method == M].seed, strict=True))
    common = gkeys & mkeys
    data = data[[((c, s) in common) for c, s in zip(data["case"], data.seed, strict=True)]]
    core = data[(data.block == "core") & data.method.isin([G, M])]
    dest = OUT / "figures"
    dest.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 120,
                         "savefig.dpi": 180})
    saved = []

    def save(fig, name):
        fig.savefig(dest / f"{name}.png", bbox_inches="tight")
        fig.savefig(dest / f"{name}.pdf", bbox_inches="tight")
        saved.append(fig)

    def curve(ax, subset, axis, values, method, label=None, color=None, linestyle="-"):
        grouped = subset[subset.method == method].groupby(axis).projector_distance
        med = grouped.median().reindex(values)
        lo = grouped.quantile(.25).reindex(values)
        hi = grouped.quantile(.75).reindex(values)
        color = color or COLORS[method]
        ax.plot(values, med, marker="o", markersize=3, color=color,
                label=label or method, linestyle=linestyle)
        ax.fill_between(values, lo, hi, color=color, alpha=.08)
        ax.grid(alpha=.15)

    fig, axes = plt.subplots(4, 3, figsize=(13, 13), sharex=True, sharey=True)
    dcolors = {10: "#1965b0", 25: "#e28b12", 50: "#88529e"}
    for i, m in enumerate((1, 2, 3, 5)):
        for j, sigma in enumerate((0., .3, 1.)):
            ax = axes[i, j]
            for d in (10, 25, 50):
                subset = core[(core.m == m) & (core.noise == sigma) & (core.d == d)]
                for method in (G, M):
                    curve(ax, subset, "n", [300, 1000, 3000], method,
                          f"d={d}, {method}", dcolors[d], "-" if method == G else "--")
            ax.set_xscale("log")
            ax.set_ylim(0, 1)
            ax.set_title(f"m={m}, σ={sigma:g}")
            if j == 0:
                ax.set_ylabel("EDR (медиана, IQR)")
            if i == 3:
                ax.set_xlabel("Число наблюдений n")
                ax.set_xticks([300, 1000, 3000], ["300", "1000", "3000"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Восстановление при росте n: сплошная — Grassmann, пунктир — MEANMAVE\n"
                 "Частичная сетка: одинаковые полные пары; число повторов меняется", y=.995)
    fig.tight_layout(rect=(0, .055, 1, .96))
    save(fig, "sample_size")

    fig, axes = plt.subplots(4, 3, figsize=(12, 12), sharex=True, sharey=True)
    for i, m in enumerate((1, 2, 3, 5)):
        for j, d in enumerate((10, 25, 50)):
            ax = axes[i, j]
            subset = core[(core.n == 1000) & (core.m == m) & (core.d == d)]
            for method in (G, M):
                curve(ax, subset, "noise", [0., .3, 1.], method)
            ax.set_title(f"d={d}, m={m}")
            ax.set_ylim(0, 1)
            if j == 0:
                ax.set_ylabel("EDR (медиана, IQR)")
            if i == 3:
                ax.set_xlabel("SD добавленного шума σ")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Чувствительность к шуму при n=1000; дисперсия сигнала = 1", y=.995)
    fig.tight_layout(rect=(0, .04, 1, .97))
    save(fig, "noise")

    pairs = core.pivot(index=["case", "seed", "d", "m", "n", "noise"],
                       columns="method", values="projector_distance").reset_index()
    pairs["delta"] = pairs[G] - pairs[M]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharey=True)
    cols = [(d, n) for d in (10, 25, 50) for n in (300, 1000, 3000)]
    for j, sigma in enumerate((0., .3, 1.)):
        table = pairs[pairs.noise == sigma].groupby(["m", "d", "n"]).delta.mean()
        matrix = np.array([[table.get((m, d, n), np.nan) for d, n in cols]
                           for m in (1, 2, 3, 5)])
        ax = axes[j]
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-.5, vmax=.5, aspect="auto")
        for row in range(4):
            for col in range(9):
                value = matrix[row, col]
                count = len(pairs[(pairs.noise == sigma) & (pairs.m == (1, 2, 3, 5)[row])
                                  & (pairs.d == cols[col][0]) & (pairs.n == cols[col][1])])
                label = f"{value:+.2f}\nk={count}" if np.isfinite(value) else "—"
                ax.text(col, row, label, ha="center", va="center", fontsize=7,
                        color="white" if abs(value) > .33 else "black")
        ax.set_title(f"σ={sigma:g}")
        ax.set_xticks(range(9), [f"{d}/{n}" for d, n in cols], rotation=60, ha="right")
        ax.set_yticks(range(4), ["m=1", "m=2", "m=3", "m=5"])
        ax.set_xlabel("d / n")
    colorbar = fig.add_axes((.90, .25, .012, .53))
    fig.colorbar(im, cax=colorbar, label="Средняя парная ΔEDR: G − M")
    fig.suptitle("Красный: MEANMAVE точнее; синий: Grassmann точнее\n"
                 "k — число полных пар в клетке; предварительные описательные результаты")
    fig.subplots_adjust(top=.8, bottom=.25, right=.86, wspace=.12)
    save(fig, "paired_heatmap")

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    links = data[(data.d == 25) & (data.n == 1000) & data.method.isin([G, M])]
    for i, m in enumerate((2, 3, 5)):
        for j, sigma in enumerate((0., .3, 1.)):
            ax = axes[i, j]
            for method in (G, M):
                grouped = links[(links.m == m) & (links.noise == sigma)
                                & (links.method == method)].groupby("link").projector_distance
                order = ["additive", "sine", "quadratic"]
                median = grouped.median().reindex(order).to_numpy()
                low = grouped.quantile(.25).reindex(order).to_numpy()
                high = grouped.quantile(.75).reindex(order).to_numpy()
                offset = -.08 if method == G else .08
                ax.errorbar(np.arange(3)+offset, median, yerr=[median-low, high-median],
                            marker="o", linestyle="none", color=COLORS[method], label=method, capsize=3)
            ax.set_title(f"m={m}, σ={sigma:g}")
            ax.set_ylim(0, 1)
            ax.grid(alpha=.15)
            ax.set_xticks(range(3), ["additive", "sine", "quadratic"])
            if j == 0:
                ax.set_ylabel("EDR (медиана, IQR)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Зависимость от функции отклика: d=25, n=1000\n"
                 "Частичный запуск; медианы/IQR полных пар, число повторов меняется")
    fig.tight_layout(rect=(0, .04, 1, .96))
    save(fig, "link_function")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    means = core.groupby(["case", "method", "m"])[["projector_distance", "fit_seconds"]].mean().reset_index()
    for method in (G, M):
        part = means[means.method == method]
        axes[0].scatter(part.fit_seconds, part.projector_distance, s=25,
                        alpha=.65, color=COLORS[method], label=method)
    axes[0].set(xscale="log", xlabel="Среднее fit time, s", ylabel="Средняя EDR",
                title="Качество и стоимость: наблюдаемые core-клетки")
    axes[0].legend()
    times = core.pivot(index=["case", "seed", "m"], columns="method", values="fit_seconds").reset_index()
    times["ratio"] = times[M] / times[G]
    values = [times[times.m == m].ratio.dropna().to_numpy() for m in (1, 2, 3, 5)]
    axes[1].boxplot(values, tick_labels=["1", "2", "3", "5"], showfliers=False)
    axes[1].axhline(1, color="gray", linestyle="--")
    axes[1].set(yscale="log", xlabel="m", ylabel="MEANMAVE time / Grassmann time",
                title="Парное отношение времени по core-клеткам")
    for ax in axes:
        ax.grid(alpha=.15)
    fig.tight_layout()
    save(fig, "quality_time")

    with PdfPages(dest / "all_figures.pdf") as pdf:
        for fig in saved:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved {len(saved)} figures (PNG/PDF) and all_figures.pdf in {dest}")


if __name__ == "__main__":
    main()
