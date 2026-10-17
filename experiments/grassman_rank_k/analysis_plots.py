"""Publication-independent descriptive plots for the rank-k analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ._analysis_io import _number


def _write_plots(
    out: Path,
    fullfit: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    inner: list[dict[str, Any]],
    sweeps: list[dict[str, Any]],
    fits: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, baseline in zip(axes, ("core", "spectral"), strict=True):
        points = [
            r
            for r in paired
            if r["baseline"] == baseline
            and r["split"] == "heldout"
            and r["scenario"] != "ALL"
            and r["arm"] == "block1"
            and r["median_fit_wall_ratio"] is not None
        ]
        for row in points:
            label = (
                "Spokoiny"
                if row["scenario"].startswith("spokoiny")
                else "Multi-additive"
            )
            ax.scatter(
                row["median_edr_difference"], row["median_fit_wall_ratio"], label=label
            )
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.axhline(0.8, color="grey", linestyle="--", linewidth=0.8)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set(
            title=f"heldout block1 / {baseline}, 5 pairs per scenario",
            xlabel="median paired EDR difference",
            ylabel="median paired total-fit wall ratio",
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fullfit.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for variant, marker in (("optimized", "o"), ("spectral", "s")):
        selected = [
            r for r in sweeps if r["variant"] == variant and r.get("ratio") is not None
        ]
        for seed, color, label in (
            (81600, "tab:blue", "development"),
            (81700, "tab:orange", "heldout"),
        ):
            points = [r for r in selected if r["seed"] == seed]
            ax.scatter(
                [r["x_k_over_r"] for r in points],
                [r["ratio"] for r in points],
                marker=marker,
                color=color,
                alpha=0.7,
                label=f"{variant} {label}",
            )
        fit = next(
            (r for r in fits if r["variant"] == variant and r["m"] is None), None
        )
        if fit:
            xs = np.linspace(0, 1, 100)
            ax.plot(
                xs, np.exp(-fit["c_least_squares"] * xs), label=f"{variant} train fit"
            )
    ax.axhline(1, color="black", linewidth=0.8)
    ax.set(xlabel="k/r", ylabel="(F_new - F_ref)/(F_old - F_ref)", ylim=(0, 1.1))
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "rank_ratio.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    selected = [
        r
        for r in inner
        if r.get("seed") == 81600
        and r.get("d") == 100
        and r.get("m") == 3
        and isinstance(r.get("trace"), list)
    ]
    for arm in (
        "core",
        "rank_one",
        "spectral",
        "flat",
        "block_energy",
        "block_cost",
        "block_spectral",
    ):
        candidates = [
            r for r in selected if r.get("arm") == arm and r.get("budget") == 100
        ]
        if not candidates:
            continue
        trace = candidates[0]["trace"]
        points = [
            (elapsed, objective)
            for p in trace
            if isinstance(p, dict)
            if (elapsed := _number(p.get("elapsed_sec"))) is not None
            if (objective := _number(p.get("profile_objective"))) is not None
        ]
        if not points:
            continue
        f0 = points[0][1]
        f0_number = _number(f0)
        if f0_number is None:
            continue
        ax.plot(
            [p[0] for p in points],
            [p[1] / f0_number for p in points],
            label=f"{arm} (n=1)",
        )
    ax.set(
        xlabel="elapsed wall time (s)", ylabel="profile objective / initial objective"
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "inner_convergence.png", dpi=150)
    plt.close(fig)
