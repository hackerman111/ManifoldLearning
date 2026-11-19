"""Воспроизводимый парный бенчмарк LOO1-WLS-ADP против LiveADP.

Запускать в один поток BLAS из корня worktree:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        uv run --no-sync python benchmarks/experiments/loo1_wls_study.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "benchmarks" / "results" / "loo1_wls_study"
DEFAULT_REPORT = ROOT / "benchmarks" / "experiments" / "LOO1_WLS_REPORT.md"

THREAD_ENV: dict[str, str] = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

SEVEN_FAMILY_NAMES: tuple[str, ...] = (
    "additive",
    "quadratic",
    "multiplicative",
    "oscillatory",
    "saturating",
    "localized",
    "nonsmooth",
)


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """Спецификация сравниваемого варианта ADP."""

    name: str
    estimator: str
    spectrum_exponent: float | None = None
    max_steps: int = 20
    lambda_penalty: float = 0.05
    stream_distances: bool | None = None


ARMS_WITH_ABLATION: tuple[ArmSpec, ...] = (
    ArmSpec(
        name="LiveADP",
        estimator="new",
        spectrum_exponent=None,
        max_steps=20,
        lambda_penalty=0.05,
    ),
    ArmSpec(
        name="LOO1-WLS (beta=0.5)",
        estimator="loo1_wls",
        spectrum_exponent=0.5,
        max_steps=25,
    ),
    ArmSpec(
        name="LOO1-WLS (beta=2/3)",
        estimator="loo1_wls",
        spectrum_exponent=2.0 / 3.0,
        max_steps=25,
    ),
)

ARMS_PRIMARY: tuple[ArmSpec, ...] = (
    ARMS_WITH_ABLATION[0],
    ARMS_WITH_ABLATION[1],
)


def _pin_threads() -> None:
    """Закрепить однопоточный BLAS до первого импорта NumPy/SciPy."""
    os.environ.update(THREAD_ENV)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _read_vmhwm_mib() -> float | None:
    """Прочитать пиковую резидентную память VmHWM из ``/proc/self/status`` в MiB."""
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    for line in status_path.read_text().splitlines():
        if line.startswith("VmHWM:"):
            parts = line.split()
            if len(parts) >= 2:
                return float(parts[1]) / 1024.0
    return None


def _read_ru_maxrss_mib() -> float:
    """Прочитать ``ru_maxrss`` текущего процесса в MiB."""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _data_sha256(X: Any, y: Any, truth: Any) -> str:
    """Посчитать контрольную сумму сгенерированных массивов для контроля парности."""
    import numpy as np

    hasher = hashlib.sha256()
    for arr in (X, y, truth):
        hasher.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return hasher.hexdigest()


def _suite1_case_index(name: str) -> int:
    """Return canonical case index matching ``pair_adp/study.py``."""
    mapping = {"Weak(8,2)": 0, "Mixture(8,2)": 3}
    if name not in mapping:
        raise ValueError(f"unknown suite-1 case: {name}")
    return mapping[name]


def _suite1_fit_seed(name: str, seed: int) -> int:
    """Return fit seed matching ``pair_adp/study.py`` channel 3."""
    import numpy as np

    case_idx = _suite1_case_index(name)
    return int(np.random.SeedSequence([seed, case_idx, 3]).generate_state(1)[0])


def _generate_suite1_case(name: str, seed: int) -> tuple[Any, Any, Any]:
    """Generate paired ``(X, y, B_star)`` matching ``pair_adp/study.py``."""
    import numpy as np

    case_idx = _suite1_case_index(name)
    basis_rng, x_rng, noise_rng = (
        np.random.default_rng(np.random.SeedSequence([seed, case_idx, ch]))
        for ch in range(3)
    )

    n, d, m = 960, 8, 2
    q, r = np.linalg.qr(basis_rng.normal(size=(d, m)))
    truth = q * np.where(np.diag(r) < 0, -1.0, 1.0)
    X = x_rng.normal(size=(n, d))
    if case_idx == 3:
        X = (X + x_rng.choice([-1.0, 1.0], size=X.shape)) / math.sqrt(2.0)

    z = X @ truth
    signal = np.sin(z[:, 0]) + 0.25 * np.cos(1.5 * z[:, 1])
    y = signal + 0.3 * noise_rng.normal(size=n)
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), truth


def _generate_case_data(
    suite: str,
    case_name: str,
    n: int,
    d: int,
    m: int,
    seed: int,
) -> tuple[Any, Any, Any]:
    """Сгенерировать парный датасет для заданного suite/case/seed."""
    if suite == "suite1":
        return _generate_suite1_case(case_name, seed)

    from experiments.r_sdr.design import catalog, generate

    base_cases = {c.name: c for c in catalog("seven-families")}
    full_name = case_name if case_name.startswith("family-") else f"family-{case_name}"
    if full_name not in base_cases:
        raise ValueError(f"unknown seven-family case: {case_name}")
    case = replace(base_cases[full_name], n=n, d=d, m=m)
    return generate(case, seed)


def _fit_single(
    X: Any,
    y: Any,
    truth: Any,
    *,
    m: int,
    n_j: int,
    h_min: float,
    n_loc: int,
    seed: int,
    arm: ArmSpec,
) -> dict[str, Any]:
    """Выполнить один fit заданного ``ArmSpec`` и вернуть метрики."""
    import numpy as np
    from experiments.r_sdr.design import metrics

    from ADP import ADP_Config, ADP_multi_index, ADP_solver, LOO1WLSConfig
    from ADP.solver.grassman import solve as grassman_solve

    if arm.estimator == "new":
        cfg = ADP_Config(
            seed=seed,
            N_J=n_j,
            h_min=h_min,
            N_loc=n_loc,
            lambda_penalty=arm.lambda_penalty,
        )
        solver = ADP_solver(grassman_solve, max_steps=arm.max_steps)
    elif arm.estimator == "loo1_wls":
        assert arm.spectrum_exponent is not None
        loo_cfg = LOO1WLSConfig(
            spectrum_exponent=arm.spectrum_exponent,
            max_steps=arm.max_steps,
            stream_distances=arm.stream_distances,
        )
        cfg = ADP_Config(
            seed=seed,
            N_J=n_j,
            h_min=h_min,
            N_loc=n_loc,
            estimator="loo1_wls",
            loo1_wls=loo_cfg,
        )
        solver = None
    else:
        raise ValueError(f"unsupported estimator: {arm.estimator}")

    model = ADP_multi_index(m, cfg, solver)
    t0 = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - t0

    assert model.basis_ is not None
    basis = np.asarray(model.basis_, dtype=np.float64)
    measured = metrics(basis, truth)
    diag = model.solver_diagnostics_
    eff = model.effective_parameters_
    return {
        "status": "ok",
        "error": None,
        "fit_seconds": float(elapsed),
        "edr": float(measured["projector_distance"]),
        "max_sin_angle": float(measured["max_sin_angle"]),
        "mean_angle_deg": float(measured["mean_angle_deg"]),
        "max_angle_deg": float(measured["max_angle_deg"]),
        "orthogonality_error": float(measured["orthogonality_error"]),
        "recovered_01": bool(measured["projector_distance"] <= 0.1),
        "outer_iterations": cast(int, eff.get("outer_iterations", len(model.trace_))),
        "selected_iteration": cast(int, eff.get("selected_iteration", -1)),
        "stop_reason": str(eff.get("stop_reason", "")),
        "n_inner_certified": int(sum(d.get("converged") is True for d in diag)),
        "n_inner_total": len(diag),
        "ru_maxrss_mib": _read_ru_maxrss_mib(),
        "vmhwm_mib": _read_vmhwm_mib(),
    }


def _run_job_dict(job: dict[str, Any]) -> dict[str, Any]:
    """Run one benchmark job in-process or inside an isolated worker."""
    suite = str(job["suite"])
    case_name = str(job["case"])
    n = int(job["n"])
    d = int(job["d"])
    m = int(job["m"])
    n_j = int(job["n_j"])
    h_min = float(job["h_min"])
    n_loc = int(job["n_loc"])
    seed = int(job["seed"])
    fit_seed = int(job["fit_seed"])
    arm = ArmSpec(**job["arm"])

    row: dict[str, Any] = {
        "suite": suite,
        "case": case_name,
        "n": n,
        "d": d,
        "m": m,
        "n_j": n_j,
        "h_min": h_min,
        "n_loc": n_loc,
        "seed": seed,
        "fit_seed": fit_seed,
        "arm": arm.name,
        "estimator": arm.estimator,
        "spectrum_exponent": arm.spectrum_exponent,
        "stream_distances": arm.stream_distances,
    }
    try:
        X, y, truth = _generate_case_data(suite, case_name, n, d, m, seed)
        row["data_sha256"] = _data_sha256(X, y, truth)
        res = _fit_single(
            X,
            y,
            truth,
            m=m,
            n_j=n_j,
            h_min=h_min,
            n_loc=n_loc,
            seed=fit_seed,
            arm=arm,
        )
        row.update(res)
    except Exception as exc:
        row.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "fit_seconds": None,
                "edr": None,
                "recovered_01": False,
                "ru_maxrss_mib": _read_ru_maxrss_mib(),
                "vmhwm_mib": _read_vmhwm_mib(),
            }
        )
    return row


def _run_isolated(job: dict[str, Any], *, timeout: float = 600.0) -> dict[str, Any]:
    """Запустить job в свежем подпроцессе для чистого замера VmHWM/ru_maxrss."""
    env = {**os.environ, **THREAD_ENV}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        job_path = tmp / "job.json"
        out_path = tmp / "out.json"
        job_path.write_text(json.dumps(job))
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-job",
            str(job_path),
            "--worker-out",
            str(out_path),
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        wall = time.perf_counter() - t0
        if proc.returncode != 0 or not out_path.exists():
            return {
                "suite": job["suite"],
                "case": job["case"],
                "n": job["n"],
                "d": job["d"],
                "m": job["m"],
                "n_j": job["n_j"],
                "h_min": job["h_min"],
                "n_loc": job["n_loc"],
                "seed": job["seed"],
                "arm": job["arm"]["name"],
                "status": "error",
                "error": (
                    f"worker failed (rc={proc.returncode}): "
                    f"{proc.stderr.strip()[-400:]}"
                ),
                "process_seconds": wall,
            }
        res = json.loads(out_path.read_text())
        res["process_seconds"] = wall
        return res


def build_jobs(suites: set[str]) -> list[dict[str, Any]]:
    """Build the full list of paired benchmark jobs for the selected suites."""
    jobs: list[dict[str, Any]] = []

    if "suite1" in suites:
        jobs.extend(
            {
                "suite": "suite1",
                "case": case_name,
                "n": 960,
                "d": 8,
                "m": 2,
                "n_j": 816,
                "h_min": 0.7,
                "n_loc": 20,
                "seed": seed,
                "fit_seed": _suite1_fit_seed(case_name, seed),
                "arm": asdict(arm),
                "isolated": False,
            }
            for case_name in ("Weak(8,2)", "Mixture(8,2)")
            for seed in range(4111800, 4111806)
            for arm in ARMS_WITH_ABLATION
        )

    if "suite2" in suites:
        jobs.extend(
            {
                "suite": "suite2",
                "case": f"family-{family}",
                "n": 1000,
                "d": 25,
                "m": 3,
                "n_j": 850,
                "h_min": 0.5,
                "n_loc": 30,
                "seed": seed,
                "fit_seed": seed,
                "arm": asdict(arm),
                "isolated": False,
            }
            for family in SEVEN_FAMILY_NAMES
            for seed in range(10)
            for arm in ARMS_WITH_ABLATION
        )

    if "suite3" in suites:
        jobs.extend(
            {
                "suite": "suite3",
                "case": f"family-{family}",
                "n": 2000,
                "d": 50,
                "m": 3,
                "n_j": 1700,
                "h_min": 0.5,
                "n_loc": 30,
                "seed": seed,
                "fit_seed": seed,
                "arm": asdict(arm),
                "isolated": True,
            }
            for family in SEVEN_FAMILY_NAMES
            for seed in range(3)
            for arm in ARMS_PRIMARY
        )

    if "suite4" in suites:
        stream_arm = ArmSpec(
            name="LOO1-WLS (beta=0.5, stream=True)",
            estimator="loo1_wls",
            spectrum_exponent=0.5,
            max_steps=25,
            stream_distances=True,
        )
        dense_arm = ArmSpec(
            name="LOO1-WLS (beta=0.5, stream=False)",
            estimator="loo1_wls",
            spectrum_exponent=0.5,
            max_steps=25,
            stream_distances=False,
        )
        jobs.extend(
            {
                "suite": "suite4",
                "case": "family-quadratic",
                "n": 4000,
                "d": 100,
                "m": 3,
                "n_j": 3400,
                "h_min": 0.5,
                "n_loc": 30,
                "seed": 0,
                "fit_seed": 0,
                "arm": asdict(arm),
                "isolated": True,
            }
            for arm in (stream_arm, dense_arm)
        )

    return jobs


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Собрать сводную статистику по группе строк (включая сбои в знаменателе)."""
    import numpy as np

    ok = [r for r in rows if r.get("status") == "ok"]
    edrs = [float(r["edr"]) for r in ok]
    times = [float(r["fit_seconds"]) for r in ok]
    rss = [float(r["vmhwm_mib"]) for r in ok if r.get("vmhwm_mib") is not None]
    ru_rss = [
        float(r["ru_maxrss_mib"]) for r in ok if r.get("ru_maxrss_mib") is not None
    ]
    return {
        "n_total": len(rows),
        "n_ok": len(ok),
        "n_fail": len(rows) - len(ok),
        "recovery_01": sum(bool(r.get("recovered_01")) for r in ok),
        "mean_edr": float(np.mean(edrs)) if edrs else None,
        "median_edr": float(np.median(edrs)) if edrs else None,
        "max_edr": float(np.max(edrs)) if edrs else None,
        "mean_fit_s": float(np.mean(times)) if times else None,
        "median_fit_s": float(np.median(times)) if times else None,
        "median_vmhwm_mib": float(np.median(rss)) if rss else None,
        "max_vmhwm_mib": float(np.max(rss)) if rss else None,
        "max_ru_maxrss_mib": float(np.max(ru_rss)) if ru_rss else None,
    }


def _paired_comparison(
    rows: list[dict[str, Any]],
    baseline_arm: str,
    candidate_arm: str,
) -> dict[str, Any]:
    """Вычислить парные дельты (wins/losses, worst regression, speedup)."""
    import numpy as np

    by_key = {(r["suite"], r["case"], r["seed"], r["arm"]): r for r in rows}
    keys = sorted({(r["suite"], r["case"], r["seed"]) for r in rows})
    deltas: list[float] = []
    speedups: list[float] = []
    worst_regression: dict[str, Any] | None = None
    best_gain: dict[str, Any] | None = None
    wins = 0
    losses = 0
    ties = 0

    for suite, case, seed in keys:
        base = by_key.get((suite, case, seed, baseline_arm))
        cand = by_key.get((suite, case, seed, candidate_arm))
        if (
            base is None
            or cand is None
            or base.get("status") != "ok"
            or cand.get("status") != "ok"
        ):
            continue
        assert base["data_sha256"] == cand["data_sha256"], (
            f"unpaired dataset mismatch on {(suite, case, seed)}"
        )
        delta = float(cand["edr"]) - float(base["edr"])
        deltas.append(delta)
        if cand["fit_seconds"] and base["fit_seconds"]:
            speedups.append(float(base["fit_seconds"]) / float(cand["fit_seconds"]))
        if delta < -1e-6:
            wins += 1
        elif delta > 1e-6:
            losses += 1
        else:
            ties += 1
        entry = {
            "suite": suite,
            "case": case,
            "seed": seed,
            "base_edr": float(base["edr"]),
            "cand_edr": float(cand["edr"]),
            "delta_edr": delta,
        }
        if worst_regression is None or delta > worst_regression["delta_edr"]:
            worst_regression = entry
        if best_gain is None or delta < best_gain["delta_edr"]:
            best_gain = entry

    return {
        "n_pairs": len(deltas),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "mean_delta_edr": float(np.mean(deltas)) if deltas else None,
        "median_delta_edr": float(np.median(deltas)) if deltas else None,
        "median_speedup": float(np.median(speedups)) if speedups else None,
        "mean_speedup": float(np.mean(speedups)) if speedups else None,
        "worst_regression": worst_regression,
        "best_gain": best_gain,
    }


def _environment_info() -> dict[str, Any]:
    """Зафиксировать окружение, версии библиотек и состояние git."""
    import numpy as np
    import scipy

    def _git(*args: str) -> str:
        res = subprocess.run(
            ["git", *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        return res.stdout.strip() if res.returncode == 0 else "unknown"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--short")),
        "thread_env": dict(THREAD_ENV),
    }


def _fmt(val: float | None, fmt: str = ".4f") -> str:
    return "n/a" if val is None else format(val, fmt)


def render_report(
    rows: list[dict[str, Any]],
    env_info: dict[str, Any],
) -> str:
    """Сформировать подробный Markdown-отчёт ``LOO1_WLS_REPORT.md``."""
    lines: list[str] = [
        "# LOO1-WLS-ADP vs LiveADP Paired Benchmark Report",
        "",
        "## Protocol & Environment",
        f"- Branch / Commit: `{env_info['git_branch']}` @ `{env_info['git_commit']}` "
        f"(dirty={env_info['git_dirty']})",
        f"- Python: `{env_info['python']}`, NumPy: `{env_info['numpy']}`, "
        f"SciPy: `{env_info['scipy']}`",
        f"- Platform: `{env_info['platform']}`",
        f"- Threads: `{json.dumps(env_info['thread_env'])}`",
        "- `LiveADP`: `ADP_Config(N_J=NJ, h_min=hmin, N_loc=Nloc, "
        "lambda_penalty=0.05)`, `ADP_solver(grassman.solve, max_steps=20)`",
        "- `LOO1-WLS`: `ADP_Config(N_J=NJ, h_min=hmin, N_loc=Nloc, "
        "estimator='loo1_wls', loo1_wls=LOO1WLSConfig(spectrum_exponent=beta, "
        "max_steps=25))`",
        "- Timing: `time.perf_counter()`; RSS: `/proc/self/status` `VmHWM` and "
        "`resource.getrusage(RUSAGE_SELF).ru_maxrss`.",
        "",
    ]

    for suite, title in (
        ("suite1", "Suite 1: Weak(8,2) & Mixture(8,2) (n=960, d=8, m=2, N_J=816)"),
        ("suite2", "Suite 2: Seven Families (n=1000, d=25, m=3, N_J=850, seeds 0..9)"),
        (
            "suite3",
            "Suite 3: Seven Families Scaling (n=2000, d=50, m=3, N_J=1700, seeds 0..2)",
        ),
        (
            "suite4",
            "Suite 4: Streaming Check (n=4000, d=100, m=3, N_J=3400, quadratic seed 0)",
        ),
    ):
        srows = [r for r in rows if r["suite"] == suite]
        if not srows:
            continue
        lines.append(f"## {title}")
        lines.append("")
        arms_present = list(dict.fromkeys(r["arm"] for r in srows))
        lines.append(
            "| Arm | OK / Total | Rec <= 0.1 | Mean EDR | Median EDR | "
            "Max EDR | Mean fit (s) | Median fit (s) | Median VmHWM (MiB) | "
            "Max VmHWM (MiB) | Max ru_maxrss (MiB) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for arm_name in arms_present:
            st = _summarize_group([r for r in srows if r["arm"] == arm_name])
            lines.append(
                f"| {arm_name} | {st['n_ok']}/{st['n_total']} | "
                f"{st['recovery_01']}/{st['n_total']} | "
                f"{_fmt(st['mean_edr'])} | {_fmt(st['median_edr'])} | "
                f"{_fmt(st['max_edr'])} | {_fmt(st['mean_fit_s'], '.3f')} | "
                f"{_fmt(st['median_fit_s'], '.3f')} | "
                f"{_fmt(st['median_vmhwm_mib'], '.1f')} | "
                f"{_fmt(st['max_vmhwm_mib'], '.1f')} | "
                f"{_fmt(st['max_ru_maxrss_mib'], '.1f')} |"
            )
        lines.append("")

        if suite in {"suite1", "suite2", "suite3"}:
            lines.append("### Per-Case Breakdown")
            lines.append("")
            lines.append(
                "| Case | Arm | OK/Tot | Rec<=0.1 | Mean EDR | Median EDR | "
                "Median fit (s) | Max VmHWM (MiB) |"
            )
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
            for case_name in sorted({r["case"] for r in srows}):
                for arm_name in arms_present:
                    st = _summarize_group(
                        [
                            r
                            for r in srows
                            if r["case"] == case_name and r["arm"] == arm_name
                        ]
                    )
                    lines.append(
                        f"| {case_name} | {arm_name} | "
                        f"{st['n_ok']}/{st['n_total']} | "
                        f"{st['recovery_01']}/{st['n_total']} | "
                        f"{_fmt(st['mean_edr'])} | {_fmt(st['median_edr'])} | "
                        f"{_fmt(st['median_fit_s'], '.3f')} | "
                        f"{_fmt(st['max_vmhwm_mib'], '.1f')} |"
                    )
            lines.append("")

            if "LiveADP" in arms_present:
                lines.append("### Paired Comparison vs LiveADP")
                lines.append("")
                for cand_arm in arms_present:
                    if cand_arm == "LiveADP":
                        continue
                    pc = _paired_comparison(srows, "LiveADP", cand_arm)
                    lines.append(
                        f"- **{cand_arm}** vs `LiveADP` ({pc['n_pairs']} pairs): "
                        f"wins/losses/ties = {pc['wins']}/{pc['losses']}/{pc['ties']}, "
                        f"mean ΔEDR = {_fmt(pc['mean_delta_edr'], '+.4f')}, "
                        f"median ΔEDR = {_fmt(pc['median_delta_edr'], '+.4f')}, "
                        f"median speedup = {_fmt(pc['median_speedup'], '.2f')}x "
                        f"(mean {_fmt(pc['mean_speedup'], '.2f')}x)"
                    )
                    if pc["worst_regression"] is not None:
                        wr = pc["worst_regression"]
                        lines.append(
                            f"  - Worst regression: `{wr['case']}` seed `{wr['seed']}` "
                            f"({wr['base_edr']:.4f} -> {wr['cand_edr']:.4f}, "
                            f"Δ={wr['delta_edr']:+.4f})"
                        )
                    if pc["best_gain"] is not None:
                        bg = pc["best_gain"]
                        lines.append(
                            f"  - Best improvement: `{bg['case']}` seed `{bg['seed']}` "
                            f"({bg['base_edr']:.4f} -> {bg['cand_edr']:.4f}, "
                            f"Δ={bg['delta_edr']:+.4f})"
                        )
                lines.append("")

        failures = [r for r in srows if r.get("status") != "ok"]
        if failures:
            lines.append("### Failures")
            lines.extend(
                f"- `{frow['case']}` seed `{frow['seed']}` `{frow['arm']}`: "
                f"`{frow.get('error')}`"
                for frow in failures
            )
            lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI для запуска парного бенчмарка или worker-подпроцесса."""
    parser = argparse.ArgumentParser(description="LOO1-WLS-ADP paired benchmark")
    parser.add_argument(
        "--suites",
        default="suite1,suite2,suite3,suite4",
        help="Comma-separated suites to run (suite1,suite2,suite3,suite4)",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--worker-job", type=Path)
    parser.add_argument("--worker-out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    _pin_threads()

    if args.worker_job is not None:
        if args.worker_out is None:
            parser.error("--worker-out is required with --worker-job")
        job = json.loads(args.worker_job.read_text())
        res = _run_job_dict(job)
        args.worker_out.write_text(json.dumps(res, indent=2) + "\n")
        return 0

    suites = {s.strip() for s in args.suites.split(",") if s.strip()}
    valid = {"suite1", "suite2", "suite3", "suite4"}
    unknown = suites - valid
    if unknown:
        parser.error(f"unknown suites: {sorted(unknown)}")

    jobs = build_jobs(suites)
    if args.dry_run:
        counts: dict[str, int] = {}
        for job in jobs:
            s = str(job["suite"])
            counts[s] = counts.get(s, 0) + 1
        print(json.dumps({"total_jobs": len(jobs), "by_suite": counts}, indent=2))
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx, job in enumerate(jobs, start=1):
        if job.get("isolated"):
            row = _run_isolated(job)
        else:
            row = _run_job_dict(job)
        rows.append(row)
        status = row.get("status")
        edr = row.get("edr")
        fit_s = row.get("fit_seconds")
        print(
            f"[{idx}/{len(jobs)}] {row['suite']} {row['case']} "
            f"seed={row['seed']} {row['arm']}: "
            f"status={status} edr={_fmt(edr)} time={_fmt(fit_s, '.3f')}s",
            flush=True,
        )

    env_info = _environment_info()
    payload = {"environment": env_info, "rows": rows}
    (args.out_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    report_md = render_report(rows, env_info)
    args.report.write_text(report_md + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
