"""Reproducible CPU fit profiles for the current multi and manifold models.

Run one fit per child process so ru_maxrss is a meaningful per-run peak.
Timing hooks only observe existing model calls; they do not change the estimator.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib
import json
import os
import platform
import pstats
import resource
import subprocess
import sys
import threading
import tracemalloc
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import scipy

from ADP import ADP_Config, ADP_Manifold, ADP_multi_index, ADP_solver
from ADP.solver.LSMR import solve as solve_lsmr

CASES = {
    "multi_control": ("multi", 300, 6, 24, 8, 2),
    "multi_base": ("multi", 600, 8, 48, 16, 2),
    "multi_J2": ("multi", 600, 8, 96, 16, 2),
    "manifold_control": ("manifold", 300, 4, 20, 8, 1),
    "manifold_base": ("manifold", 600, 6, 40, 12, 1),
    "manifold_J2": ("manifold", 600, 6, 80, 12, 1),
}


class RssSampler:
    """Sample process RSS during fit; ru_maxrss supplies the full-run peak."""

    def __init__(self) -> None:
        self.active = "unattributed"
        self.peaks: dict[str, int] = defaultdict(int)
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        page_size = os.sysconf("SC_PAGE_SIZE")
        with Path("/proc/self/statm").open(encoding="ascii") as statm:
            while not self._stop.is_set():
                statm.seek(0)
                resident = int(statm.readline().split()[1]) * page_size
                self.peaks[self.active] = max(self.peaks[self.active], resident)
                self.samples += 1
                self._stop.wait(0.002)

    def __enter__(self) -> RssSampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()


class TimeProfiler:
    """IndexProfiler interface without tracemalloc, for phase wall time."""

    sampler: RssSampler | None = None

    def __init__(self) -> None:
        self.times: dict[str, float] = defaultdict(float)
        self.started = perf_counter()
        self.total = 0.0

    @contextmanager
    def stage(self, name: str):
        previous = self.sampler.active if self.sampler is not None else None
        if self.sampler is not None:
            self.sampler.active = name
        started = perf_counter()
        try:
            yield
        finally:
            self.times[name] += perf_counter() - started
            if self.sampler is not None:
                self.sampler.active = previous

    def finish(self) -> None:
        self.total = perf_counter() - self.started

    def model_profile(self) -> dict:
        return {
            "stages": {k: {"time_seconds": v} for k, v in self.times.items()},
            "total_time_seconds": self.total,
        }


class NoProfiler(TimeProfiler):
    @contextmanager
    def stage(self, _name: str):
        yield


class HookTimes:
    def __init__(self, sampler: RssSampler) -> None:
        self.sampler = sampler
        self.depth = 0
        self.times: dict[str, float] = defaultdict(float)
        self.top: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def wrap(self, name: str, function):
        def measured(*args, **kwargs):
            self.counts[name] += 1
            top = self.depth == 0
            previous = self.sampler.active
            if top:
                self.sampler.active = name
            self.depth += 1
            started = perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = perf_counter() - started
                self.times[name] += elapsed
                self.depth -= 1
                if top:
                    self.top[name] += elapsed
                    self.sampler.active = previous

        return measured


def _data(n: int, d: int, model: str) -> tuple[np.ndarray, np.ndarray]:
    data_seed, noise_seed = np.random.SeedSequence(20260923).spawn(2)
    X = np.random.default_rng(data_seed).uniform(-1.0, 1.0, (n, d))
    if model == "multi":
        Y = np.sin(2.0 * X[:, 0]) + 0.7 * np.square(X[:, 1])
    else:
        Y = X[:, 0] + 0.3 * np.square(X[:, 1])
    Y += 0.02 * np.random.default_rng(noise_seed).normal(size=n)
    return X, Y


def _multi_hooks(stack: ExitStack, hooks: HookTimes) -> None:
    module = importlib.import_module("ADP.engine.common.index_fit")
    for name in (
        "pairwise_distance2",
        "initialize_basis_local_with_spectrum",
        "search_bandwidth",
        "calculate_multi_weight",
        "calculate_statistics",
        "calculate_alpha_k",
    ):
        original = getattr(module, name)
        stack.enter_context(patch.object(module, name, hooks.wrap(name, original)))


def _manifold_hooks(model: ADP_Manifold, hooks: HookTimes) -> None:
    names = (
        "_search_bandwidth",
        "_local_gradients",
        "_build_manifold_graph",
        "_initialize_projectors",
        "_random_directions",
        "_calculate_statistics",
        "_one_step",
        "_search_anisotropy",
        "_feasible_scale",
        "_mean_mass",
        "_local_slopes",
        "_build_B_system",
        "_solve_B",
        "_recover_projector",
        "_objective",
    )
    for name in names:
        setattr(model, name, hooks.wrap(name, getattr(model, name)))


def _one(case: str, profile: str, variant: str = "current") -> dict:
    model_name, n, d, J, P, m = CASES[case]
    X, Y = _data(n, d, model_name)
    if model_name == "multi":
        config = ADP_Config(
            seed=17,
            N_loc=30,
            N_lin=40,
            N_J=J,
            N_phi=P,
            outer_steps=2,
            h_min=0.3,
            index_init="local",
            estimator="new",
            batch_size=32,
        )
        model = ADP_multi_index(
            m, config, ADP_solver(solve_lsmr, max_steps=50, tol=1e-6)
        )
        settings = {**asdict(config), "kernel": "max(1-q**2,0), q=distance2/h**2"}
        settings["solver"] = {"method": "LSMR", **model.solver.settings}
    else:
        model = ADP_Manifold(
            m,
            N_loc=30,
            N_lin=80 if d > 4 else 60,
            N_J=J,
            N_phi=P,
            N_manifold=6,
            sync_steps=2,
            a=2.0,
            h_min=0.6,
            batch_size=32,
            seed=17,
            solver="cg",
            cg_tol=1e-8,
            scale_boundary="stop",
        )
        settings = {
            key: value for key, value in vars(model).items() if not key.startswith("_")
        }
        settings["kernel"] = "max(1-q**2,0), q=distance2/h**2"

    row: dict = {
        "case": case,
        "profile_mode": profile,
        "variant": variant,
        "settings": settings,
    }
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    sampler = RssSampler() if profile != "off" else None
    hooks = HookTimes(sampler) if sampler is not None else None
    python_peak = None
    profiler = cProfile.Profile() if profile == "cprofile" else None
    with ExitStack() as stack:
        if variant == "reference":
            from benchmarks.optimization_candidates import (
                hpao_operator_reference,
                penalty_reference,
                recover_reference,
            )

            if model_name == "multi":
                solver_module = importlib.import_module("ADP.solver.LSMR")
                stack.enter_context(
                    patch.object(
                        solver_module, "_linear_operator", hpao_operator_reference
                    )
                )
            else:
                stack.enter_context(
                    patch.object(model, "_penalty_action", penalty_reference)
                )
                stack.enter_context(
                    patch.object(model, "_recover_projector", recover_reference)
                )
        if model_name == "multi":
            module = importlib.import_module("ADP.core.multi.ADP_multi_index")
            if profile == "off":
                stack.enter_context(patch.object(module, "IndexProfiler", NoProfiler))
            elif profile in {"time", "cprofile"}:
                TimeProfiler.sampler = sampler
                stack.enter_context(patch.object(module, "IndexProfiler", TimeProfiler))
            if hooks is not None:
                _multi_hooks(stack, hooks)
        elif hooks is not None:
            _manifold_hooks(model, hooks)
        if sampler is not None:
            stack.enter_context(sampler)
        if profile == "traced" and model_name == "manifold":
            tracemalloc.start()
        if profiler is not None:
            profiler.enable()
        started = perf_counter()
        try:
            model.fit(X, Y)
            row["error"] = None
        except Exception as error:  # Preserve failed solver runs in raw evidence.
            row["error"] = f"{type(error).__name__}: {error}"
        row["fit_seconds"] = perf_counter() - started
        if profiler is not None:
            profiler.disable()
        if profile == "traced" and model_name == "manifold":
            python_peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()

    row["rss_pre_fit_bytes"] = rss_before
    row["rss_peak_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    row["rss_sampled_phase_peak_bytes"] = dict(sampler.peaks) if sampler else None
    row["rss_samples"] = sampler.samples if sampler else 0
    row["python_traced_peak_bytes"] = python_peak
    if hooks is not None:
        row["hook_seconds_inclusive"] = dict(hooks.times)
        row["hook_seconds_top_level"] = dict(hooks.top)
        row["hook_calls"] = dict(hooks.counts)
    if row["error"] is None:
        if model_name == "multi":
            row["phases"] = model.profile_
            row["basis"] = model.basis_.tolist()
            row["stop_reason"] = model.result_.stop_reason
            row["solver"] = [item["solver"] for item in model.trace_]
            row["trace"] = [
                {k: v for k, v in item.items() if k not in {"basis", "beta"}}
                for item in model.trace_
            ]
        else:
            row["projectors"] = model.projectors_.tolist()
            row["stop_reason"] = model.stop_reason_
            row["trace"] = model.trace_
            row["effective_config"] = model.effective_config_
    if profiler is not None:
        stats = pstats.Stats(profiler)
        row["cprofile_top_cumulative"] = _top_stats(stats, "cumulative")
        row["cprofile_top_self"] = _top_stats(stats, "time")
    return row


def _top_stats(stats: pstats.Stats, sort: str) -> list[dict]:
    entries = sorted(
        stats.stats.items(),
        key=lambda entry: entry[1][3 if sort == "cumulative" else 2],
        reverse=True,
    )[:30]
    return [
        {
            "file": key[0],
            "line": key[1],
            "function": key[2],
            "calls": value[1],
            "self_seconds": value[2],
            "cumulative_seconds": value[3],
        }
        for key, value in entries
    ]


def _source_state() -> dict:
    root = Path(__file__).resolve().parents[1]
    sources = [*sorted((root / "ADP").rglob("*.py")), Path(__file__).resolve()]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument(
        "--profile", choices=("off", "time", "traced", "cprofile"), default="time"
    )
    parser.add_argument(
        "--variant", choices=("current", "reference"), default="current"
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        print(json.dumps(_one(args.case, args.profile, args.variant)))
        return
    if args.runs < 1 or args.output is None or args.output.exists():
        parser.error("--runs must be positive and --output must be a new path")
    root = Path(__file__).resolve().parents[1]
    source_before = _source_state()
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        text=True,
        cwd=root,
    )
    command = [
        sys.executable,
        "-m",
        "benchmarks.fit_bottlenecks",
        "--child",
        "--case",
        args.case,
        "--profile",
        args.profile,
        "--variant",
        args.variant,
    ]
    rows = []
    for repeat in range(args.runs + 1):
        run = subprocess.run(
            command, capture_output=True, text=True, cwd=root, check=False
        )
        if run.returncode:
            rows.append({"error": run.stderr[-4000:], "returncode": run.returncode})
        else:
            rows.append(json.loads(run.stdout))
        rows[-1]["repeat"] = repeat
        rows[-1]["warmup"] = repeat == 0
        print(
            f"{args.case} {args.profile} {repeat}/{args.runs}: "
            f"{rows[-1].get('fit_seconds', 'error')}",
            flush=True,
        )
    report = {
        "command": command,
        "case_shape": CASES[args.case],
        "profile_mode": args.profile,
        "variant": args.variant,
        "warmup_count": 1,
        "timed_repeats": args.runs,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=root
        ).strip(),
        "git_status_before": git_status,
        "git_status_after": subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            text=True,
            cwd=root,
        ),
        "source_sha256_before": source_before,
        "source_sha256_after": _source_state(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "blas": np.__config__.show(mode="dicts"),
        "threads": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
