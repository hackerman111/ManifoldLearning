"""Парный full-fit benchmark CPU / GPU statistics / GPU statistics+LSMR.

Запуск: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --extra gpu
  --group bench python -m benchmarks.index_gpu --output /tmp/index_gpu.json
Числа включают upload/download и synchronize, но исключают CUDA/JIT warmup.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import resource
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from ADP import ADP_Config, ADP_multi_index, ADP_single_index, ADP_solver
from ADP.cli.main import _quality, _synthetic_data
from ADP.gpu import require_cupy
from ADP.solver.LSMR import solve

SHAPES = {
    "small": (1000, 32, 100, 12),
    "medium": (2000, 100, 200, 20),
    "large": (4000, 200, 200, 20),
    "stress": (10000, 1000, 1000, 20),
}


def _run(cp, data, shape, m, backend, seed, index_init, outer_steps=2, solver_steps=3):
    n, d, J, P = shape
    config = ADP_Config(
        seed=seed,
        N_J=J,
        N_phi=P,
        N_loc=20,
        N_lin=2 * d,
        outer_steps=outer_steps,
        index_init=index_init,
        select_step="last",
        gpu=backend != "cpu",
        gpu_solver=backend == "gpu_solver",
    )
    solver = ADP_solver(solve, max_steps=solver_steps, tol=1e-6)
    model = (
        ADP_single_index(config, solver)
        if m == 1
        else ADP_multi_index(m, config, solver)
    )
    pool = cp.cuda.MemoryPool()
    peak = 0

    def allocate(size):
        nonlocal peak
        pointer = pool.malloc(size)
        peak = max(peak, pool.used_bytes())
        return pointer

    result = {"backend": backend, "shape": [n, d, J, P, m], "seed": seed, "error": None}
    with cp.cuda.using_allocator(allocate):
        cp.cuda.get_current_stream().synchronize()
        start = time.perf_counter()
        try:
            model.fit(data[0], data[1])
            cp.cuda.get_current_stream().synchronize()
            result["seconds"] = time.perf_counter() - start
            index = model.beta_ if m == 1 else model.basis_.T
            result.update(
                quality=_quality("single" if m == 1 else "multi", index, data[2]),
                index=index.tolist(),
                stages=model.profile_["stages"],
                host_traced_peak_bytes=model.profile_["peak_memory_bytes"],
                iterations=sum(
                    row["solver"]["lsmr_iterations_total"] for row in model.trace_
                ),
                solves=sum(row["solver"]["lsmr_solves_total"] for row in model.trace_),
                loss=[row["solver"]["loss"] for row in model.trace_],
                residual_ratio=[
                    row["solver"]["normal_residual_ratio"] for row in model.trace_
                ],
                h=[row["h"] for row in model.trace_],
                factor=[row["factor"] for row in model.trace_],
                stop_reason=model.result_.stop_reason,
                gpu_pool_peak_used_bytes=peak,
                gpu_pool_reserved_bytes=pool.total_bytes(),
            )
        except Exception as error:
            # Ошибки сохраняются: benchmark не должен отбрасывать неудачные fit.
            result.update(
                error=f"{type(error).__name__}: {error}",
                seconds=time.perf_counter() - start,
            )
    pool.free_all_blocks()
    return result


def _seeds(seed: int) -> tuple[int, int]:
    data_stream, method_stream = np.random.SeedSequence(seed).spawn(2)
    return int(data_stream.generate_state(1)[0]), int(
        method_stream.generate_state(1)[0]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shapes", nargs="+", choices=SHAPES, default=["small", "medium", "large"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--index-init", choices=["local", "random"], default="local")
    parser.add_argument("--outer-steps", type=int, default=2)
    parser.add_argument("--solver-max-steps", type=int, default=3)
    args = parser.parse_args()
    if args.outer_steps < 1 or args.solver_max_steps < 1:
        parser.error("step counts must be positive")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    cp = require_cupy()
    config_info = io.StringIO()
    with contextlib.redirect_stdout(config_info):
        np.show_config()
    root = Path(__file__).resolve().parents[1]
    source_hash = hashlib.sha256()
    for path in sorted((root / "ADP").rglob("*.py")):
        source_hash.update(str(path.relative_to(root)).encode())
        source_hash.update(path.read_bytes())
    report = {
        "protocol": {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "dirty": subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ),
            "adp_source_sha256": source_hash.hexdigest(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "cupy": cp.__version__,
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver": cp.cuda.runtime.driverGetVersion(),
            "platform": platform.platform(),
            "blas_config": config_info.getvalue(),
            "dtype": "float64",
            "warmup": "one fit per shape/backend/mode; excluded, recorded separately",
            "timing": (
                "perf_counter + CUDA stream synchronize; includes transfers and "
                "profiler"
            ),
            "memory": (
                "host tracemalloc stage high-water; GPU allocator pool peak "
                "used/reserved; excludes CUDA library workspaces"
            ),
            "seed_scheme": (
                "SeedSequence(seed).spawn(2): data/method; each splits again; "
                "paired CPU/GPU inputs and NumPy directions"
            ),
            "kernel": (
                "max(1-t**2,0), t=localization squared distance/h**2 (see weights.py)"
            ),
            "bandwidth": (
                "current search_bandwidth target N_loc; h/a per outer step; "
                "rho/alpha from current CPU mass search"
            ),
            "directions": (
                "auto: single localized, multi isotropic; redraw every outer step"
            ),
            "centers": "uniform without replacement, no displacement, training_set=all",
            "solver": (
                f"HPAO LSMR, max_steps={args.solver_max_steps}, tol=1e-6, "
                "theta=0.1, krylov_tol=1e-10, "
                "default trust radius, maxiter=max(50,5*index.size)"
            ),
            "config_defaults": {
                k: v for k, v in asdict(ADP_Config()).items() if k != "kernel"
            },
            "config_overrides": {
                "N_loc": 20,
                "N_lin": "2*d",
                "N_J": "J",
                "N_phi": "P",
                "outer_steps": args.outer_steps,
                "select_step": "last",
                "seed": "model_seed",
                "gpu": "backend != cpu",
                "gpu_solver": "backend == gpu_solver",
            },
            "index_init": args.index_init,
            "repeats": args.repeats,
            "threads": {
                k: os.environ.get(k)
                for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
            },
        },
        "rows": [],
        "warmups": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        report["protocol"]["threadpools"] = threadpool_info()
        for m in (1, 3):
            for name in args.shapes:
                shape = SHAPES[name]
                warm_data_seed, warm_model_seed = _seeds(42)
                warm_data = _synthetic_data(shape[0], shape[1], m, 0.05, warm_data_seed)
                for backend in ("cpu", "gpu_statistics", "gpu_solver"):
                    report["active"] = {
                        "phase": "warmup",
                        "case": name,
                        "m": m,
                        "backend": backend,
                    }
                    args.output.write_text(
                        json.dumps(report, indent=2, ensure_ascii=False)
                    )
                    print(f"warmup {name} m={m} {backend}", flush=True)
                    warm = _run(
                        cp,
                        warm_data,
                        shape,
                        m,
                        backend,
                        warm_model_seed,
                        args.index_init,
                        args.outer_steps,
                        args.solver_max_steps,
                    )
                    warm.pop("index", None)
                    warm["case"] = name
                    report["warmups"].append(warm)
                    report["active"] = None
                    args.output.write_text(
                        json.dumps(report, indent=2, ensure_ascii=False)
                    )
                    if warm["error"]:
                        print(f"warmup failed: {warm['error']}", flush=True)
                for seed in args.seeds:
                    data_seed, model_seed = _seeds(seed)
                    data = _synthetic_data(shape[0], shape[1], m, 0.05, data_seed)
                    reference = None
                    for repeat in range(args.repeats):
                        for backend in ("cpu", "gpu_statistics", "gpu_solver"):
                            report["active"] = {
                                "phase": "measurement",
                                "case": name,
                                "m": m,
                                "backend": backend,
                                "seed": seed,
                                "repeat": repeat,
                            }
                            args.output.write_text(
                                json.dumps(report, indent=2, ensure_ascii=False)
                            )
                            row = _run(
                                cp,
                                data,
                                shape,
                                m,
                                backend,
                                model_seed,
                                args.index_init,
                                args.outer_steps,
                                args.solver_max_steps,
                            )
                            row.update(
                                case=name,
                                repeat=repeat,
                                seed=seed,
                                data_seed=data_seed,
                                model_seed=model_seed,
                            )
                            if row["error"] is None:
                                index = np.atleast_2d(row.pop("index"))
                                if backend == "cpu":
                                    reference = index
                                if reference is not None:
                                    # Сравнение подпространств без d*d projectors.
                                    row["subspace_error_vs_cpu"] = float(
                                        np.linalg.norm(
                                            index - (index @ reference.T) @ reference
                                        )
                                        / np.sqrt(m)
                                    )
                            report["rows"].append(row)
                            report["active"] = None
                            report["process_rss_high_water_kib"] = resource.getrusage(
                                resource.RUSAGE_SELF
                            ).ru_maxrss
                            args.output.write_text(
                                json.dumps(report, indent=2, ensure_ascii=False)
                            )
                            print(
                                f"{name} m={m} seed={seed} {backend}: "
                                f"{row['seconds']:.3f}s error={row['error']}",
                                flush=True,
                            )


if __name__ == "__main__":
    main()
