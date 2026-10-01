"""Paired angle-value evaluation: cached scalar profile vs batched SVD refit.

This measures only an inner frozen-statistics curve, not a full EDR solver.
Both variants receive the same precomputed U_j[Y,v] in a rotated gauge.
Run with BLAS threading set before importing NumPy:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python SVD/theory_2026_10_01/angle_microbenchmark.py
"""
import contextlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
import tracemalloc

import numpy as np


def prepare(c, target, left, right, tau):
    j,p,k=c.shape
    augmented=np.concatenate((c,np.broadcast_to(np.sqrt(tau)*np.eye(k),(j,k,k))),axis=1)
    q=np.linalg.qr(augmented,mode="reduced")[0]
    xs=np.stack((target,left,right),axis=-1)
    xs=np.concatenate((xs,np.zeros((j,k,3))),axis=1)
    residual=xs-q @ (q.swapaxes(-1,-2) @ xs)
    gram=residual.swapaxes(-1,-2) @ residual
    return gram


def scalar_values(gram, masses, angles, tau):
    # Vectorized across centers, one array allocation of O(J) per angle.
    values=[]
    for angle in angles:
        ct,st=np.cos(angle),np.sin(angle)
        num=gram[:,1,0]*ct+gram[:,2,0]*st
        den=gram[:,1,1]*ct**2+2*gram[:,1,2]*ct*st+gram[:,2,2]*st**2+tau
        values.append(masses @ (gram[:,0,0]-num**2/den))
    return np.array(values)


def direct_values(c, target, left, right, masses, angles, tau):
    j,p,k=c.shape; m=k+1
    rhs=np.concatenate((target,np.zeros((j,m))),axis=1)
    values=[]
    for angle in angles:
        moving=left*np.cos(angle)+right*np.sin(angle)
        design=np.concatenate((c,moving[:,:,None]),axis=2)
        design=np.concatenate((design,np.broadcast_to(np.sqrt(tau)*np.eye(m),(j,m,m))),axis=1)
        # Batched augmented SVD; no normal equations and no J Python solves.
        u,s,vt=np.linalg.svd(design,full_matrices=False)
        coordinates=np.einsum("jpm,jp->jm",u,rhs)/s
        coeff=np.einsum("jkm,jk->jm",vt,coordinates)
        residual=rhs-np.einsum("jpm,jm->jp",design,coeff)
        values.append(masses @ np.einsum("jp,jp->j",residual,residual))
    return np.array(values)


def timed(call):
    start=time.perf_counter(); result=call(); elapsed=time.perf_counter()-start
    return result,elapsed


def tracked_peak(call):
    tracemalloc.start(); call(); _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    return peak


def main():
    rows=[]; tau=.2; d=300; p=40; m=3; j=128
    angles=np.linspace(-1.5,1.5,61)
    for seed in [602001,602002,602003]:
        rng=np.random.default_rng(seed)
        # True dense U and frame products; common projection cost recorded.
        u=rng.normal(size=(j,p,d))
        frame=np.linalg.qr(rng.normal(size=(d,m+1)),mode="reduced")[0]
        cache,project_s=timed(lambda:u @ frame)
        c,left,right=cache[:,:,:m-1],cache[:,:,m-1],cache[:,:,m]
        target=rng.normal(size=(j,p)); masses=np.exp(rng.normal(size=j))
        gram=prepare(c,target,left,right,tau)
        fast=lambda: scalar_values(gram,masses,angles,tau)
        direct=lambda: direct_values(c,target,left,right,masses,angles,tau)
        fast(); direct()  # warm up both variants before paired measurements
        # Common projection is also warmed and measured seven times.
        project_times=[]
        for _ in range(7):
            cache,elapsed=timed(lambda:u @ frame)
            project_times.append(elapsed)
        project_s=statistics.median(project_times)
        prep_times=[]; fast_times=[]; direct_times=[]
        for repeat in range(7):
            gram,pt=timed(lambda:prepare(c,target,left,right,tau)); prep_times.append(pt)
            # Alternate order to reduce systematic warm/cache-order effects.
            if repeat % 2:
                slow,dt=timed(direct); quick,ft=timed(fast)
            else:
                quick,ft=timed(fast); slow,dt=timed(direct)
            fast_times.append(ft); direct_times.append(dt)
        error=float(np.linalg.norm(quick-slow)/np.linalg.norm(slow))
        if error>1e-11:
            raise AssertionError(f"seed {seed}: curve error {error}")
        prep_med=statistics.median(prep_times); fast_med=statistics.median(fast_times); direct_med=statistics.median(direct_times)
        rows.append(dict(seed=seed,J=j,p=p,d=d,m=m,tau=tau,angle_count=len(angles),
            relative_curve_error=error,common_projection_s=project_s,
            preparation_s=prep_med,scalar_curve_s=fast_med,batched_svd_curve_s=direct_med,
            total_time_ratio=(project_s+prep_med+fast_med)/(project_s+direct_med),
            curve_with_prep_time_ratio=(prep_med+fast_med)/direct_med,
            preparation_peak_traced_bytes=tracked_peak(lambda:prepare(c,target,left,right,tau)),
            scalar_curve_peak_traced_bytes=tracked_peak(fast),
            batched_svd_curve_peak_traced_bytes=tracked_peak(direct),
            timing_samples=dict(common_projection=project_times,preparation=prep_times,scalar=fast_times,batched_svd=direct_times)))
    config=io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    meta=dict(python=platform.python_version(),numpy=np.__version__,dtype="float64",
        threads={key:os.environ.get(key) for key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]},
        numpy_blas_config=config.getvalue(),
        memory_method="tracemalloc separate run; traced allocations only, excludes native LAPACK workspace/RSS and already-live inputs",
        repeats=7,statistic="median",scope="61 angle values for one frozen rank-one plane; no angle minimization/full fit/recovery comparison",
        fixed_design="U=synthetic independent Gaussian; frame=orthonormal; I=Gaussian; masses=lognormal; no kernel/bandwidth/direction refresh",
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        dirty_paths=subprocess.check_output(["git","status","--short"],text=True).splitlines())
    output=Path(__file__).resolve().parent / "angle_results.json"
    output.write_text(json.dumps(dict(metadata=meta,rows=rows),indent=2)+"\n")
    print(json.dumps([dict(seed=r["seed"],error=r["relative_curve_error"],curve_with_prep_time_ratio=r["curve_with_prep_time_ratio"],total_time_ratio=r["total_time_ratio"]) for r in rows],indent=2))


if __name__=="__main__":
    main()
