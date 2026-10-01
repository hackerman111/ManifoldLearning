"""Independent, small dense references for EDR_unified_theory.tex.

No production imports, solver benchmarks or recovery claims. Run from root:
    python SVD/theory_2026_10_01/check_formulas.py
The JSON records formula errors, finite-difference tolerances, seeds and env.
"""
from __future__ import annotations

import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import scipy
from scipy.linalg import expm


RESULTS: list[dict] = []


def check(name, actual, expected, tol=2e-11, **context):
    actual, expected = np.asarray(actual), np.asarray(expected)
    error = float(np.linalg.norm(actual - expected) / max(1.0, np.linalg.norm(expected)))
    passed = bool(np.isfinite(error) and error <= tol)
    RESULTS.append(dict(name=name, error=error, tolerance=tol, passed=passed, **context))
    if not passed:
        raise AssertionError(f"{name}: {error:.3e} > {tol:.3e}; {context}")


def orth(x):
    return np.linalg.qr(x, mode="reduced")[0]


def spectral(x, power):
    values, vectors = np.linalg.eigh(x)
    return (vectors * values**power) @ vectors.T


def grassmann_exp(y, z):
    v, s, at = np.linalg.svd(z, full_matrices=False)
    a = at.T
    return y + (y @ a * (np.cos(s) - 1) + v * np.sin(s)) @ at


def refit(u, target, y, tau):
    m = y.shape[1]
    design = np.vstack((u @ y, np.sqrt(tau) * np.eye(m)))
    rhs = np.r_[target, np.zeros(m)]
    g = np.linalg.lstsq(design, rhs, rcond=None)[0]
    return g, rhs - design @ g


def profile(us, targets, masses, y, y0, tau, lam):
    loss = 0.0
    euclidean = np.zeros_like(y)
    residuals = []
    for u, target, mass in zip(us, targets, masses):
        g, residual = refit(u, target, y, tau)
        loss += mass * (residual @ residual)
        euclidean -= 2 * mass * np.outer(u.T @ residual[:len(target)], g)
        residuals.append(np.sqrt(mass) * residual)
    # Stable chordal form, independent of m - ||y0.T y||^2 cancellation.
    prox = y - y0 @ (y0.T @ y)
    loss += lam * np.sum(prox**2)
    euclidean -= 2 * lam * y0 @ (y0.T @ y)
    grad = euclidean - y @ (y.T @ euclidean)
    residuals.append(np.sqrt(lam) * prox.ravel())
    return loss, grad, np.concatenate(residuals)


def trial(seed, d=8, m=3, p=6, j_count=4, tau=0.23):
    rng = np.random.default_rng(seed)
    y, y0 = orth(rng.normal(size=(d, m))), orth(rng.normal(size=(d, m)))
    z = rng.normal(size=(d, m))
    z = (z - y @ (y.T @ z)) * 0.12
    us = rng.normal(size=(j_count, p, d))
    targets = rng.normal(size=(j_count, p))
    masses = np.exp(rng.normal(size=j_count))
    lam = 0.37
    ctx = dict(seed=seed, d=d, m=m, p=p, tau=tau)

    # Exponential reference uses ambient orthogonal generator, not SVD.
    gen = z @ y.T - y @ z.T
    candidate = grassmann_exp(y, z)
    check("exp_against_ambient_expm", candidate, expm(gen) @ y, **ctx)
    check("exp_orthonormality", candidate.T @ candidate, np.eye(m), **ctx)
    v, s, at = np.linalg.svd(z, full_matrices=False)
    a = at.T
    seq = y.copy()
    for ell in range(m):
        ya = y @ a[:, ell]
        gl = s[ell] * (np.outer(v[:, ell], ya) - np.outer(ya, v[:, ell]))
        seq = expm(gl) @ seq
    check("commuting_plane_rotations", seq, candidate, **ctx)
    polar = (y + z) @ spectral(np.eye(m) + z.T @ z, -0.5)
    angle_z = (v * np.arctan(s)) @ at
    check("polar_angle_map", polar, grassmann_exp(y, angle_z), **ctx)
    q = orth(y + z)
    check("qr_polar_subspace", q @ q.T, polar @ polar.T, **ctx)
    small_z = 1e-3 * z
    remainder = grassmann_exp(y, small_z) - y - small_z + 0.5 * y @ small_z.T @ small_z
    check("exp_second_order", remainder, np.zeros_like(y), tol=1e-10, **ctx)

    # Profile and full reduced Jacobian, including coefficient derivatives.
    loss, gradient, residual = profile(us, targets, masses, y, y0, tau, lam)
    rotate = orth(rng.normal(size=(m, m)))
    check("orthogonal_profile_invariance", profile(us, targets, masses, y @ rotate, y0, tau, lam)[0], loss, **ctx)
    h = 1e-5
    plus = profile(us, targets, masses, grassmann_exp(y, h*z), y0, tau, lam)
    minus = profile(us, targets, masses, grassmann_exp(y, -h*z), y0, tau, lam)
    check("profile_gradient_central_difference", (plus[0]-minus[0])/(2*h), np.sum(gradient*z), tol=3e-8, **ctx)
    jac = []
    for u, target, mass in zip(us, targets, masses):
        g, r = refit(u, target, y, tau)
        design, dz = u @ y, u @ z
        dg = np.linalg.solve(design.T @ design + tau*np.eye(m), dz.T @ r[:p] - design.T @ dz @ g)
        jac.append(np.sqrt(mass) * np.r_[-dz @ g-design @ dg, -np.sqrt(tau)*dg])
    jac.append((np.sqrt(lam) * (z-y0 @ (y0.T @ z))).ravel())
    jac = np.concatenate(jac)
    check("full_profile_jacobian_central_difference", (plus[2]-minus[2])/(2*h), jac, tol=3e-8, **ctx)
    check("jacobian_envelope_gradient", 2*residual @ jac, np.sum(gradient*z), **ctx)

    # Schur complement scalar profile vs direct augmented least squares.
    a1 = rng.normal(size=m)
    a1 /= np.linalg.norm(a1)
    rot = orth(np.column_stack((a1, rng.normal(size=(m, m-1)))))
    a1, aperp = rot[:, 0], rot[:, 1:]
    v1 = rng.normal(size=d)
    v1 -= y @ (y.T @ v1)
    v1 /= np.linalg.norm(v1)
    for u, target in zip(us, targets):
        constant = u @ y @ aperp
        pi = np.eye(p)-constant @ np.linalg.solve(constant.T @ constant+tau*np.eye(m-1), constant.T)
        left, right = u @ y @ a1, u @ v1
        k = target @ pi @ target
        b, c = left @ pi @ target, right @ pi @ target
        aa, dd, ee = left @ pi @ left, left @ pi @ right, right @ pi @ right
        for angle in [-1.51, -0.2, 0.0, 0.7, 1.57]:
            ct, st = np.cos(angle), np.sin(angle)
            yt = y + np.outer(y @ a1*(ct-1)+v1*st, a1)
            scalar = k-(b*ct+c*st)**2/(aa*ct**2+2*dd*ct*st+ee*st**2+tau)
            direct_r = refit(u, target, yt, tau)[1]
            check("scalar_schur_profile", scalar, direct_r @ direct_r, angle=angle, **ctx)

    # Fixed-g operator identity, energy, exact rank-one line search.
    fixed_g = rng.normal(size=(j_count, m))
    prior, bmat = y0.T, rng.normal(size=(m, d))
    def forward(b):
        return np.array([np.sqrt(mass)*u @ b.T @ g for mass, u, g in zip(masses, us, fixed_g)])
    def adjoint(e):
        return sum(np.sqrt(mass)*np.outer(g, u.T @ r) for mass, u, g, r in zip(masses, us, fixed_g, e))
    def lop(b):
        return adjoint(forward(b))+lam*b
    def objective(b):
        r = np.sqrt(masses[:,None])*targets-forward(b)
        return np.sum(r*r)+lam*np.sum((b-prior)**2)
    cmat = adjoint(np.sqrt(masses[:,None])*targets)+lam*prior
    e = rng.normal(size=(j_count,p))
    check("forward_adjoint", np.sum(forward(bmat)*e), np.sum(bmat*adjoint(e)), **ctx)
    rhs = cmat-lop(bmat)
    atom = np.outer(a1, v1)
    curvature, numerator = np.sum(atom*lop(atom)), np.sum(rhs*atom)
    check("exact_rank_one_gain", objective(bmat)-objective(bmat+atom*numerator/curvature), numerator**2/curvature, **ctx)

    # Arbitrary fixed atom/core reference is built by explicit basis probes.
    ar = orth(rng.normal(size=(m, min(2,m))))
    vr = orth(rng.normal(size=(d, min(2,m))))
    rdim = ar.shape[1]
    atoms = [np.outer(ar[:,i],vr[:,k]) for i in range(rdim) for k in range(rdim)]
    design = np.column_stack([np.r_[forward(e).ravel(),np.sqrt(lam)*e.ravel()] for e in atoms])
    data = np.r_[(np.sqrt(masses[:,None])*targets-forward(prior)).ravel(), np.zeros(m*d)]
    core = np.linalg.lstsq(design,data,rcond=None)[0]
    normal = np.array([[np.sum(x*lop(y)) for y in atoms] for x in atoms])
    core_rhs = np.array([np.sum(x*(cmat-lop(prior))) for x in atoms])
    check("full_core_cross_terms", core, np.linalg.solve(normal,core_rhs), **ctx)
    diagonal_idx = [i*rdim+i for i in range(rdim)]
    diag = np.linalg.lstsq(design[:,diagonal_idx],data,rcond=None)[0]
    core_loss = float(np.linalg.norm(data-design @ core)**2)
    diag_loss = float(np.linalg.norm(data-design[:,diagonal_idx] @ diag)**2)
    RESULTS.append(dict(name="full_core_not_worse_than_diagonal", passed=core_loss<=diag_loss+1e-10, full_core_loss=core_loss, diagonal_loss=diag_loss, **ctx))

    # Exact QR compression of the joint [U,I] column space.
    for u,target in zip(us,targets):
        qr_q,qr_r = np.linalg.qr(np.column_stack((u,target)),mode="reduced")
        pred = rng.normal(size=d)
        check("untruncated_joint_qr", np.linalg.norm(target-u @ pred), np.linalg.norm(qr_r[:,-1]-qr_r[:,:d] @ pred), **ctx)


def special_cases():
    rng=np.random.default_rng(601010)
    m,d,r=3,7,2
    gf,hf=rng.normal(size=(m,m)),rng.normal(size=(d,d))
    g,h=gf @ gf.T+np.eye(m),hf @ hf.T+np.eye(d)
    c=rng.normal(size=(m,d))
    gi,hi=spectral(g,-0.5),spectral(h,-0.5)
    w=gi @ c @ hi
    ul,ss,vt=np.linalg.svd(w,full_matrices=False)
    br=gi @ ((ul[:,:r]*ss[:r]) @ vt[:r]) @ hi
    full=np.linalg.solve(h,np.linalg.solve(g,c).T).T
    energy=lambda b: np.sum(b*(g @ b @ h))-2*np.sum(c*b)
    check("two_sided_svd_tail", energy(br)-energy(full), np.sum(ss[r:]**2))
    check("rank_preserving_whitening", np.linalg.matrix_rank(br), r)
    prior=orth(rng.normal(size=(d,m))).T
    cdelta=c-g @ prior @ h
    wd=gi @ cdelta @ hi
    ud,sd,vd=np.linalg.svd(wd,full_matrices=False)
    delta=gi @ ((ud[:,:r]*sd[:r]) @ vd[:r]) @ hi
    check("correction_shift_energy", energy(prior+delta)-energy(full),np.sum(sd[r:]**2))

    target=np.diag([2.,1.]); weights=np.array([[.01,1],[1,100.]])
    ordinary=np.diag([2.,0.]); better=np.diag([0.,1.])
    ordinary_loss=float(np.sum(weights*(target-ordinary)**2))
    better_loss=float(np.sum(weights*(target-better)**2))
    if not better_loss < ordinary_loss:
        raise AssertionError("weighted SVD counterexample failed")
    RESULTS.append(dict(name="ordinary_svd_weighted_counterexample",passed=True,ordinary_loss=ordinary_loss,feasible_rank_one_loss=better_loss))

    # Observation-level covariance metric vs ADP moment metric, and w^2 noise.
    x=rng.normal(size=(20,4)); yy=rng.normal(size=20); weights=rng.uniform(.1,1,20)
    mass=weights.sum(); z=x-weights @ x/mass; yc=yy-weights @ yy/mass
    sigma=(z.T*weights) @ z/mass; c=(z.T*weights) @ yc/mass
    vh=np.linalg.solve(sigma,c); candidate=rng.normal(size=4)
    const=float(np.sum(weights*(yc-z @ vh)**2))
    check("mave_centered_covariance_identity",np.sum(weights*(yc-z @ candidate)**2),const+mass*(vh-candidate) @ sigma @ (vh-candidate))
    noise_map=z.T*weights/mass
    check("moment_noise_squared_weights",noise_map @ noise_map.T,(z.T*weights**2) @ z/mass**2)
    wrong_cov=sigma/mass
    RESULTS.append(dict(name="kernel_covariance_not_sigma_over_mass",passed=bool(np.linalg.norm(noise_map @ noise_map.T-wrong_cov)>1e-3),absolute_difference=float(np.linalg.norm(noise_map @ noise_map.T-wrong_cov))))
    i=np.eye(4); y=orth(rng.normal(size=(4,2))); spectrum=np.array([.3,1.2]); alpha=.2
    tensor=alpha**2*i+(y*spectrum) @ y.T
    vec=rng.normal(size=4)
    for power in [-.5,.5,1.0]:
        fast=alpha**(2*power)*vec+y @ (( (alpha**2+spectrum)**power-alpha**(2*power))*(y.T @ vec))
        check("spectral_low_rank_tensor_action",fast,spectral(tensor,power) @ vec,power=power)
    # Rank loss discontinuity for tau=0 (exact axis, no roundoff cosine).
    u=np.array([[1.,0.]]); target=np.ones(1)
    near=np.array([[1e-5],[np.sqrt(1-1e-10)]]); at=np.array([[0.],[1.]])
    near_loss=np.sum(refit(u,target,near,0)[1]**2); at_loss=np.sum(refit(u,target,at,0)[1]**2)
    check("rank_loss_profile_near",near_loss,0)
    check("rank_loss_profile_boundary",at_loss,1)
    RESULTS.append(dict(name="rank_loss_discontinuity_tau_zero",passed=True,near_loss=float(near_loss),boundary_loss=float(at_loss)))
    # Correction rank and subspace rank are different even for exact rotation.
    y=np.eye(5)[:,:2]; v=np.eye(5)[:,2]; a=np.array([1.,0.]); angle=.7
    rotated=y+np.outer(y @ a*(np.cos(angle)-1)+v*np.sin(angle),a)
    check("rotation_correction_rank",np.linalg.matrix_rank(rotated-y),1)
    check("rotation_basis_rank",np.linalg.matrix_rank(rotated),2)

    # Penalty whitening must transform the data block as well as the ridge.
    raw=rng.normal(size=(m,d)); penalty=spectral(h,.5)
    u=rng.normal(size=(4,d)); target=rng.normal(size=4); coeff=rng.normal(size=m)
    original=np.linalg.norm(target-u @ raw.T @ coeff)**2+np.linalg.norm((raw-prior) @ penalty)**2
    transformed=raw @ penalty; transformed_prior=prior @ penalty
    transformed_u=u @ spectral(h,-.5)
    whitened=np.linalg.norm(target-transformed_u @ transformed.T @ coeff)**2+np.linalg.norm(transformed-transformed_prior)**2
    check("penalty_whitening_both_blocks",original,whitened)
    check("penalty_whitening_correction_rank",np.linalg.matrix_rank((raw-prior) @ penalty),np.linalg.matrix_rank(raw-prior))

    # A tangent Tucker field has orthonormal full-rank local bases, including
    # the zero-core case. SVD is applied to Z_j, not to final basis slices.
    y0=np.eye(7)[:,:3]; vv=np.eye(7)[:,3:5]; aa=orth(rng.normal(size=(3,2)))
    cores=rng.normal(size=(2,2,2))
    for weights in [np.array([0.,0.]),np.array([1.,.3]),np.array([-.2,.8])]:
        zj=vv @ np.einsum("l,lqk->qk",weights,cores) @ aa.T
        yj=grassmann_exp(y0,zj)
        check("tangent_tucker_horizontal",y0.T @ zj,np.zeros((3,3)))
        check("tangent_tucker_full_basis",yj.T @ yj,np.eye(3))

    # No ridge and an unobserved direction: zero curvature also has zero
    # linear term. There is no R/D step or universal 1/lambda certificate.
    u=np.array([[1.,0.]]); b=np.array([[.7,.2]]); atom=np.array([[0.,1.]])
    original=np.sum((np.ones(1)-u @ b.T[:,0])**2)
    moved=np.sum((np.ones(1)-u @ (b+3*atom).T[:,0])**2)
    check("lambda_zero_null_atom_flat",moved,original)
    check("lambda_zero_null_atom_curvature",np.linalg.norm(u @ atom.T[:,0])**2,0.)


def main():
    trials=[(601001,8,3,6,4,.23),(601002,6,1,4,3,.4),(601003,8,3,2,4,.15),(601004,8,3,6,4,0.),(601005,5,3,10,3,.23)]
    output=Path(__file__).resolve().parent / "formula_results.json"
    try:
        for args in trials:
            trial(*args)
        special_cases()
    finally:
        metadata=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,dtype="float64",finite_difference_step=1e-5,scope="independent small mathematical identities; no performance/recovery benchmark")
        try:
            metadata["commit"]=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
            metadata["dirty_paths"]=subprocess.check_output(["git","status","--short"],text=True).splitlines()
        except (OSError,subprocess.CalledProcessError):
            metadata["commit"]=None
        output.write_text(json.dumps(dict(metadata=metadata,checks=RESULTS,passed=all(x["passed"] for x in RESULTS)),indent=2)+"\n")
    if not all(x["passed"] for x in RESULTS):
        raise AssertionError("An inequality/diagnostic failed; inspect formula_results.json")
    print(f"Passed {len(RESULTS)} checks; results: {output}")


if __name__=="__main__":
    main()
