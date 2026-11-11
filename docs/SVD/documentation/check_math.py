"""Малые независимые проверки формул SVD_solver.tex; запуск без обучения ADP."""

import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(20260929)
metrics = {}


def close(name, actual, expected, tolerance=1e-9):
    error = float(np.linalg.norm(np.asarray(actual) - expected))
    metrics[name] = error
    assert error <= tolerance * max(1.0, float(np.linalg.norm(expected))), (name, error)


def matrix(operator, shape):
    return np.column_stack(
        [operator(row.reshape(shape)).ravel() for row in np.eye(np.prod(shape))]
    )


J, p, m, d, k = 5, 4, 3, 7, 2
P = np.linalg.qr(rng.normal(size=(d, m)))[0].T
U = rng.normal(size=(J, p, d))
U[0, :, -1] = U[0, :, 0]  # локальное вырождение
U[1] = 0
g = rng.normal(size=(J, m))
I = rng.normal(size=(J, p))
N = np.array([0.0, 0.01, 1.0, 5.0, 20.0])
lam = 0.7
root = np.sqrt(N)


def A(X):
    return root[:, None] * np.einsum("jpd,md,jm->jp", U, X, g)


def adj(z):
    return np.einsum("j,jm,jpd,jp->md", root, g, U, z)


def L(X):
    return adj(A(X)) + lam * X


def F(X):
    return np.sum((A(X) - root[:, None] * I) ** 2) + lam * np.sum((X - P) ** 2)


C = adj(root[:, None] * I) + lam * P
X = rng.normal(size=(m, d))
z = rng.normal(size=(J, p))
close("adjoint", np.sum(A(X) * z), np.sum(X * adj(z)))
H = U.transpose(0, 2, 1) @ U
close("normal_operator", L(X), sum(N[j] * np.outer(g[j], g[j]) @ X @ H[j] for j in range(J)) + lam * X)
K = matrix(L, (m, d))
K_column = sum(N[j] * np.kron(H[j], np.outer(g[j], g[j])) for j in range(J)) + lam * np.eye(m * d)
close("column_vectorization", K_column @ X.ravel(order="F"), L(X).ravel(order="F"))
Bstar = np.linalg.solve(K, C.ravel()).reshape(m, d)
Aug = np.vstack((matrix(A, (m, d)), np.sqrt(lam) * np.eye(m * d)))
rhs = np.concatenate(((root[:, None] * I).ravel(), np.sqrt(lam) * P.ravel()))
close("full_augmented_solution", np.linalg.lstsq(Aug, rhs, rcond=None)[0].reshape(m, d), Bstar)
Q = C - L(X)
delta = X - Bstar
gap = F(X) - F(Bstar)
close("energy_identity", gap, np.sum(delta * L(delta)))
assert np.linalg.norm(delta) <= np.linalg.norm(Q) / lam + 1e-10
assert gap <= np.sum(Q**2) / lam + 1e-10
shift_rhs = np.concatenate(((root[:, None] * I - A(X)).ravel(), np.sqrt(lam) * (P - X).ravel()))
close("shifted_regularization", X + np.linalg.lstsq(Aug, shift_rhs, rcond=None)[0].reshape(m, d), Bstar)

a = rng.normal(size=m)
a /= np.linalg.norm(a)
v = rng.normal(size=d)
v /= np.linalg.norm(v)
Z = np.outer(a, v)
R = np.sum(Q * Z)
D = np.sum(Z * L(Z))
close("rank_one_gain", F(X + R / D * Z) - F(X), -R**2 / D)
weights = N * np.sum((U @ v) ** 2, axis=1)
Gv = g.T @ (weights[:, None] * g) + lam * np.eye(m)
alpha = np.linalg.solve(Gv, Q @ v)
alpha /= np.linalg.norm(alpha)
newZ = np.outer(alpha, v)
assert np.sum(Q * newZ)**2 / np.sum(newZ * L(newZ)) >= R**2 / D - 1e-10
Av = (root * (g @ a))[:, None, None] * U
Ha = np.einsum("jpd,jpe->de", Av, Av) + lam * np.eye(d)
vraw = np.linalg.solve(Ha, Q.T @ a)
e = I - np.einsum("jpd,md,jm->jp", U, X, g)
av_aug = np.vstack((Av.reshape(J * p, d), np.sqrt(lam) * np.eye(d)))
av_rhs = np.concatenate(((root[:, None] * e).ravel(), np.sqrt(lam) * (P - X).T @ a))
close("conditional_v_solution", vraw, np.linalg.lstsq(av_aug, av_rhs, rcond=None)[0])

V = np.linalg.qr(rng.normal(size=(d, k)))[0]
left = np.linalg.qr(rng.normal(size=(m, k)))[0]
atoms = np.stack([np.outer(left[:, s], V[:, s]) for s in range(k)])
M = np.array([[np.sum(atoms[s] * L(atoms[t])) for t in range(k)] for s in range(k)])
h = np.array([np.sum(C * atom) for atom in atoms])
scales = np.linalg.solve(M, h)
Bs = np.einsum("k,kmd->md", scales, atoms)
close("scale_stationarity", np.array([np.sum((C - L(Bs)) * atom) for atom in atoms]), np.zeros(k))

def projected(Y):
    return L(Y @ V.T) @ V


small = matrix(projected, (m, k))
Y = np.linalg.solve(small, (C @ V).ravel()).reshape(m, k)
close("projected_residual", (C - L(Y @ V.T)) @ V, np.zeros((m, k)))
Ya = np.linalg.solve(small, ((C - L(P)) @ V).ravel()).reshape(m, k)
Ba = P + Ya @ V.T
assert F(Ba) <= F(P) + 1e-10
close("affine_projected_residual", (C - L(Ba)) @ V, np.zeros((m, k)))

Lb = rng.normal(size=(m, k))
right = rng.normal(size=(d, k))
alphas = g @ Lb
Zprior = P - X
Tmat = np.vstack((matrix(lambda Y: root[:, None] * np.einsum("jpd,dk,jk->jp", U, Y, alphas), (d, k)),
                  np.sqrt(lam) * matrix(lambda Y: Y @ Lb.T, (d, k))))
block_normal = sum(N[j] * H[j] @ right @ np.outer(alphas[j], alphas[j]) for j in range(J)) + lam * right @ (Lb.T @ Lb)
close("coupled_block_normal", (Tmat.T @ Tmat @ right.ravel()).reshape(d, k), block_normal)
block_rhs = np.concatenate(((root[:, None] * e).ravel(), np.sqrt(lam) * Zprior.T.ravel()))
expected_rhs = sum(N[j] * np.outer(U[j].T @ e[j], alphas[j]) for j in range(J)) + lam * Zprior.T @ Lb
close("coupled_block_rhs", (Tmat.T @ block_rhs).reshape(d, k), expected_rhs)
Yj = U @ V
left_normal = sum(N[j] * np.outer(g[j], g[j]) @ Lb @ (Yj[j].T @ Yj[j]) for j in range(J)) + lam * Lb
close("block_left_operator", projected(Lb), left_normal)
qr, rr = np.linalg.qr(right, mode="reduced")
close("block_qr_preserves_product", (Lb @ rr.T) @ qr.T, Lb @ right.T)

Lfac, Rfac = Lb, right
residual = I - np.einsum("jpd,dk,mk,jm->jp", U, Rfac, Lfac, g)
gradL = -2 * np.einsum("j,jm,jp,jpk->mk", N, g, residual, U @ Rfac) + 2 * lam * (Lfac @ (Rfac.T @ Rfac) - P @ Rfac)
gradR = -2 * np.einsum("j,jpd,jp,jk->dk", N, U, residual, g @ Lfac) + 2 * lam * (Rfac @ (Lfac.T @ Lfac) - P.T @ Lfac)
for name, factor, gradient in (("left", Lfac, gradL), ("right", Rfac, gradR)):
    direction = rng.normal(size=factor.shape)
    step = 1e-6
    fun = (lambda Y: F(Y @ Rfac.T)) if name == "left" else (lambda Y: F(Lfac @ Y.T))
    close("factor_gradient_" + name, (fun(factor + step * direction) - fun(factor - step * direction)) / (2 * step), np.sum(gradient * direction), 1e-6)

eta = rng.uniform(0.2, 1.0, size=J)
G = g.T @ ((N * eta)[:, None] * g) + lam * np.eye(m)
Ciso = rng.normal(size=(m, d))
chol = np.linalg.cholesky(G)
W = np.linalg.solve(chol, Ciso)
u, s, vt = np.linalg.svd(W, full_matrices=False)
Br = np.linalg.solve(chol.T, (u[:, :k] * s[:k]) @ vt[:k])
Bfull = np.linalg.solve(G, Ciso)
fi = lambda B: np.sum(B * (G @ B)) - 2 * np.sum(Ciso * B)
close("isotropic_spectral_tail", fi(Br) - fi(Bfull), np.sum(s[k:]**2))
close("isotropic_next_gain", np.linalg.norm(np.linalg.solve(chol, Ciso - G @ Br), 2)**2, s[k]**2)
close("rank_penalty_floor", np.sum((P - np.linalg.svd(P, full_matrices=False)[0][:, :k] @ np.linalg.svd(P, full_matrices=False)[2][:k])**2), m - k)

# Отдельная почти изотропная задача проверяет уточнённую границу возмущения.
raw_perturbation = rng.normal(size=(J, d, d))
perturbation_scale = lam / (20 * np.sum(N * np.sum(g**2, axis=1)))
raw_perturbation /= max(np.linalg.norm(block, 2) for block in raw_perturbation)
Unear = np.eye(d)[None, :, :] + perturbation_scale * raw_perturbation
Hnear = Unear.transpose(0, 2, 1) @ Unear
etanear = np.trace(Hnear, axis1=1, axis2=2) / d
Gnear = g.T @ ((N * etanear)[:, None] * g) + lam * np.eye(m)
Knear = sum(N[j] * np.kron(np.outer(g[j], g[j]), Hnear[j]) for j in range(J)) + lam * np.eye(m * d)
rho = sum(N[j] * np.sum(g[j]**2) * np.linalg.norm(Hnear[j] - etanear[j] * np.eye(d), 2) for j in range(J))
theta = rho / np.linalg.eigvalsh(Gnear)[0]
assert theta < 1
geig, gu = np.linalg.eigh(Gnear)
Ginvhalf = (gu / np.sqrt(geig)) @ gu.T
whitening = np.kron(Ginvhalf, np.eye(d))
spectrum = np.linalg.eigvalsh(whitening @ Knear @ whitening)
assert spectrum[0] >= 1 - theta - 1e-10
assert spectrum[-1] <= 1 + theta + 1e-10
metrics["relative_perturbation_theta"] = float(theta)
qnear = rng.normal(size=(m, d))
uq, sq, vq = np.linalg.svd(Ginvhalf @ qnear, full_matrices=False)
z0 = np.outer(Ginvhalf @ uq[:, 0], vq[0])
gain = np.sum(qnear * z0)**2 / (z0.ravel() @ Knear @ z0.ravel())
assert gain >= sq[0]**2 / (1 + theta) - 1e-10
metrics["spectral_candidate_gain"] = float(gain)

Bhat = Bstar + 1e-3 * rng.normal(size=Bstar.shape)
pi = lambda B: np.linalg.qr(B.T)[0] @ np.linalg.qr(B.T)[0].T
sigma = np.linalg.svd(Bstar, compute_uv=False)[-1]
error = np.linalg.norm(Bhat - Bstar)
assert np.linalg.norm(pi(Bhat) - pi(Bstar)) <= np.sqrt(2) * error / sigma + 1e-10
Dfull = Bstar - X
# Приближённый шаг специально имеет ненулевую нормальную невязку.
eta_descent = 0.5
step_loss = min(0.01, eta_descent * lam * np.linalg.norm(Dfull) / (4 * np.linalg.norm(Q)))
Dapprox = (1 - step_loss) * Dfull
normal_residual = Q - L(Dapprox)
close("inexact_descent_identity", F(X + Dapprox) - F(X), -np.sum(Dapprox * L(Dapprox)) - 2 * np.sum(normal_residual * Dapprox))
assert np.linalg.norm(normal_residual) <= eta_descent * lam * np.linalg.norm(Dapprox) / 2
assert F(X + Dapprox) - F(X) <= -(1 - eta_descent) * lam * np.sum(Dapprox**2)
coupling = np.linalg.solve(np.array([[2., 1.], [1., 2.]]), np.array([0., 1.]))
close("galerkin_coupling_counterexample", coupling, np.array([-1/3, 2/3]))

output = Path(__file__).with_name("math_checks.json")
output.write_text(json.dumps({"seed": 20260929, "status": "passed", "checks": metrics}, ensure_ascii=False, indent=2) + "\n")
print(f"Проверки формул и неравенств пройдены; записано {len(metrics)} показателей: {output}")
