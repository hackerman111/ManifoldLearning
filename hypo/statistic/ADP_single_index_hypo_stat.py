import numpy as np
from scipy import sparse

from ADP import ADP_single_index


class ADP_single_index_hypo_stat(ADP_single_index):
    def Calculate_statistic(
        self,
        weights,
        directions,
        *,
        method="direct",
        X=None,
        Y=None,
        batch_size=32,
    ):
        methods = {
            "direct": self.Calculate_statistic_direct,
            "matrix": self.Calculate_statistic_matrix,
            "sparse": self.Calculate_statistic_sparse,
        }
        try:
            calculate = methods[method]
        except (KeyError, TypeError):
            raise ValueError(f"unknown statistic method: {method!r}") from None

        kwargs = {"X": X, "Y": Y}
        if method == "matrix":
            kwargs["batch_size"] = batch_size
        return calculate(weights, directions, **kwargs)

    def Calculate_statistic_direct(self, weights, directions, *, X=None, Y=None):
        X, Y, W, Phi = self._prepare_inputs(weights, directions, X, Y, dense=True)
        mass = W.sum(axis=1)
        mean = W @ X / mass[:, None]
        centered = X[None, :, :] - mean[:, None, :]
        Q = np.einsum("jpd,jnd->jpn", Phi, centered, optimize=True)
        A = W / mass[:, None]
        residual = np.einsum("jn,jpn->jp", A, Q, optimize=True)
        eta = self._normalized_residual(A, Q, residual)
        Q *= W[:, None, :]

        return {
            "I": np.einsum("jpn,n->jp", Q, Y, optimize=True),
            "U": np.einsum("jpn,jnd->jpd", Q, centered, optimize=True),
            "mass": mass,
            "mean": mean,
            "n_eff": 1.0 / np.square(A).sum(axis=1),
            "eta": eta,
        }

    def Calculate_statistic_matrix(
        self,
        weights,
        directions,
        *,
        X=None,
        Y=None,
        batch_size=32,
    ):
        if (
            not isinstance(batch_size, (int, np.integer))
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")

        X, Y, W, Phi = self._prepare_inputs(weights, directions, X, Y, dense=True)
        J, P = Phi.shape[:2]
        mass = W.sum(axis=1)
        A = W / mass[:, None]
        x_bar = X.mean(axis=0)
        y_bar = A @ Y
        Xc = X - x_bar
        Mc = A @ Xc
        mean = Mc + x_bar
        I = np.empty((J, P))
        U = np.empty((J, P, X.shape[1]))
        eta = np.empty((J, P))

        for start in range(0, J, batch_size):
            stop = min(start + batch_size, J)
            Ab = A[start:stop]
            Phib = Phi[start:stop]
            Q = np.einsum("bpd,nd->bpn", Phib, Xc, optimize=True)
            Q -= np.einsum("bpd,bd->bp", Phib, Mc[start:stop])[:, :, None]
            residual = np.einsum("bn,bpn->bp", Ab, Q, optimize=True)
            eta[start:stop] = self._normalized_residual(Ab, Q, residual)
            Q -= residual[:, :, None]
            H = Ab[:, None, :] * Q
            s = H.sum(axis=2)
            I[start:stop] = mass[start:stop, None] * (
                np.einsum("bpn,n->bp", H, Y, optimize=True)
                - s * y_bar[start:stop, None]
            )
            U[start:stop] = mass[start:stop, None, None] * (
                np.einsum("bpn,nd->bpd", H, Xc, optimize=True)
                - s[:, :, None] * Mc[start:stop, None, :]
            )

        return {
            "I": I,
            "U": U,
            "mass": mass,
            "mean": mean,
            "n_eff": 1.0 / np.square(A).sum(axis=1),
            "eta": eta,
        }

    def Calculate_statistic_sparse(self, weights, directions, *, X=None, Y=None):
        X, Y, W, Phi = self._prepare_inputs(weights, directions, X, Y, dense=False)
        J, P = Phi.shape[:2]
        mass = np.asarray(W.sum(axis=1)).ravel()
        A = W.copy()
        rows = np.repeat(np.arange(J), np.diff(A.indptr))
        A.data /= mass[rows]
        x_bar = X.mean(axis=0)
        y_bar = np.asarray(A @ Y).ravel()
        Xc = X - x_bar
        Mc = np.asarray(A @ Xc)
        mean = Mc + x_bar
        I = np.empty((J, P))
        U = np.empty((J, P, X.shape[1]))
        eta = np.empty((J, P))
        ones = np.ones(X.shape[0])

        for p in range(P):
            q = np.einsum(
                "ed,ed->e",
                Phi[rows, p],
                Xc[A.indices] - Mc[rows],
                optimize=True,
            )
            H = sparse.csr_matrix(
                (A.data * q, A.indices, A.indptr), shape=A.shape, copy=False
            )
            residual = np.asarray(H @ ones).ravel()
            denominator = np.asarray(abs(H) @ ones).ravel()
            eta[:, p] = np.divide(
                np.abs(residual),
                denominator,
                out=np.zeros_like(residual),
                where=denominator != 0,
            )
            q -= residual[rows]
            H = sparse.csr_matrix(
                (A.data * q, A.indices, A.indptr), shape=A.shape, copy=False
            )
            s = np.asarray(H @ ones).ravel()
            I[:, p] = mass * (np.asarray(H @ Y).ravel() - s * y_bar)
            U[:, p] = mass[:, None] * (np.asarray(H @ Xc) - s[:, None] * Mc)

        squared = A.copy()
        squared.data **= 2
        return {
            "I": I,
            "U": U,
            "mass": mass,
            "mean": mean,
            "n_eff": 1.0 / np.asarray(squared.sum(axis=1)).ravel(),
            "eta": eta,
        }

    @staticmethod
    def _normalized_residual(A, Q, residual):
        denominator = np.einsum("jn,jpn->jp", A, np.abs(Q), optimize=True)
        return np.divide(
            np.abs(residual),
            denominator,
            out=np.zeros_like(residual),
            where=denominator != 0,
        )

    def _prepare_inputs(self, weights, directions, X, Y, *, dense):
        X = self.X.T if X is None else X
        Y = self.Y if Y is None else Y
        X = self._finite_real_array(X, "X")
        Y = self._finite_real_array(Y, "Y")
        Phi = self._finite_real_array(directions, "directions")

        if X.ndim != 2:
            raise ValueError("X must have shape (n, d)")
        if Y.ndim != 1 or Y.shape[0] != X.shape[0]:
            raise ValueError("Y must have shape (n,)")
        if Phi.ndim != 3:
            raise ValueError("directions must have shape (J, P, d)")
        if Phi.shape[1] == 0:
            raise ValueError("directions must contain at least one direction")

        if sparse.issparse(weights):
            if not np.issubdtype(weights.dtype, np.number) or np.iscomplexobj(weights):
                raise TypeError("weights must have a real numeric dtype")
            if not np.all(np.isfinite(weights.data)):
                raise ValueError("weights must contain only finite values")
            if np.any(weights.data < 0):
                raise ValueError("weights must be nonnegative")
            W = sparse.csr_matrix(weights, dtype=float, copy=True)
            W.sum_duplicates()
            W.eliminate_zeros()
            W.sort_indices()
            if dense:
                W = W.toarray()
        else:
            W = self._finite_real_array(weights, "weights")
            if W.ndim != 2:
                raise ValueError("weights must have shape (J, n)")
            if np.any(W < 0):
                raise ValueError("weights must be nonnegative")
            if not dense:
                W = sparse.csr_matrix(W)

        if W.ndim != 2 or W.shape[1] != X.shape[0]:
            raise ValueError("weights must have shape (J, n)")
        if Phi.shape[0] != W.shape[0] or Phi.shape[2] != X.shape[1]:
            raise ValueError("directions must have shape (J, P, d)")

        with np.errstate(over="ignore", invalid="ignore"):
            mass = np.asarray(W.sum(axis=1)).ravel()
        if not np.all(np.isfinite(mass)):
            raise ValueError("every weight row must have finite mass")
        if np.any(mass <= 0):
            raise ValueError("every weight row must have positive mass")
        return X, Y, W, Phi

    @staticmethod
    def _finite_real_array(value, name):
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
            raise TypeError(f"{name} must have a real numeric dtype")
        array = array.astype(float, copy=False)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array
