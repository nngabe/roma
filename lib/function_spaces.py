from typing import List, Callable
import numpy as np
from scipy import linalg, interpolate
from sklearn import gaussian_process as gp

import jax
import jax.numpy as jnp
import jax.random as jr
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline as spline
import equinox as eqx

prng = lambda i=0: jax.random.PRNGKey(i)

class PowerSeries(eqx.Module):
    r"""Power series.

    p(x) = \sum_{i=0}^{N-1} a_i x^i

    Args:
        N (int)
        M (float): `M` > 0. The coefficients a_i are randomly sampled from [-`M`, `M`].
    """

    N: int
    M: float
    a: jnp.ndarray
    alpha: float

    def __init__(self, N=100, M=1.):
        self.N = N
        self.M = M
        self.a = self.M * ( 2. * jax.random.uniform(prng(0), (self.N,)) - 1.) 
        self.alpha = 0.25

    def __call__(self, xs):
        return jax.numpy.polyval(self.a, xs**self.alpha, unroll=128) 


class GRF(eqx.Module):
    """Gaussian random field (Gaussian process) in 1D."""

    N: int
    T: float
    interp: Callable
    x: np.ndarray
    K: np.ndarray
    L: np.ndarray
    num_func: int

    def __init__(self, T=1, kernel="RBF", length_scale=1, N=1000, num_func=10, interp="cubic"):
        self.N = N
        self.T = T
        self.interp  = lambda feat: spline(self.x, feat)
        self.x = np.linspace(0, T, num=N)
        self.num_func = num_func
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        elif kernel == "ExpSineSquared":
            K = gp.kernels.ExpSineSquared(length_scale=length_scale, periodicity=T)
        self.K = K(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))

    def field(self, key):
        u = jr.normal(key, (self.N, self.num_func))
        return jnp.einsum('ij,jk -> ki', self.L, u)

    def __call__(self, x, key):
        func_feats = self.field(key)
        f = [spline(self.x, ff) for ff in func_feats]
        func_vals = jnp.array([_f(x) for _f in f]).reshape(x.shape[0],-1)
        return func_vals

class GRF_KL(eqx.Module):
    """Gaussian random field (Gaussian process) in 1D.

    The random sampling algorithm is based on truncated Karhunen-Loeve (KL) expansion.

    """

    def __init__(
        self, T=1, kernel="RBF", length_scale=1, num_eig=10, N=100, interp="cubic"
    ):
        if not isclose(T, 1):
            raise ValueError("GRF_KL only supports T = 1.")

        self.num_eig = num_eig
        if kernel == "RBF":
            kernel = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            kernel = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        eigval, eigvec = eig(kernel, num_eig, N, eigenfunction=True)
        eigvec *= eigval**0.5
        x = np.linspace(0, T, num=N)
        self.eigfun = [
            interpolate.interp1d(x, y, kind=interp, copy=False, assume_sorted=True)
            for y in eigvec.T
        ]

    def bases(self, sensors):
        """Evaluate the eigenfunctions at a list of points `sensors`."""
        return np.array([np.ravel(f(sensors)) for f in self.eigfun])

    def random(self, size):
        return np.random.randn(size, self.num_eig)

    def eval_one(self, feature, x):
        eigfun = [f(x) for f in self.eigfun]
        return np.sum(eigfun * feature)

    def eval_batch(self, features, xs):
        eigfun = np.array([np.ravel(f(xs)) for f in self.eigfun]) 
        return np.dot(features, eigfun)


class GRF2D(eqx.Module):
    """Gaussian random field in [0, 1]x[0, 1].

    The random sampling algorithm is based on Cholesky decomposition of the covariance
    matrix.

    """

    def __init__(self, kernel="RBF", length_scale=1, N=100, interp="splinef2d"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(0, 1, num=N)
        self.y = np.linspace(0, 1, num=N)
        xv, yv = np.meshgrid(self.x, self.y)
        self.X = np.vstack((np.ravel(xv), np.ravel(yv))).T
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        self.K = K(self.X)
        self.L = np.linalg.cholesky(self.K + 1e-12 * np.eye(self.N**2))

    def random(self, size):
        u = np.random.randn(self.N**2, size)
        return np.dot(self.L, u).T

    def eval_one(self, feature, x):
        y = np.reshape(feature, (self.N, self.N))
        return interpolate.interpn((self.x, self.y), y, x, method=self.interp)[0]

    def eval_batch(self, features, xs):
        points = (self.x, self.y)
        ys = np.reshape(features, (-1, self.N, self.N))
        res = map(lambda y: interpolate.interpn(points, y, xs, method=self.interp), ys)
        return np.vstack(list(res)).astype(config.real(np))


def wasserstein2(space1, space2):
    """Compute 2-Wasserstein (W2) metric to measure the distance between two ``GRF``."""
    return (
        np.trace(space1.K + space2.K - 2 * linalg.sqrtm(space1.K @ space2.K)) ** 0.5
        / space1.N**0.5
    )


def eig(kernel, num, Nx, eigenfunction=True):
    """Compute the eigenvalues and eigenfunctions of a kernel function in [0, 1]."""
    h = 1 / (Nx - 1)
    c = kernel(np.linspace(0, 1, num=Nx)[:, None])[0] * h
    A = np.empty((Nx, Nx))
    for i in range(Nx):
        A[i, i:] = c[: Nx - i]
        A[i, i::-1] = c[: i + 1]
    A[:, 0] *= 0.5
    A[:, -1] *= 0.5

    if not eigenfunction:
        return np.flipud(np.sort(np.real(np.linalg.eigvals(A))))[:num]

    eigval, eigvec = np.linalg.eig(A)
    eigval, eigvec = np.real(eigval), np.real(eigvec)
    idx = np.flipud(np.argsort(eigval))[:num]
    eigval, eigvec = eigval[idx], eigvec[:, idx]
    for i in range(num):
        eigvec[:, i] /= np.trapz(eigvec[:, i] ** 2, dx=h) ** 0.5
    return eigval, eigvec
