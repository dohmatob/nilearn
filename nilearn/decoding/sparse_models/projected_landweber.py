# Author: DOHMATOB Elvis

import warnings
import numpy as np
from nilearn.decoding.sparse_models.common import (
    squared_loss_lipschitz_constant, gradient_id, div_id)


def lp_norm(z, p):
    # Compute lp norm of each 'atom'.
    if hasattr(p, '__iter__'):
        if np.isinf(p[1]):
            return np.max(lp_norm(z, p[0]), axis=0)
        else:
            raise NotImplementedError("l_%s mixed-norm not implemented!" % p)
    elif p == 2.:
        return np.sqrt(np.sum(z * z, axis=0))
    elif p == 1.:
        return np.sum(np.abs(z), axis=0)
    elif np.isinf(p):
        return np.max(np.abs(z), axis=0)
    else:
        warnings.warn("Don't know how to compute L_%g norm efficiently!" % p)
        return np.sum(np.abs(z) ** p) ** (1. / p)


def proj_l2infty(y, kappa, inplace=True):
    """Euclidean projection of `y` onto l2,infty ball of radius `kappa`."""
    # Misc / sanity
    if y.ndim == 1:
        y = np.array(y)[np.newaxis, :]
    else:
        assert y.ndim == 2, y.ndim
    out = y if inplace else np.array(y)

    # Do the actual projection (closed-form formula).
    lp = lp_norm(y, 2.)
    nz = (lp > 0.)
    shrink = np.zeros_like(y)
    shrink[:, nz] = kappa / np.maximum(lp[nz][np.newaxis, :], kappa)
    out *= shrink
    return out


def projected_landweber(A, AT, z, proj_p, p, lambd, stepsize, max_iter=100,
                        dgap_tol=1e-4, verbose=1, align="", init=None,
                        return_info=False):
    if init is None:
        init = {}
    n_features = A.shape[1]
    y = np.zeros(n_features) if init is not None else init
    dgap = np.inf
    converged = True
    for k in xrange(max_iter):
        if verbose:
            print "%sPROJECTED LANDWEBER: iteration %i/%i: dgap=%g" % (
                align, k + 1, max_iter, dgap)
        if dgap < dgap_tol:
            if verbose:
                print ("%sConverged after %i iterations (dgap = %g < %g"
                       " = dgap_tol)" % (align, k + 1, dgap, dgap_tol))
            break
        res = z - A(y)
        aux = AT(res)
        dgap = lambd * lp_norm(aux, p) - np.dot(y, aux)
        y += stepsize * aux  # foreward step
        y = proj_p(y, lambd)  # backward step
    else:
        converged = False

    if return_info:
        return y, dict(converged=converged)
    else:
        return y


class GradientId(object):
    def __init__(self, shape, l1_ratio):
        self.input_shape = shape
        n_features = np.prod(shape)
        self.shape = ((len(shape) + 1) * n_features, n_features)
        self.l1_ratio = l1_ratio
        self.L = 1.1 * (4 * len(shape) * (1 - l1_ratio) ** 2 + l1_ratio ** 2)

    def __call__(self, w):
        return gradient_id(w.reshape(self.input_shape),
                           l1_ratio=self.l1_ratio).reshape(self.shape[0])


class DivId(object):
    def __init__(self, shape, l1_ratio):
        self.input_shape = [len(shape) + 1] + list(shape)
        n_features = np.prod(shape)
        self.shape = (n_features, (len(shape) + 1) * n_features)
        self.shape = (n_features, (len(shape) + 1) * n_features)
        self.l1_ratio = l1_ratio
        self.L = 1.1 * (4 * len(shape) * (1 - l1_ratio) ** 2 + l1_ratio ** 2)

    def __call__(self, w):
        return -div_id(w.reshape(self.input_shape),
                       l1_ratio=self.l1_ratio).reshape(self.shape[0])


def prox_tv_l1(im, l1_ratio=.5, weight=50, dgap_tol=5.e-5, max_iter=10,
               verbose=False, init=None, return_info=False):
    D = GradientId(im.shape, l1_ratio)
    DT = DivId(im.shape, l1_ratio)
    proj = lambda w, kappa: proj_l2infty(w.reshape(
            (len(D.input_shape) + 1, -1)), kappa).ravel()
    out = projected_landweber(DT, D, im.reshape(D.shape[1]), proj,
                              (2, np.inf), weight, 2. / D.L, init=init,
                              max_iter=max_iter, dgap_tol=dgap_tol,
                              verbose=verbose and 0, return_info=return_info,
                              align="\t")
    if return_info:
        theta, info = out
    else:
        theta = out
    out = im - DT(theta).reshape(im.shape)
    if return_info:
        return out, info
    else:
        return out


class LinOp(object):
    def __init__(self, A):
        self.A = A
        self.shape = A.shape
        self.L = squared_loss_lipschitz_constant(A)

    def __call__(self, vec):
        return np.dot(self.A, vec)

    def T(self):
        return LinOp(self.A.T)


def test_proj_l2inty_not_outside_ball():
    import itertools
    from nose.tools import assert_true
    rng = np.random.RandomState(42)
    salt = 1e-10
    for d, p, kappa in itertools.product(xrange(1, 4), xrange(2, 10),
                                         np.logspace(-2, 2, num=6)):
        a = rng.randn(d, p)
        pa = proj_l2infty(a, kappa)
        assert_true(lp_norm(pa, (2., np.inf)) <= kappa + salt)


if __name__ == '__main__':
    from lp1_mixed_norm import proj_lp1
    A = LinOp(np.eye(5)[:, :3])
    z = np.array([3., 4., 0, 4, 0])
    p = 2.
    lambd = 5.
    proj_p = lambda y, lambd: proj_lp1(y.reshape((-1, 1)), lambd, p=p).ravel()
    print projected_landweber(A, A.T(), z, proj_p, p, lambd, 2. / A.L,
                              align="\t")
