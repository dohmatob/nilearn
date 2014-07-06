# Author: DOHMATOB Elvis

from math import sqrt
import warnings
import numpy as np
from nilearn.decoding.sparse_models.common import (
    squared_loss_lipschitz_constant, gradient_id, div_id, _unmask)

ZERO_PLUS = 1e-10


def norm_lp(z, p):
    """lp (mixed-)norm."""

    if hasattr(p, '__iter__'):
        # handle mixed-norms
        if np.isinf(p[1]):
            return np.max(norm_lp(z, p[0]), axis=0)
        elif p[1] == 1.:
            return np.sum(norm_lp(z, p[0]), axis=0)
        elif p[1] == 2.:
            aux = norm_lp(z, p[0])
            return np.sqrt(np.sum(aux * aux, axis=0))
        else:
            raise NotImplementedError(
                "l_%s mixed-norm not implemented!" % str(p))
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
    lp = norm_lp(y, 2.)
    nz = (lp > 0.)
    shrink = np.zeros_like(y)
    shrink[:, nz] = kappa / np.maximum(lp[nz][np.newaxis, :], kappa)
    out *= shrink
    toto = norm_lp(out, (2, np.inf))
    assert toto <= kappa + 1e-8, (toto, lambd)
    return out


def fista(f, f_grad, g_prox, total_energy, size, stepsize, verbose=1,
          max_iter=100, tol=0., callback=None):
    from math import sqrt
    x = np.zeros(size)
    y = np.array(x)
    t = 0
    old_energy = np.inf
    history = []
    for k in xrange(max_iter):
        old_x = x.copy()
        energy = total_energy(x)
        history.append(energy)
        energy_delta = old_energy - energy
        if verbose:
            print "FISTA: iteration %i/%i: E=%g, dE=%g" % (
                k + 1, max_iter, energy, energy_delta)
        if callback:
            callback(locals())
        if np.abs(energy_delta) < tol:
            if verbose:
                print "Coverged (dE < %g)" % tol
            break
        y -= stepsize * f_grad(y)
        x = g_prox(y, stepsize)
        old_t = t
        t = .5 * (1 + sqrt(1. + 4 * t ** 2))
        y = x + (1. + (old_t - 1.) / t) * (x - old_x)
        old_energy = energy

    return x, np.array(history)


def fista(f, f_grad, g_prox, total_energy, size, stepsize, verbose=1,
          max_iter=100, tol=0., callback=None):
    x = np.zeros(size)
    y = np.array(x)
    t = 0
    old_energy = np.inf
    history = []
    for k in xrange(max_iter):
        old_x = x.copy()
        energy = total_energy(x)
        history.append(energy)
        energy_delta = old_energy - energy
        if verbose:
            print "FISTA: iteration %i/%i: E=%g, dE=%g" % (
                k + 1, max_iter, energy, energy_delta)
        if callback:
            callback(locals())
        if np.abs(energy_delta) < tol:
            if verbose:
                print "Coverged (dE < %g)" % tol
            break
        y -= stepsize * f_grad(y)
        x = g_prox(y, stepsize)
        old_t = t
        t = .5 * (1 + sqrt(1. + 4 * t ** 2))
        y = x + (1. + (old_t - 1.) / t) * (x - old_x) * 0.
        old_energy = energy

    return y, np.array(history)


def fast_projected_gradient(A, AT, z, norm_lp, dual_norm_lp, proj_lp, lambd,
                            stepsize, init=None, max_iter=10, dgap_tol=1e-4,
                            verbose=1, align="", return_info=False):
    if init is None:
        init = {}
    n_features = A.shape[1]
    y = np.zeros(n_features) if init is not None else init
    dgap = np.inf
    converged = True
    history = []
    for k in xrange(max_iter):
        if verbose:
            print "%sFPG: iteration %i/%i: dgap=%g" % (
                align, k + 1, max_iter, dgap)
        if dgap < dgap_tol:
            if verbose:
                print ("%sConverged after %i iterations (%g = dgap < dgap_tol"
                       " = %g)" % (align, k + 1, dgap, dgap_tol))
            break
        res = z - A(y)
        aux = AT(res)
        dgap = lambd * dual_norm_lp(aux) - np.dot(y, aux)
        history.append(dgap)
        assert dgap >= 0. or -dgap < ZERO_PLUS, dgap
        y += stepsize * aux  # foreward step
        y = proj_lp(y, lambd)  # backward step
    else:
        converged = False

    # history = np.array(history)
    # import pylab as pl
    # pl.loglog(history - history.min())
    # pl.show()
    if return_info:
        return y, res, history, dict(converged=converged)
    else:
        return y, res, history


class GradientId(object):
    def __init__(self, shape, l1_ratio):
        self.input_shape = shape
        self.n_features = np.prod(shape)
        self.ndim = len(shape)
        self.shape = ((self.ndim + 1) * self.n_features, self.n_features)
        self.l1_ratio = l1_ratio
        self.L = 4 * len(shape) * (1 - l1_ratio) ** 2 + l1_ratio ** 2

    def __call__(self, w):
        """matvec operation."""
        return gradient_id(w.reshape(self.input_shape),
                           l1_ratio=self.l1_ratio).reshape(self.shape[0])


class DivId(object):
    def __init__(self, shape, l1_ratio):
        self.ndim = len(shape)
        self.input_shape = [self.ndim + 1] + list(shape)
        self.n_features = np.prod(shape)
        self.shape = (self.n_features, (self.ndim + 1) * self.n_features)
        self.l1_ratio = l1_ratio
        self.L = 1.1 * (4 * len(shape) * (1 - l1_ratio) ** 2 + l1_ratio ** 2)

    def __call__(self, w):
        """matvec operation."""
        return -div_id(w.reshape(self.input_shape),
                       l1_ratio=self.l1_ratio).reshape(self.shape[0])


def prox_tv_l1(im, l1_ratio=.5, weight=50, dgap_tol=5.e-5, max_iter=10,
               verbose=False, init=None, return_info=False):
    """TV-l1 prox by means of projected Landweber iterations."""
    D = GradientId(im.shape, l1_ratio)
    DT = DivId(im.shape, l1_ratio)

    def proj(w, kappa): return proj_l2infty(w.reshape(
            (D.ndim + 1, -1)), kappa).ravel()

    norm = lambda w: norm_lp(w.reshape((D.ndim + 1, D.n_features)),
                             (2, np.inf))
    dual_norm = lambda w: norm_lp(w.reshape((D.ndim + 1, D.n_features)),
                             (2, 1))
    from grail import plw
    out = fast_projected_gradient(
        DT, D, im.reshape(D.shape[1]), norm, dual_norm, proj, weight, 2. / D.L, init=init,
        max_iter=max_iter, dgap_tol=dgap_tol, return_info=return_info,
        verbose=verbose, align="\t")
    if return_info:
        _, res, _, info = out
    else:
        _, res, _ = out
    res = res.reshape(im.shape)
    if return_info:
        return res, info
    else:
        return res


class NesterovSmoother(object):
    def __init__(self, X, y, alpha, l1_ratio, mask=None, shape=None,
                 callback=None):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.mask = mask
        self.shape = shape if mask is None else mask.shape
        self.n_axes = len(self.shape)
        self.n_features = X.shape[1]
        self.p = np.prod(self.shape)
        self.callback = callback

    def u_opt(self, Dw):
        return proj_l2infty(Dw.reshape((self.n_axes + 1, self.p)) / self.mu,
                            1., inplace=False)

    def fg(self, w):
        if self.callback:
            if self.callback(dict(w=w)):
                raise StopIteration('die!')

        w = _unmask(w, self.mask)
        w = w.reshape(self.shape)
        Dw = gradient_id(w, l1_ratio=self.l1_ratio)
        u = self.u_opt(Dw)
        energy = np.dot(Dw.ravel(), u.ravel()) - .5 * self.mu * (u * u).sum()
        energy *= self.alpha
        grad = -div_id(u.reshape([len(self.shape) + 1] + list(self.shape)),
                       l1_ratio=self.l1_ratio)
        if self.mask is None:
            w = w.ravel()
            grad = grad.ravel()
        else:
            w = w[self.mask]
            grad = grad[self.mask]
        grad *= self.alpha
        if self.X is None:
            res = w - self.y
            energy += .5 * np.dot(res, res)
            grad += res
        else:
            res = np.dot(self.X, w) - self.y
            energy += .5 * np.dot(res, res)
            grad += np.dot(self.X.T, res)
        return energy, grad

    def solve(self, mu=1., continuation=True, max_iter=100, verbose=1,
              init=None, **kwargs):
        from scipy.optimize import fmin_l_bfgs_b
        if init is None:
            init = {}
        x0 = init.get("w", np.zeros(self.n_features))
        mu = init.get("mu", mu)
        mu = max(mu, 1e-8)
        #while mu >= 1e-8:
        for mu in [mu, mu * .5]:
            try:
                self.mu = mu
                x0 = fmin_l_bfgs_b(self.fg, x0, iprint=verbose,
                                   maxiter=max_iter)[0]
                # if continuation:
                #     mu *= .5
                # else:
                #     break
            except StopIteration:
                break
        return x0, None, dict(w=x0.copy(), mu=mu)

def tv_solver(im, l1_ratio=.5, weight=50., verbose=1, max_iter=100):
    smoother = NesterovSmoother(im.shape, l1_ratio, .01)
    pass
    


class LinOp(object):
    def __init__(self, A):
        self.A = A
        self.shape = A.shape
        self.n_features = A.shape[1]
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
        assert_true(norm_lp(pa, (2., np.inf)) <= kappa + salt)


if __name__ == '__main__':
    from lp1_mixed_norm import proj_lp1
    A = LinOp(np.eye(5)[:, :3])
    z = np.array([3., 0., 4, 0., 0.])  # + .01 * np.random.randn(5)
    p = 2.
    lambd = 5.
    proj = lambda y, lambd: proj_lp1(y.reshape((-1, 1)), lambd, p=p).ravel()
    norm = lambda y: norm_lp(y, p)
    print fast_projected_gradient(A, A.T(), z, proj, norm, lambd, 2. / A.L,
                                  align="\t")
