import warnings
import numpy as np


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
    return out


def proj_lp1(y, kappa, p=2., inplace=True):
    """Projection of `y` onto the lp,1 mixed-norm ball of radius `kappa`.


    Parameters
    ----------
    y: 2D array of shape (d, p)
        Point being euclidean-projected.

    kappa: positve float
        Radius of the lp,1 ball onto which projection is to be done.

    p: float > 1, optional (default 2)
        The `p` in the mixed-norm lp,1.

    inplace: boolean, optional (default False)
        If set, then `y` is modified in-place to contain the output.

    Notes
    -----
    The overall algorithm has linear time and space complexity, ie. O(p).

    References
    ----------
    Jitkomut Songsiri, "Projection onto l1-norm Ball with Application to
        Indentification of Sparse Autoregressive Models"
    """

    # Misc / sanity
    if y.ndim == 1:
        y = np.array(y)[np.newaxis, :]
    else:
        assert y.ndim == 2, y.ndim
        out = y if inplace else np.array(y, dtype=np.float)
    ncol = y.shape[1]

    # It's easier to project onto unit ball.
    kappa *= 1.
    out /= kappa

    # Compute lp norm of each 'atom'.
    if p == 2.:
        d = np.sqrt(np.sum(out * out, axis=0))
    elif p == 1.:
        d = np.sum(np.abs(out), axis=0)
    elif np.isinf(p):
        d = np.max(np.abs(out), axis=0)
    else:
        warnings.warn("Don't know how to compute L_%g norm efficiently!" % p)
        d = np.sum(np.abs(out) ** p, axis=0) ** (1. / p)
    ### The trick begins here #################################################
    lp1 = d.sum()  # the lp1 norm of y
    if 1. < lp1:
        '''
        Root-finding: find the unique lambda > 0. such that:
            np.sum(np.maximum(d - lambd, 0)) - 1 = 0
        The algorithm below runs in linear time, i.e O(ncol).
        '''
        d_ = np.array(d)  # Keep orig d intact for later (during thresholding)!
        d_.sort()
        s = np.cumsum(d_[::-1])[::-1]
        s -= 1.
        s_ = np.zeros(ncol + 1)
        s_[0] = s[0]
        s_[1:-1] = s[1:] - (np.arange(1, ncol)[::-1] * d_[:-1])
        s_[-1] = -1.
        bp = np.nonzero(s_ >= 0)[0][-1]
        lambd = s[bp] / (ncol - bp)
        assert lambd > 0.

        # Thresholding. By construction, at the end we must have:
        #     ||out||_(p,1) = 1.
        shrink = np.zeros_like(out)
        msk = d.nonzero()
        shrink[:, msk] = np.maximum(1. - lambd / d[msk][np.newaxis, :], 0.)
        out *= shrink

    # Re-scale to correct for ball radius.
    out *= kappa

    return out


def fast_projected_landweber(A, AT, z, lambd, norm_lp, proj_lp, init=None,
                             dgap_tol=1e-4, max_iter=100, no_acc=False,
                             callback=None, verbose=1, return_info=True,
                             align=""):
    """Fast Projected Landweber Iteration (FPLI).

    The problem solved is:

        argmin_u .5 * < z - A(u), z - A(u)> s.t. ||u||_p <= lambda

    where lambda > 0, and p denotes a (possibly mixed) p norm, A is a bounded,
    linear operator of shape (s, t) and z is and s-dimemsional vector.

    Notes
    -----
    - Duality hereunder should be understood in the sense of Fenchel,
      Rockafellar, Chambolle, etc.

    - This algorithm can be used to solve (isotropic!) TV-denoising by letting
      A = div, and p = (2, np.inf).

    Returns
    -------
    y: 1D array of length `A.shape[1]`
        A solution to the problem.

    res: 1D array of shape `A.shape[0]`
        Residual: z - A(y)
    info: dict
        Convergence info. Keys are:
        "converged": bool: True if algorithm converged (to withing prescribed)
                     precision on the dual-gap.
    """
    if init is None:
        init = {}
    y = np.zeros(A.shape[1])
    u = np.array(y)
    stepsize = 1. / A.L  # or anything smaller than this
    dgap = np.inf
    t = 1.
    converged = False
    for k in xrange(max_iter):
        old_u = u
        if verbose:
            print "%sProjected Landweber: iter %i/%i: dgap=%g" % (
                align, k + 1, max_iter, dgap)
        if dgap < dgap_tol:
            if verbose:
                print "%sConverged (dgap < %g) after %i iterations." % (
                    align, dgap, k + 1)
            converged = True
            break
        # compute dual gap
        res = z - A(y)
        aux = AT(res)
        dgap = lambd * norm_lp(aux) - np.dot(y, aux)

        # invoke callback
        if callback:
            if callback(locals()):
                print "%sCallback exited with nonzero status. Aborting..." % (
                    align)
                break

        # forward step
        aux *= stepsize
        y += aux

        # backward step
        u = proj_lp(y, lambd)
        if no_acc:
            y = u
        else:
            # acceleration (Nesterov) step
            old_t = t
            t = .5 * (1 + np.sqrt(1 + 4 * t ** 2))
            acc = (old_t - 1.) / t
            y = u + acc * (u - old_u)

    if return_info:
        return y, res, dict(converged=converged)
    else:
        return y, res


def prox_tv_l1(im, l1_ratio=0., weight=50, dgap_tol=5.e-5, max_iter=10,
               verbose=False, init=None, return_info=False, no_acc=False,
               callback=None):
    """TV-l1 prox by means of projected Landweber iterations."""

    from common import gradient_id, div_id

    class GradientId(object):
        """Object that knows how to compute gradient + id."""
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
        """Object that knows how to compute divergence + id."""
        def __init__(self, shape, l1_ratio):
            self.ndim = len(shape)
            self.input_shape = [self.ndim + 1] + list(shape)
            self.n_features = np.prod(shape)
            self.shape = (self.n_features, (self.ndim + 1) * self.n_features)
            self.l1_ratio = l1_ratio
            self.L = 1.1 * (4 * len(shape) * (1 - l1_ratio
                                              ) ** 2 + l1_ratio ** 2)

        def __call__(self, w):
            """matvec operation."""
            return -div_id(w.reshape(self.input_shape),
                           l1_ratio=self.l1_ratio).reshape(self.shape[0])

    dual_norm = lambda w: norm_lp(w.reshape((D.ndim + 1, D.n_features)),
                             (2, 1.))
    proj = lambda w, kappa: proj_l2infty(w.reshape(
            (D.ndim + 1, -1)), kappa).ravel()
    shape = im.shape
    D = GradientId(shape, l1_ratio)
    DT = DivId(shape, l1_ratio)
    init = None if init is None else init.ravel()
    out = fast_projected_landweber(
        DT, D, im.ravel(), weight, dual_norm, proj, dgap_tol=dgap_tol,
        init=init, max_iter=max_iter, no_acc=no_acc, callback=callback,
        verbose=verbose and 0, align="\t")

    if return_info:
        return out[1].reshape(shape), out[2]
    else:
        return out[1].reshape(shape)

if __name__ == "__main__":
    import pylab as pl

    ### load picture #########################################################
    im = pl.imread(
        "/home/elvis/Downloads/toolbox_signal/altered_hibiscus.jpeg")

    # corrupt the picture by adding white noise
    noise_std = 10.
    z = im + noise_std * np.random.randn(*im.shape)

    ### TV denoising #########################################################
    import time
    dgaps = {}
    times = {}
    lambd = 5.
    for no_acc in [True, False]:
        solver = '%s-PLI' % ('Ordinary' if no_acc else 'Fast')
        dgaps[solver] = []
        times[solver] = []
        t0 = time.time()

        def cb(env):
            times[solver].append(time.time() - t0)
            dgaps[solver].append(env['dgap'])

        out = prox_tv_l1(z, weight=lambd, max_iter=1000, no_acc=no_acc,
                         callback=cb, return_info=False, verbose=1)
        times[solver] = np.array(times[solver])
        dgaps[solver] = np.array(dgaps[solver])

    ### Visualizations ########################################################
    min_dgap = min(*map(np.min, dgaps.values()))
    pl.gray()
    pl.imshow(im)
    pl.axis('off')
    pl.title("original")

    pl.figure()
    pl.imshow(z)
    pl.axis('off')
    pl.title("corrupted (noise std: %g)" % noise_std)
    pl.axis("off")

    pl.figure()
    pl.imshow(out)
    pl.title("corrected (TV denoising with lambda=%g)" % lambd)
    pl.axis('off')
    pl.tight_layout()

    pl.figure()
    pl.title("Projected Landweber Iteration (PLI) for TV-denoising")
    colors = "mcybrg"
    for c, solver in enumerate(times.keys()):
        pl.semilogy(times[solver], dgaps[solver] - min_dgap, linewidth=3,
                  c=colors[c], label=solver)

    pl.xlabel("time (s)")
    pl.ylabel("excess dual gap")
    pl.legend(loc="best")
    pl.show()
