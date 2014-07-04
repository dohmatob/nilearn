"""
Euclidean projections onto l(p,1) mixed-norm balls, proximity operators of
l(p,infty) mixed-norms, etc., and similar business.

"""
# Author: DOHMATOB Elvis Dopgima
# License: simplified BSD

import warnings
import numpy as np


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
    assert y.ndim == 2, y
    out = y if inplace else np.array(y)
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


def _harmonic_conjugate(p):
    """Given p > 1, this function computes it's harminic conjugate.

    Precisely, we return q > 1 st. 1 / p + 1 / q = 1.
    """

    assert p >= 1.
    if p == 1:
        return np.inf
    elif np.isinf(p):
        return 1.
    else:
        return p / (p - 1.)


def prox_lpinfty(y, kappa, p=2, inplace=True):
    """Computes the prox at y, of the lp,infty mixed-norm. By definition, this
    precisely corresponds to the unique solution of the following problem:

        argmin .5 * < z - y, z - y > + kappa * ||z||_p,infty
            z

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

    Referneces
    ----------
    [1] - Jalal M. Fadili and G. Peyre's paper,
          "Total Variation with Fist Order Schemes"
    """

    # We use the fact that prox_lp,infty = 1 - proj_lp,1 (dual norms)
    out = y if inplace else np.array(y)
    out -= proj_lp1(y, kappa, p=_harmonic_conjugate(p), inplace=inplace)
    return out


def test_proj_lp1_simple_examples():
    np.testing.assert_array_equal(
        proj_lp1(np.array([[1. / 4, 3. / 4.]]) + 1e-3, 1.),
        np.array([[0.25, 0.75]]))

    np.testing.assert_array_equal(
        proj_lp1(np.array([[1.]]) + 1e-2, 1.), np.array([[1.]]))


def test_proj_lp1_is_not_outside_ball():
    import itertools
    from nose.tools import assert_true
    rng = np.random.RandomState(42)
    salt = 1e-10

    for d, p, kappa in itertools.product(xrange(1, 4), xrange(2, 10),
                                         np.logspace(-2, 2, num=6)):
        a = rng.randn(d, p)
        pa = proj_lp1(a, kappa)
        assert_true(np.sum(np.sqrt(np.sum(pa * pa, axis=0))) <= kappa + salt)


def test_proj_lp1_is_identity_on_interior():
    rng = np.random.RandomState(42)

    for d in xrange(1, 4):
        for p in xrange(1, 5):
            a = rng.randn(d, p)
            kappa = np.sum(np.sqrt(np.sum(a * a, axis=0)))
            np.testing.assert_array_almost_equal(proj_lp1(a, kappa, inplace=0),
                                                 a, decimal=8)


def test_prox_lpinfty_generalizes_prox_l1():
    from nilearn.decoding.sparse_models.operators import prox_l1
    rng = np.random.RandomState(42)
    points = rng.randn(10)
    for kappa in np.logspace(-2, +2, num=5):
        np.testing.assert_array_almost_equal(np.array([prox_lpinfty(
                        np.array([[point]]), kappa, inplace=False
                        ) for point in points]).ravel(), prox_l1(
                points.copy(), kappa), decimal=12)


def test_proj_lp1_inplace():
    from nose.tools import assert_true
    rng = np.random.RandomState(42)
    y = rng.randn(1, 5)
    kappa = np.abs(y).sum()

    # z is modified inplace
    z = y * 1.1
    np.testing.assert_array_equal(proj_lp1(z, kappa), z)

    # z stays intact
    z = y * 1.1
    assert_true(np.any(proj_lp1(z, kappa, inplace=False) != z))


def test_harmonic_conjugate():
    from nose.tools import assert_equal
    assert_equal(_harmonic_conjugate(np.inf), 1.)
    assert_equal(_harmonic_conjugate(1.), np.inf)
    assert_equal(_harmonic_conjugate(2.), 2.)


def test_prox_lpinfty_inplace():
    from nose.tools import assert_true
    rng = np.random.RandomState(42)
    y = rng.randn(1, 5)
    kappa = np.abs(y).sum()

    # z is modified inplace
    z = y * 1.1
    np.testing.assert_array_equal(prox_lpinfty(z, kappa), z)

    # z stays intact
    z = y * 1.1
    assert_true(np.any(prox_lpinfty(z, kappa, inplace=False) != z))


if __name__ == '__main__':
    import pylab as pl
    noise_sigma = 5e-2
    support = [0, 1, 2, 3, 4, 27, 19, 87, 100, 200, 600, -15, -10]
    x = np.zeros(1000)
    kappa = len(support)  # we assume we know the DoF
    x[support] = 1
    y = x + np.random.randn(*x.shape) * noise_sigma
    sol = proj_lp1(y[np.newaxis, :], kappa, inplace=False).ravel()

    pl.suptitle("lp,1 projections demo")
    ax1 = pl.subplot(221)
    ax1.plot(x, "s-")
    ax1.set_title("original signal x")
    ax2 = pl.subplot(222)
    ax2.plot(y, "s-")
    ax2.set_title("corrupt signal y (noise sigma: %g)" % noise_sigma)
    ax3 = pl.subplot(223)
    ax3.plot(sol, "s-")
    ax3.set_title(
        "signal recovered by projecting y onto the l1 ball of radius %g" % (
            kappa))
    ax4 = pl.subplot(224)
    ax4.plot(x - sol, "s-")
    ax4.set_title("residual")
    pl.show()
