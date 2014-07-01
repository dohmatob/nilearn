"""
sklearn-compatible Cross-Validation module for TV-l1, S-LASSO, etc. models

"""
# Author: DOHMATOB Elvis Dopgima,
#         Gaspar Pizarro,
#         Gael Varoquaux,
#         Alexandre Gramfort,
#         Bertrand Thirion,
#         and others.
# License: simplified BSD

from functools import partial
import numpy as np
from sklearn.externals.joblib import Memory, Parallel, delayed
from sklearn.cross_validation import check_cv
from .._utils.fixes import center_data, LabelBinarizer, roc_auc_score
from .common import _sigmoid
from .estimators import (_BaseRegressor, _BaseClassifier, _BaseEstimator,
                         SmoothLassoRegressor, SmoothLassoClassifier,
                         TVl1Classifier, TVl1Regressor)
from .smooth_lasso import smooth_lasso_logistic, smooth_lasso_squared_loss
from .tv import tvl1_solver
from ._cv_tricks import (EarlyStoppingCallback, RegressorFeatureSelector,
                         ClassifierFeatureSelector, _my_alpha_grid)
from .._utils.fixes import is_regressor, is_classifier


def logistic_path_scores(solver, X, y, alphas, l1_ratio, train,
                         test, tol=1e-4, max_iter=1000, init=None,
                         mask=None, verbose=0, key=None,
                         screening_percentile=10., memory=Memory(None),
                         **kwargs):
    """Function to compute scores of different alphas in classification
    used by CV objects.

    Parameters
    ----------
    alphas: list of floats
        List of regularization parameters being considered.

    l1_ratio: float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV (resp. Smooth Lasso) penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.

    solver: function handle
       See for example tv.TVl1Classifier documentation.

    """

    # univariate feature screening
    selector = ClassifierFeatureSelector(percentile=screening_percentile,
                                         mask=mask)
    X = selector.fit_transform(X, y)
    mask = selector.mask_

    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]
    test_scores = []

    def _test_score(w):
        return 1. - roc_auc_score(
            (y_test > 0.), _sigmoid(np.dot(X_test, w[:-1]) + w[-1]))

    # setup callback mechanism for ealry stopping
    callerback = EarlyStoppingCallback(X_test, y_test, verbose=verbose)
    env = dict(counter=0)

    def _callback(_env):
        if not isinstance(_env, dict):
            _env = dict(w=_env)

        _env['w'] = _env['w'][:-1]  # strip off intercept
        env["counter"] += 1
        _env["counter"] = env["counter"]

        return callerback(_env)

    best_score = np.inf
    for alpha in alphas:
        w, _, init = solver(
            X_train, y_train, alpha, l1_ratio, mask=mask, tol=tol,
            max_iter=max_iter, init=init, verbose=verbose, callback=_callback,
            **kwargs)
        score = _test_score(w)
        test_scores.append(score)
        if score <= best_score:
            best_score = score
            best_alpha = alpha

    # Re-fit best model to high precision (i.e without early stopping, etc.).
    # N.B: This work is cached, just in case another worker on another fold
    # reports the same best alpha. Also note that the re-fit is done only on
    # the train (i.e X_train), a piece of the design X.
    best_w, _, init = memory.cache(solver)(
        X_train, y_train, best_alpha, l1_ratio, mask=mask, tol=tol,
        max_iter=max_iter, verbose=verbose, **kwargs)

    # unmask univariate screening
    best_w = selector.inverse_transform(best_w)

    return test_scores, best_w, key


def squared_loss_path_scores(solver, X, y, alphas, l1_ratio, train, test,
                             tol=1e-4, max_iter=1000, init=None, mask=None,
                             debias=False, ymean=0., verbose=0,
                             key=None, screening_percentile=10.,
                             memory=Memory(None), **kwargs):
    """Function to compute scores of different alphas in regression.
    used by CV objects.

    Parameters
    ----------
    alphas: list of floats
        List of regularization parameters being considered.

    l1_ratio: float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV (resp. Smooth Lasso) penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.

    solver: function handle
       See for example tv.TVl1Regressor documentation.

    """

    # univariate feature screening
    selector = RegressorFeatureSelector(percentile=screening_percentile,
                                        mask=mask)
    X = selector.fit_transform(X, y)
    mask = selector.mask_

    # make train / test datasets
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]

    def _test_score(w):
        """Helper function to compute score of model with given wieghts map (
        loadings vector).

        """

        # debias to correct for DoF
        if debias:
            y_pred = np.dot(X_test, w)
            scaling = np.dot(y_pred, y_pred)
            if scaling > 0.:
                scaling = np.dot(y_pred, y_test) / scaling
                w *= scaling
        y_pred = np.dot(X_test, w) + ymean  # don't forget to add intercept!
        score = .5 * np.mean((y_test - y_pred) ** 2)
        return score

    # setup callback mechanism for ealry stopping
    callerback = EarlyStoppingCallback(X_test, y_test, verbose=verbose)
    env = dict(counter=0)

    def _callback(_env):
        if not isinstance(_env, dict):
            _env = dict(w=_env)

        env["counter"] += 1
        _env["counter"] = env["counter"]

        return callerback(_env)

    # rumble down regularization path (with warm-starts)
    test_scores = []
    best_score = np.inf
    for alpha in alphas:
        w, _, init = solver(
            X_train, y_train, alpha, l1_ratio, mask=mask, tol=tol,
            max_iter=max_iter, init=init, callback=_callback, verbose=verbose,
            **kwargs)

        # compute score on test data
        score = _test_score(w)
        test_scores.append(score)
        if score <= best_score:
            best_alpha = alpha
            best_score = score

    # Re-fit best model to high precision (i.e without early stopping, etc.).
    # N.B: This work is cached, just in case another worker on another fold
    # reports the same best alpha. Also note that the re-fit is done only on
    # the train (i.e X_train), a piece of the design X.
    best_w, _, init = memory.cache(solver)(
        X_train, y_train, best_alpha, l1_ratio, mask=mask, tol=tol,
        max_iter=max_iter, verbose=verbose, **kwargs)

    # Unmask univariate screening
    best_w = selector.inverse_transform(best_w)

    return test_scores, best_w, key


class _BaseCV(_BaseEstimator):
    """
    Parameters
    ----------
    alphas: list of floats, optional (default None)
        Choices for the constant that scales the overall regularization term.
        This parameter is mutually exclusive with the `n_alphas` parameter.

    n_alphas: int, optional (default 10).
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    eps: float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    alpha_min: float, optional (default 1e-6)
        Minimum value of alpha to consider. This is mutually exclusive with the
        `eps` parameter.

    l1_ratio: float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV, etc., penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.

    mask: multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    screeing_percentile: float in the interval [0, 100]; Optional (default 10)
        Percentile value for ANOVA univariate feature selection. A value of
        100 means "keep all features".

    max_iter: int
        Defines the iterations for the solver. Defaults to 1000

    tol: float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose: int, optional (default 0)
        Verbosity level.

    backtracking: bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

    callback: callable(dict) -> bool
        Function called at the end of every energy descendent iteration of the
        solver. If it returns True, the loop breaks.

    n_jobs: int, optional (default 1)
        Number of jobs to use for One-vs-All classification.

    cv: int, a cv generator instance, or None (default 10)
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    Attributes
    ----------
    `alpha_`: float
         Best alpha found by cross-validation

    `coef_`: array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    `intercept_`: array, shape = [n_classes-1]
         Intercept (a.k.a. bias) added to the decision function.
         It is available only when parameter intercept is set to True.

    `scores_`: 2d array of shape (n_alphas, n_folds)
        Scores (misclassification) for each alpha, and on each fold

    """

    path_scores_func = None

    def __init__(self, alphas=None, l1_ratio=.5, mask=None, max_iter=1000,
                 tol=1e-4, memory=Memory(None), copy_data=True,
                 standardize=False, normalize=False, alpha_min=1e-6,
                 verbose=0, n_jobs=1, callback=None, n_alphas=10, eps=1e-3,
                 fit_intercept=True, cv=10, backtracking=False,
                 bagging=True, screening_percentile=10.):
        super(_BaseCV, self).__init__(
            l1_ratio=l1_ratio, mask=mask, max_iter=max_iter, tol=tol,
            memory=memory, copy_data=copy_data, verbose=verbose,
            fit_intercept=fit_intercept, backtracking=backtracking,
            normalize=normalize, standardize=standardize)
        self.n_jobs = n_jobs
        self.cv = cv
        self.n_alphas = n_alphas
        self.eps = eps
        self.alphas = alphas
        self.alpha_min = alpha_min
        self.bagging = bagging
        self.screening_percentile = 10
        assert 0. <= screening_percentile <= 100.

        # sanitize path_scores_func
        if self.path_scores_func is None:
            raise ValueError(
                "Class '%s' doesn't have a `path_scores_func` attribute!" % (
                    self.__class__.__name__))

    @property
    def short_name(self):
        return '%s(l1_ratio=%g)' % (self.__class__.__name__, self.l1_ratio)

    def fit(self, X, y):
        # misc
        self.__class__.__name__.endswith("CV")
        model_class = eval(self.__class__.__name__[:-2])
        solver = eval(self.solver)
        path_scores_func = eval(self.path_scores_func)
        tricky_kwargs = {}
        if hasattr(self, "debias"):
            tricky_kwargs["debias"] = getattr(self, "debias")

        # always a good idea to centralize / normalize data
        ymean = 0.
        if self.standardize:
            X, y, Xmean, ymean, Xstd = center_data(
                X, y, copy=True, normalize=self.normalize,
                fit_intercept=self.fit_intercept)
            if is_regressor(self):
                tricky_kwargs["ymean"] = ymean

        # make / sanitize alpha grid
        if self.alphas is None:
            # XXX Are these alphas reasonable ?
            alphas = _my_alpha_grid(X, y, l1_ratio=self.l1_ratio,
                                    eps=self.eps, n_alphas=self.n_alphas,
                                    standardize=self.standardize,
                                    normalize=self.normalize,
                                    alpha_min=self.alpha_min,
                                    fit_intercept=self.fit_intercept,
                                    logistic=is_classifier(self))
        else:
            alphas = np.array(self.alphas)

        # always sort alphas from largest to smallest
        alphas = np.array(sorted(alphas)[::-1])

        cv = list(check_cv(self.cv, X=X, y=y, classifier=is_classifier(self)))
        self.n_folds_ = len(cv)

        # misc (different for classifier and regressor)
        if is_classifier(self):
            X, y = self._pre_fit(X, y)
        if is_classifier(self) and self.n_classes_ > 2:
            n_problems = self.n_classes_
        else:
            n_problems = 1
            y = y.ravel()
        self.scores_ = [[]] * n_problems
        w = np.zeros((n_problems, X.shape[1] + int(is_classifier(self))))

        # parameter to path_scores function
        path_params = dict(mask=self.mask, tol=self.tol, verbose=self.verbose,
                           max_iter=self.max_iter, rescale_alpha=True,
                           backtracking=self.backtracking, memory=self.memory,
                           screening_percentile=self.screening_percentile)
        path_params.update(tricky_kwargs)

        _ovr_y = lambda c: y[:, c] if is_classifier(
            self) and self.n_classes_ > 2 else y

        # main loop: loop on classes and folds
        for test_scores, best_w, c in Parallel(n_jobs=self.n_jobs)(
            delayed(self.memory.cache(path_scores_func))(
                solver, X, _ovr_y(c), alphas, self.l1_ratio, train, test,
                key=c, **path_params) for c in xrange(n_problems) for (
                train, test) in cv):
            test_scores = np.reshape(test_scores, (-1, 1))
            if not len(self.scores_[c]):
                self.scores_[c] = test_scores
            else:
                self.scores_[c] = np.hstack((self.scores_[c], test_scores))
            if self.bagging:
                w[c] += best_w

        self.alphas_ = alphas
        self.i_alpha_ = [np.argmin(np.mean(self.scores_[c], axis=-1))
                         for c in xrange(n_problems)]
        if n_problems == 1:
            self.i_alpha_ = self.i_alpha_
        self.alpha_ = alphas[self.i_alpha_]

        if self.bagging:
            # take average of best weights maps over folds
            w /= self.n_folds_
        else:
            # re-fit model with best params
            # XXX run this in parallel (use n_jobs)!
            for c in xrange(n_problems):
                params = dict((k, v) for k, v in self.get_params().iteritems()
                              if k in model_class().get_params())
                params["alpha"] = self.alpha_[c]
                if is_regressor(self):
                    selector = RegressorFeatureSelector(
                        percentile=self.screening_percentile,
                        mask=self.mask)
                else:
                    selector = ClassifierFeatureSelector(
                        percentile=self.screening_percentile,
                        mask=self.mask)
                X = selector.fit_transform(X, y)
                params["mask"] = selector.mask_
                w[c] = selector.inverse_transform(model_class(
                        **params).fit(X, y).w_)

        if is_classifier(self):
            self._set_coef_and_intercept(w)
        else:
            self.coef_ = w
            if self.standardize:
                self._set_intercept(Xmean, ymean, Xstd)
            else:
                self.intercept_ = 0.

        if n_problems == 1:
            w = w[0]
            self.scores_ = self.scores_[0]

        return self


class _BaseRegressorCV(_BaseCV, _BaseRegressor):
    """
    Parameters
    ----------
    alphas: list of floats, optional (default None)
        Choices for the constant that scales the overall regularization term.
        This parameter is mutually exclusive with the `n_alphas` parameter.

    n_alphas: int, optional (default 10).
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    eps: float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    alpha_min: float, optional (default 1e-6)
        Minimum value of alpha to consider. This is mutually exclusive with the
        `eps` parameter.

    l1_ratio: float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV, etc., penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.

    mask: multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    screeing_percentile: float in the interval [0, 100]; Optional (default 10)
        Percentile value for ANOVA univariate feature selection. A value of
        100 means "keep all features".

    max_iter: int
        Defines the iterations for the solver. Defaults to 1000

    tol: float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose: int, optional (default 0)
        Verbosity level.

    backtracking: bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

    callback: callable(dict) -> bool
        Function called at the end of every energy descendent iteration of the
        solver. If it returns True, the loop breaks.

    n_jobs: int, optional (default 1)
        Number of jobs to use for One-vs-All classification.

    cv: int, a cv generator instance, or None (default 10)
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    Attributes
    ----------
    `alpha_`: float
         Best alpha found by cross-validation

    `coef_`: array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    `intercept_`: array, shape = [n_classes-1]
         Intercept (a.k.a. bias) added to the decision function.
         It is available only when parameter intercept is set to True.

    `scores_`: 2d array of shape (n_alphas, n_folds)
        Scores (misclassification) for each alpha, and on each fold

    """

    path_scores_func = "squared_loss_path_scores"

    def __init__(self, alphas=None, l1_ratio=.5, mask=None, max_iter=1000,
                 tol=1e-4, memory=Memory(None), copy_data=True,
                 verbose=0, n_jobs=1, callback=None, n_alphas=10, eps=1e-3,
                 fit_intercept=True, cv=10, debias=False, normalize=True,
                 backtracking=False, standardize=True, alpha_min=1e-6,
                 bagging=True):
        super(_BaseRegressorCV, self).__init__(
            l1_ratio=l1_ratio, mask=mask, max_iter=max_iter, tol=tol,
            memory=memory, copy_data=copy_data, verbose=verbose,
            fit_intercept=fit_intercept, backtracking=backtracking,
            standardize=standardize, normalize=normalize)
        self.n_jobs = n_jobs
        self.cv = cv
        self.n_alphas = n_alphas
        self.debias = debias
        self.eps = eps
        self.alphas = alphas
        self.alpha_min = alpha_min
        self.bagging = bagging

    def fit(self, X, y):
        return _BaseCV.fit(self, X, y)


class _BaseClassifierCV(_BaseClassifier, _BaseCV):
    """
    Parameters
    ----------
    alphas: list of floats, optional (default None)
        Choices for the constant that scales the overall regularization term.
        This parameter is mutually exclusive with the `n_alphas` parameter.

    n_alphas: int, optional (default 10).
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    eps: float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    alpha_min: float, optional (default 1e-6)
        Minimum value of alpha to consider. This is mutually exclusive with the
        `eps` parameter.

    l1_ratio: float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV, etc., penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.

    mask: multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    screeing_percentile: float in the interval [0, 100]; Optional (default 10)
        Percentile value for ANOVA univariate feature selection. A value of
        100 means "keep all features".

    max_iter: int
        Defines the iterations for the solver. Defaults to 1000

    tol: float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose: int, optional (default 0)
        Verbosity level.

    backtracking: bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

    callback: callable(dict) -> bool
        Function called at the end of every energy descendent iteration of the
        solver. If it returns True, the loop breaks.

    n_jobs: int, optional (default 1)
        Number of jobs to use for One-vs-All classification.

    cv: int, a cv generator instance, or None (default 10)
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    Attributes
    ----------
    `alpha_`: float
         Best alpha found by cross-validation

    `coef_`: array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    `intercept_`: array, shape = [n_classes-1]
         Intercept (a.k.a. bias) added to the decision function.
         It is available only when parameter intercept is set to True.

    `scores_`: 2d array of shape (n_alphas, n_folds)
        Scores (misclassification) for each alpha, and on each fold

    Notes
    -----
    XXX For now, CV only works for two-class problems!

    """

    path_scores_func = "logistic_path_scores"

    def __init__(self, alphas=None, l1_ratio=.5, mask=None, max_iter=1000,
                 tol=1e-4, memory=Memory(None), copy_data=True, eps=1e-3,
                 verbose=0, n_jobs=1, callback=None, n_alphas=10,
                 alpha_min=1e-6, fit_intercept=True, cv=10, backtracking=False,
                 bagging=True):
        super(_BaseClassifierCV, self).__init__(
            l1_ratio=l1_ratio, mask=mask, max_iter=max_iter, tol=tol,
            memory=memory, copy_data=copy_data, verbose=verbose,
            fit_intercept=fit_intercept, backtracking=backtracking)
        self.n_jobs = n_jobs
        self.cv = cv
        self.n_alphas = n_alphas
        self.eps = eps
        self.alphas = alphas
        self.alpha_min = alpha_min
        self.bagging = bagging

    def _pre_fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self._rescale_alpha(X)

        # encode target classes as -1 and 1
        self._enc = LabelBinarizer(pos_label=1, neg_label=-1)
        y = self._enc.fit_transform(y)
        self.classes_ = self._enc.classes_
        self.n_classes_ = len(self.classes_)

        if self.mask is not None:
            self.n_features_ = np.prod(self.mask.shape)
        else:
            self.n_features_ = X.shape[1]

        return X, y

    def _set_coef_and_intercept(self, w):
        self.w_ = np.array(w)
        if self.w_.ndim == 1:
            self.w_ = self.w_[np.newaxis, :]
        self.coef_ = self.w_[:, :-1]
        self.intercept_ = self.w_[:, -1]

    def fit(self, X, y):
        return _BaseCV.fit(self, X, y)


class SmoothLassoClassifierCV(_BaseClassifierCV, SmoothLassoClassifier):
    """
    Cross-valided Smooth-Lasso logistic regression model with L1 + L2
    regularization.

    w = argmin - (1 / n_samples) * log(sigmoid(y * w.T * X)) +
          w      alpha * (l1_ratio ||w||_1 (1-l1_ratio) * .5 * <Gw, Gw>)
    where G is the spatial gradient operator

    Parameters
    ----------
    alphas: list of floats, optional (default None)
        Choices for the constant that scales the overall regularization term.
        This parameter is mutually exclusive with the `n_alphas` parameter.

    n_alphas: int, optional (default 10).
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    eps: float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    alpha_min: float, optional (default 1e-6)
        Minimum value of alpha to consider. This is mutually exclusive with the
        `eps` parameter.

    l1_ratio: float
        Constant that mixes L1 and G2 penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.
        Defaults to 0.5.

    mask: multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    max_iter: int
        Defines the iterations for the solver. Defaults to 1000

    screeing_percentile: float in the interval [0, 100]; Optional (default 10)
        Percentile value for ANOVA univariate feature selection. A value of
        100 means "keep all features".

    tol: float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose: int, optional (default 0)
        Verbosity level.

    backtracking: bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

    callback: callable(dict) -> bool
        Function called at the end of every energy descendent iteration of the
        solver. If it returns True, the loop breaks.

    n_jobs: int, optional (default 1)
        Number of jobs to use for One-vs-All classification.

    cv: int, a cv generator instance, or None (default 10)
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    Attributes
    ----------
    `alpha_`: float
         Best alpha found by cross-validation

    `coef_`: array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    `intercept_`: array, shape = [n_classes-1]
         Intercept (a.k.a. bias) added to the decision function.
         It is available only when parameter intercept is set to True.

    `scores_`: 2d array of shape (n_alphas, n_folds)
        Scores (misclassification) for each alpha, and on each fold

    Notes
    -----
    XXX For now, CV only works for two-class problems!

    """

    def __init__(self, alphas=None, l1_ratio=.5, mask=None, max_iter=1000,
                 tol=1e-4, memory=Memory(None), copy_data=True, eps=1e-3,
                 verbose=0, n_jobs=1, callback=None, n_alphas=10,
                 alpha_min=1e-6, fit_intercept=True, cv=10, backtracking=False,
                 bagging=True, screening_percentile=10.):
        super(SmoothLassoClassifierCV, self).__init__(
            l1_ratio=l1_ratio, mask=mask, max_iter=max_iter,
            tol=tol, memory=memory, copy_data=copy_data, verbose=verbose,
            fit_intercept=fit_intercept, backtracking=backtracking)
        self.n_jobs = n_jobs
        self.cv = cv
        self.n_alphas = n_alphas
        self.eps = eps
        self.alphas = alphas
        self.alpha_min = alpha_min
        self.bagging = bagging
        self.screening_percentile = screening_percentile

    def fit(self, X, y):
        """Fit is on grid of alphas and best alpha estimated by
        cross-validation.

        """

        return _BaseClassifierCV.fit(self, X, y)


class SmoothLassoRegressorCV(_BaseRegressorCV, SmoothLassoRegressor):
    """
    Cross-valided Smooth-Lasso logistic regression model with L1 + L2
    regularization.

    w = argmin  n_samples^(-1) * || y - X w ||^2 + alpha * l1_ratio ||w||_1
           w      + alpha * (1 - l1_ratio) * ||Gw||^2_2

    Parameters
    ----------
    alphas: list of floats, optional (default None)
        Choices for the constant that scales the overall regularization term.

    n_alphas: int, optional (default 10).
        Generate this number of alphas per regularization path.

    eps: float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    alpha_min: float, optional (default 1e-6)
        Minimum value of alpha to consider. This is mutually exclusive with the
        `eps` parameter.

    l1_ratio: float
        Constant that mixes L1 and G2 penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.
        Defaults to 0.5.

    mask: multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    screeing_percentile: float in the interval [0, 100]; Optional (default 10)
        Percentile value for ANOVA univariate feature selection. A value of
        100 means "keep all features".

    max_iter: int
        Defines the iterations for the solver. Defaults to 1000

    tol: float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose: int, optional (default 0)
        Verbosity level.

    backtracking: bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

    callback: callable(dict) -> bool
        Function called at the end of every energy descendent iteration of the
        solver. If it returns True, the loop breaks.

    n_jobs: int, optional (default 1)
        Number of jobs to use for One-vs-All classification.

    cv: int, a cv generator instance, or None (default 10)
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    Attributes
    ----------
    `alpha_`: float
         Best alpha found by cross-validation

    `coef_`: array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    `intercept_`: array, shape = [n_classes-1]
         Intercept (a.k.a. bias) added to the decision function.
         It is available only when parameter intercept is set to True.

    `scores_`: 2d array of shape (n_alphas, n_folds)
        Scores (misclassification) for each alpha, and on each fold

    """

    def __init__(self, alphas=None, l1_ratio=.5, mask=None, max_iter=1000,
                 tol=1e-4, memory=Memory(None), copy_data=True, eps=1e-3,
                 verbose=0, n_jobs=1, callback=None, debias=False,
                 fit_intercept=True, normalize=True, n_alphas=10,
                 standardize=True, cv=10, backtracking=False, alpha_min=1e-6,
                 bagging=True, screening_percentile=10.):
        super(SmoothLassoRegressorCV, self).__init__(
            self, l1_ratio=l1_ratio, mask=mask, max_iter=max_iter,
            tol=tol, memory=memory, copy_data=copy_data, verbose=verbose,
            fit_intercept=fit_intercept, backtracking=backtracking,
            normalize=normalize, standardize=standardize)
        self.n_jobs = n_jobs
        self.cv = cv
        self.debias = debias
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.alpha_min = alpha_min
        self.bagging = bagging
        self.screening_percentile = screening_percentile

    def fit(self, X, y):
        """Fit is on grid of alphas and best alpha estimated by
        cross-validation.

        """

        return _BaseRegressorCV.fit(self, X, y)


class TVl1ClassifierCV(_BaseClassifierCV, TVl1Classifier):
    """Cross-validated TV-l1 penalized logisitic regression.

    The underlying optimization problem is the following:

        w = argmin_w -(1 / n_samples) * log(sigmoid(y * w.T * X))
                 +  alpha * (l1_ratio * ||w||_1 + (1 - l1_ratio) * ||w||_TV)

    Parameters
    ----------
    alphas: list of floats, optional (default None)
        Choices for the constant that scales the overall regularization term.
        This parameter is mutually exclusive with the `n_alphas` parameter.

    n_alphas: int, optional (default 10).
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    eps: float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    alpha_min: float, optional (default 1e-6)
        Minimum value of alpha to consider. This is mutually exclusive with the
        `eps` parameter.

    l1_ratio: float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.

    mask: multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    screeing_percentile: float in the interval [0, 100]; Optional (default 10)
        Percentile value for ANOVA univariate feature selection. A value of
        100 means "keep all features".

    max_iter: int
        Defines the iterations for the solver. Defaults to 1000

    tol: float
        Defines the tolerance for convergence. Defaults to 1e-4.

    prox_max_iter: int, optional (default 5000)
        Maximum number of iterations for inner FISTA loop in which
        the prox of TV is approximated.

    verbose: int, optional (default 0)
        Verbosity level.

    backtracking: bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

    callback: callable(dict) -> bool
        Function called at the end of every energy descendent iteration of the
        solver. If it returns True, the loop breaks.

    n_jobs: int, optional (default 1)
        Number of jobs to use for One-vs-All classification.

    cv: int, a cv generator instance, or None (default 10)
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    Attributes
    ----------
    `alpha_`: float
         Best alpha found by cross-validation

    `coef_`: array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    `intercept_`: array, shape = [n_classes-1]
         Intercept (a.k.a. bias) added to the decision function.
         It is available only when parameter intercept is set to True.

    `scores_`: 2d array of shape (n_alphas, n_folds)
        Scores (misclassification) for each alpha, and on each fold

    Notes
    -----
    XXX For now, CV only works for two-class problems!

    """

    def __init__(self, alphas=None, l1_ratio=.5, mask=None, max_iter=1000,
                 tol=1e-4, memory=Memory(None), copy_data=True, eps=1e-3,
                 verbose=0, n_jobs=1, callback=None, n_alphas=10,
                 fit_intercept=True, cv=10, backtracking=False,
                 alpha_min=1e-6, bagging=True, screening_percentile=10.):
        super(TVl1ClassifierCV, self).__init__(
            l1_ratio=l1_ratio, mask=mask, max_iter=max_iter, tol=tol,
            memory=memory, copy_data=copy_data, verbose=verbose,
            fit_intercept=fit_intercept, backtracking=backtracking)
        self.n_jobs = n_jobs
        self.cv = cv
        self.n_alphas = n_alphas
        self.eps = eps
        self.alphas = alphas
        self.alpha_min = alpha_min
        self.bagging = bagging
        self.screening_percentile = screening_percentile

    def fit(self, X, y):
        return _BaseClassifierCV.fit(self, X, y)


class TVl1RegressorCV(_BaseRegressorCV, TVl1Regressor):
    """Cross-validated TV-l1 penalized logisitic regression.

    The underlying optimization problem is the following:

        w = argmin_w -(1 / n_samples) * log(sigmoid(y * w.T * X))
                 +  alpha * (l1_ratio * ||w||_1 + (1 - l1_ratio) * ||w||_TV)

    Parameters
    ----------
    alphas: list of floats, optional (default None)
        Choices for the constant that scales the overall regularization term.
        This parameter is mutually exclusive with the `n_alphas` parameter.

    n_alphas: int, optional (default 10).
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    eps: float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    alpha_min: float, optional (default 1e-6)
        Minimum value of alpha to consider. This is mutually exclusive with the
        `eps` parameter.

    l1_ratio: float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.

    mask: multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    screeing_percentile: float in the interval [0, 100]; Optional (default 10)
        Percentile value for ANOVA univariate feature selection. A value of
        100 means "keep all features".

    max_iter: int
        Defines the iterations for the solver. Defaults to 1000

    tol: float
        Defines the tolerance for convergence. Defaults to 1e-4.

    prox_max_iter: int, optional (default 5000)
        Maximum number of iterations for inner FISTA loop in which
        the prox of TV is approximated.

    verbose: int, optional (default 0)
        Verbosity level.

    backtracking: bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

    callback: callable(dict) -> bool
        Function called at the end of every energy descendent iteration of the
        solver. If it returns True, the loop breaks.

    n_jobs: int, optional (default 1)
        Number of jobs to use for One-vs-All classification.

    cv: int, a cv generator instance, or None (default 10)
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    Attributes
    ----------
    `alpha_`: float
         Best alpha found by cross-validation

    `coef_`: array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    `intercept_`: array, shape = [n_classes-1]
         Intercept (a.k.a. bias) added to the decision function.
         It is available only when parameter intercept is set to True.

    `scores_`: 2d array of shape (n_alphas, n_folds)
        Scores (misclassification) for each alpha, and on each fold

    """

    def __init__(self, alphas=None, l1_ratio=.5, mask=None, max_iter=1000,
                 tol=1e-4, memory=Memory(None), copy_data=True,
                 verbose=0, n_jobs=1, callback=None, n_alphas=10, eps=1e-3,
                 fit_intercept=True, cv=10, debias=False, normalize=True,
                 backtracking=False, standardize=True, alpha_min=1e-6,
                 bagging=True, screening_percentile=10.):
        super(TVl1RegressorCV, self).__init__(
            l1_ratio=l1_ratio, mask=mask, max_iter=max_iter, tol=tol,
            memory=memory, copy_data=copy_data, verbose=verbose,
            fit_intercept=fit_intercept, backtracking=backtracking,
            standardize=standardize, normalize=normalize)
        self.n_jobs = n_jobs
        self.cv = cv
        self.n_alphas = n_alphas
        self.debias = debias
        self.eps = eps
        self.alphas = alphas
        self.alpha_min = alpha_min
        self.bagging = bagging
        self.screening_percentile = screening_percentile

    def fit(self, X, y):
        return _BaseRegressorCV.fit(self, X, y)


def plot_cv_scores(cvobj, i_best_alpha=None, title=None, ylabel=None,
                   errorbars=True):
    """Plots CV scores against regularization parameter values (alpha grid).

    `cvobj` can be a CV object (_BaseCV instance) or a pair (alphas, scores).

    """

    import pylab as pl

    if isinstance(cvobj, _BaseCV):
        alphas = cvobj.alphas_
        scores = cvobj.scores_
        i_best_alpha = cvobj.i_alpha_
        if title is None:
            title = cvobj.short_name
        if ylabel is None:
            if is_classifier(cvobj):
                ylabel = "1 - AUC"
            else:
                ylabel = "Mean-Squared Error"

        if is_classifier(cvobj) and cvobj.n_classes_ > 2:
            for c in xrange(cvobj.n_classes_):
                i = None if i_best_alpha is None else i_best_alpha[c]
                plot_cv_scores((alphas, scores[c]), i_best_alpha=i,
                               title="class %i vrs rest: %s" % (c, title),
                               ylabel=ylabel, errorbars=errorbars)
            return
    else:
        assert hasattr(cvobj, "__iter__")
        alphas, scores = cvobj
        if ylabel is None:
            ylabel = "Error (Out-of-Bag)"

    pl.figure()
    lalphas_ = -np.log10(alphas)
    if errorbars:
        means = np.mean(scores, axis=-1)
        stds = np.std(scores, axis=-1)
        pl.errorbar(lalphas_, means, yerr=stds)
        if i_best_alpha is not None:
            pl.axhline(means[i_best_alpha], linestyle="--")
    else:
        pl.plot(lalphas_, scores, "s-")
    if not i_best_alpha is None:
        pl.axvline(lalphas_[i_best_alpha], linestyle="--",
                   label="Best alpha: %.1e" % alphas[i_best_alpha])
    pl.ylabel(ylabel)
    pl.xlabel("-Log10(alpha)")
    pl.legend(loc="best")
    if title:
        pl.title(title)
