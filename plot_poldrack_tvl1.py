"""
TV-l1 regression example on Poldrack's Mixed Gambles dataset.

"""
# Author: DOHMATOB Elvis Dopgima

import numpy as np
from sklearn.externals.joblib import Memory

memory = Memory("cache")

### Load data #################################################################
import nibabel
from nilearn.decoding.sparse_models.common import _unmask
from load_poldrack import load_poldrack
X, y, mask, affine = load_poldrack(full_brain=1,
                                   memory=memory)
mask_niimg = nibabel.Nifti1Image(mask.astype(np.float), affine)
anat_niimg = nibabel.Nifti1Image(_unmask(X.mean(axis=0), mask.ravel()
                                   ).reshape(mask.shape), affine)

### Fit model (by cross-validation) ###########################################
import os
from nilearn.decoding.sparse_models.cv import TVl1RegressorCV
n_jobs = int(os.environ.get("N_JOBS", 1))
tvl1cv = TVl1RegressorCV(n_alphas=10, tol=1e-4, max_iter=1000,
                       l1_ratio=.75, verbose=1, memory=memory,
                       mask=mask, n_jobs=n_jobs).fit(X, y)

# Retrieve the TV-l1 discriminating weights
coef_ = _unmask(tvl1cv.coef_, mask.ravel()).reshape(mask.shape)
coef_niimg = nibabel.Nifti1Image(coef_, affine)
coef_niimg.to_filename("coef.nii.gz")

### Visualization #############################################################
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map

# weights
slicer = plot_stat_map(coef_niimg, anat_niimg, title="TV-l1 weights",
                       slicer="y", cut_coords=range(10, 30, 2))
slicer.contour_map(mask_niimg, levels=[.5], colors='r')

# CV
from nilearn.decoding.sparse_models.cv import plot_cv_scores
plot_cv_scores(tvl1cv, errorbars=False)
plt.show()
