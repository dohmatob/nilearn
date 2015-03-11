import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel
from sklearn.cross_validation import LeaveOneLabelOut
from nilearn.image import index_img
from nilearn.datasets import fetch_haxby
from nilearn.decoding import SpaceNetClassifier
from nilearn.image import mean_img
from nilearn.plotting import plot_stat_map

data_files = fetch_haxby()
behavior = np.loadtxt(data_files.session_target[0], dtype=np.str, skiprows=1)
targets, session = behavior.T
session = session.astype(np.int)
conditions = {}
conditions['face_house'] = np.logical_or(targets == 'face',
                                         targets == 'house')
conditions['tools_scramble'] = np.logical_or(targets == 'scissors',
                                             targets == 'scrambledpix')
niimgs  = nibabel.load(data_files.func[0])
background_img = mean_img(data_files.func[0])
n_jobs = int(os.environ.get("N_JOBS", 1))
penalty = "smooth-lasso"
for condition_name, condition_mask in sorted(conditions.items()):
    labels = session[condition_mask]
    labels_train = labels[labels < 6]
    cv = LeaveOneLabelOut(labels=labels_train)
    condition_mask_train = (labels < 6)
    condition_mask_test = (labels >= 6)
    X_train = index_img(niimgs, condition_mask_train)
    X_test = index_img(niimgs, condition_mask_test)
    y_train = targets[condition_mask_train]
    y_test = targets[condition_mask_test]

    for early_stopping_tol in [-1.e-4, 1e2]:
        decoder = SpaceNetClassifier(memory="cache", penalty=penalty,
                                     verbose=2, n_jobs=n_jobs, cv=cv,
                                     early_stopping_tol=early_stopping_tol)
        decoder.fit(X_train, y_train)
        y_pred = decoder.predict(X_test)
        accuracy = (y_pred == y_test).mean() * 100.
        print "Results"
        print "=" * 80
        coef_img = decoder.coef_img_
        plot_stat_map(coef_img, background_img,
                      title="%s (early-stopping tol = %g): accuracy %g%%" % (
                          penalty, early_stopping_tol, accuracy),
                      cut_coords=(20, -34, -16))
        coef_img.to_filename('haxby_%s_es_tol=%g_weights.nii' % (
                penalty, early_stopping_tol))
        print "- %s (early-stopping tol=%g) %s" % (
            penalty, early_stopping_tol, '-' * 60)
        print "Number of train samples : %i" % condition_mask_train.sum()
        print "Number of test samples  : %i" % condition_mask_test.sum()
        print "Classification accuracy : %g%%" % accuracy
        print "_" * 80
plt.show()
