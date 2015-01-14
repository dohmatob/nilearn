"""
Decoding with SpaceNet on Haxby dataset
==========================================

Here is a simple example of decoding with a SpaceNet prior (i.e S-LASSO,
TV-l1, etc.), reproducing the Haxby 2001 study on a face vs house
discrimination task.
"""
# author: DOHMATOB Elvis Dopgima,
#         VAROQUAUX Gael


### Load haxby dataset ########################################################
import nibabel
from nilearn.datasets import fetch_haxby
data_files = fetch_haxby()

### Load Target labels ########################################################
import numpy as np
labels = np.recfromcsv(data_files.session_target[0], delimiter=" ")


### split data into train and test samples ####################################
target = labels['labels']


### Loop over Smooth-LASSO and TV-L1 penalties ###############################
from nilearn.decoding import SpaceNetClassifier
penalties = ['smooth-lasso', 'tv-l1']
decoders = {}
accuracies = {}
n_jobs = 36
categories = ["face-house", "shoe-house", "face-cat", "cat-house"]
accuracies["SVC"] = {}
for category in categories:
    a, b = category.split("-")
    condition_mask = np.logical_or(target == a, target == b)
    condition_mask_train = np.logical_and(condition_mask,
                                          labels['chunks'] <= 6)
    condition_mask_test = np.logical_and(condition_mask, labels['chunks'] > 6)

    # make X (design matrix) and y (response variable)
    niimgs  = nibabel.load(data_files.func[0])
    data, affine = niimgs.get_data(), niimgs.get_affine()
    X_train = nibabel.Nifti1Image(data[:, :, :, condition_mask_train], affine)
    y_train = target[condition_mask_train]
    X_test = nibabel.Nifti1Image(data[:, :, :, condition_mask_test], affine)
    y_test = target[condition_mask_test]

    for penalty in penalties:
        if not penalty in accuracies:
            accuracies[penalty] = {}
        decoder = SpaceNetClassifier(memory="cache", penalty=penalty,
                                     verbose=2, n_jobs=n_jobs,
                                     l1_ratios=[.25, .5, .75])
        decoder.fit(X_train, y_train)
        y_pred = decoder.predict(X_test)
        accuracies[penalty][category] = (y_pred == y_test).mean() * 100.
        decoders[penalty] = decoder

    # construct mask for super-support of all decoders
    support_mask = decoder.mask_img_.get_data().astype(np.bool)
    support = np.zeros(support_mask.sum())
    for penalty in penalties:
        decoder = decoders[penalty]
        support[decoder.coef_[0] != 0.] = 1
    support_mask[support_mask] = support
    support_mask = nibabel.Nifti1Image(support_mask.astype(np.float),
                                       decoder.mask_img_.get_affine())

    # fit SVC
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    from sklearn.base import clone
    high = decoders.values()[0].alpha_grids_.max()
    low = high * decoder.eps
    gammas = np.reciprocal(np.logspace(np.log10(high), np.log10(low),
                                       decoders.values()[0].n_alphas))
    masker = clone(decoder.masker_)
    masker.set_params(mask_img=support_mask)
    X_train = masker.fit_transform(X_train)
    svc = SVC(kernel='linear')
    decoder = GridSearchCV(estimator=svc, param_grid=dict(gamma=gammas),
                           n_jobs=n_jobs)
    decoder.fit(X_train, y_train)
    y_pred = decoder.predict(masker.transform(X_test))
    decoder.coef_img_ = masker.inverse_transform(decoder.best_estimator_.coef_)
    accuracies["SVC"][category] = (y_pred == y_test).mean() * 100.
    decoders["SVC"] = decoder


### Visualization #############################################################
import matplotlib.pyplot as plt
from nilearn.image import mean_img
from nilearn.plotting import plot_stat_map
background_img = mean_img(data_files.func[0])
background_img.to_filename("haxby_background_img.nii.gz")
print "Results"
print "=" * 80

# for category in categories:
#     print category
#     for penalty, decoder in sorted(decoders.items()):
#         coef_img = decoder.coef_img_
#         coef_img_filename = 'haxby_%s_weights.nii.gz' % penalty
#         slicer = plot_stat_map(coef_img, background_img,
#                                display_mode="xyz"[1:3],
#                                cut_coords=(20, -34, -16)[1:3])
#         slicer.add_contours(support_mask)
#         plt.savefig(coef_img_filename.split(".")[0] + ".png")
#         coef_img.to_filename(coef_img_filename)
#         print "\t- %s %s" % (penalty, '-' * 60)
#         print "\tNumber of train samples : %i" % condition_mask_train.sum()
#         print "\tNumber of test samples  : %i" % condition_mask_test.sum()
#         print "\tClassification accuracy : %g%%" % (
#             accuracies[penalty][category])
#         print "_" * 80
#     print


###############################################################################
# make a rudimentary diagram
plt.figure()
tick_position = np.arange(len(categories))
plt.xticks(tick_position, categories, rotation=45)

for color, (penalty, decoder) in zip(
        ['b', 'c', 'm', 'g', 'y', 'k', '.5', 'r', '#ffaaaa'],
        decoders.iteritems()):
    score_means = [accuracies[penalty][category].mean()
                   for category in categories]
    plt.bar(tick_position, score_means, label=penalty,
            width=.11, color=color)
    tick_position = tick_position + .09

plt.show()
