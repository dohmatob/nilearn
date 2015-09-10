"""
Mathematical operations working on Niimg-like objects like a
(3+n)-dimensional block of data, and an affine.
"""
from .resampling import resample_img, reorder_img
from .image import high_variance_confounds, smooth_img, crop_img, \
    mean_img, swap_img_hemispheres, index_img, iter_img
from .image import new_img_like  # imported this way to avoid circular imports
from .._utils.niimg_conversions import concat_niimgs as concat_imgs
from .resampling import coord_transform, get_bounds


__all__ = ['resample_img', 'high_variance_confounds', 'smooth_img',
           'crop_img', 'mean_img', 'reorder_img', 'swap_img_hemispheres',
           'concat_imgs', 'index_img', 'iter_img', 'new_img_like',
           "coord_transform", "get_bounds"]
