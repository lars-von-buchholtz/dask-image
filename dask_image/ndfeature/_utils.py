import numpy as np
from functools import wraps

from skimage.feature._hessian_det_appx import _hessian_matrix_det

import dask.array as da


def output2ndarray(func):
    """decorator to convert output to numpy array

    this is needed for the _hessian_matrix_det function that outputs
    a memoryview"""

    @wraps(func)
    def wrapped(*args,**kwargs):
        return np.asarray(func(*args,**kwargs))
    return wrapped

# modify hessian determinant approximation function to output numpy array
_hessian_matrix_det = output2ndarray(_hessian_matrix_det)


def _daskarray_to_float(image):
    """helper function to convert dask integer arrays to float"""

    image = da.asarray(image) # make sure image is a dask array

    #get array type, kind and itemsize
    dtypeobj_in = image.dtype
    dtype_in = dtypeobj_in.type
    kind_in = dtypeobj_in.kind
    itemsize_in = dtypeobj_in.itemsize

    # if input was already float, just return it
    if kind_in == "f":
        return image

    # get min and max values captured by the input format
    if kind_in in "ui":
        imin_in = np.iinfo(dtype_in).min
        imax_in = np.iinfo(dtype_in).max

    # get the smallest float format that captures the size of the input
    dtype_out = next(
        dt
        for dt in [np.float16, np.float32, np.float64]
        if np.dtype(dt).itemsize >= itemsize_in
    )

    #convert image to output format (float)
    image = image.astype(dtype_out)

    if kind_in == "u":
        # for unsigned integers, distribute between 0.0 and 1.0
        image *= 1.0 / imax_in

    else:
        # for signed integers , distribute between -1.0 and 1.0
        image += 0.5
        image *= 2.0 / (imax_in - imin_in)

    return image


def _exclude_border(mask, exclude_border):
    """
    Helper function to remove peaks near the borders
    """

    # if a scalar is provided, expand it to the mask image dimension
    exclude_border = (exclude_border,) * mask.ndim if np.isscalar(
        exclude_border) \
        else exclude_border

    # if the wrong size sequence is provided, raise an error
    if len(exclude_border) != mask.ndim:
        raise ValueError("exclude_border has to be boolean, int scalar\
         or a sequence of length: number of dimensions of the image")

    # build a filter for the border by zero-padding a center dask array of ones
    center_dim = tuple(np.subtract(mask.shape, [2 * i for i in exclude_border]))
    borders = tuple([(i,) * 2 for i in exclude_border])
    border_filter = da.pad(da.ones(center_dim), borders, 'constant')

    assert border_filter.shape == mask.shape

    # filter the input mask by the border filter
    return mask * border_filter


def _get_high_intensity_peaks(image, mask, num_peaks):
    """
    Helper function to return the num_peaks highest intensity peak coordinates.

    Adapted to dask from skimage.feature.peak._get_high_intensity_peaks
    """
    # get coordinates of peaks
    coord = tuple([c.compute() for c in da.nonzero(mask)])

    # sort by peak intensity
    intensities = image.vindex[coord]
    idx_maxsort = np.argsort(intensities)
    coord = np.vstack(coord).T[idx_maxsort]

    # select num_peaks peaks
    if coord.shape[0] > num_peaks:
        coord = coord[-num_peaks:]

    # return highest peak first
    return coord[::-1]
