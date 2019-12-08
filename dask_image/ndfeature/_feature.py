import numpy as np
import math


import dask.array as da

from skimage.feature.peak import peak_local_max as ski_peak_local_max
from skimage.feature._hessian_det_appx import _hessian_matrix_det

from ..ndfilters._gaussian import gaussian_laplace, gaussian_filter
from ._skimage_utils import _exclude_border, _prune_blobs
from functools import wraps
from skimage.transform import integral_image

def output2ndarray(func):
    @wraps(func)
    def wrapped(*args,**kwargs):
        return np.asarray(func(*args,**kwargs))
    return wrapped

_hessian_matrix_det = output2ndarray(_hessian_matrix_det)

def _daskarray_to_float(image):

    image = da.asarray(image)
    dtypeobj_in = image.dtype
    dtype_in = dtypeobj_in.type
    kind_in = dtypeobj_in.kind
    itemsize_in = dtypeobj_in.itemsize

    if kind_in == "f":
        return image

    if kind_in in "ui":
        imin_in = np.iinfo(dtype_in).min
        imax_in = np.iinfo(dtype_in).max

    dtype_out = next(
        dt
        for dt in [np.float16, np.float32, np.float64]
        if np.dtype(dt).itemsize >= itemsize_in
    )
    image = image.astype(dtype_out)

    if kind_in == "u":
        # for unsigned integers, distribute between 0.0 and 1.0
        image *= 1.0 / imax_in

    else:
        # for signed integers , distribute between
        image += 0.5
        image *= 2.0 / (imax_in - imin_in)

    return image


def _get_high_intensity_peaks(image, mask, num_peaks):
    """
    Return the highest intensity peak coordinates.

    Adapted from skimage.feature.peak._get_high_intensity_peaks
    """
    # get coordinates of peaks
    coord = tuple([c.compute() for c in da.nonzero(mask)])
    # select num_peaks peaks
    if len(coord[0]) > num_peaks:
        intensities = image[coord]
        idx_maxsort = np.argsort(intensities)
        coord = np.transpose(coord)[idx_maxsort][-num_peaks:]
    else:
        coord = np.column_stack(coord)
    # Highest peak first
    return coord[::-1]




def peak_local_max(
    image,
    min_distance=1,
    threshold_abs=None,
    threshold_rel=None,
    exclude_border=True,
    indices=True,
    num_peaks=np.inf,
    footprint=None,
):

    """Find peaks in a dask image as coordinate list or boolean mask.
    Peaks are the local maxima in a region of `2 * min_distance + 1`
    (i.e. peaks are separated by at least `min_distance`).
    If there are multiple local maxima with identical pixel intensities
    inside the region defined with `min_distance`,
    the coordinates of all such pixels are returned.
    If both `threshold_abs` and `threshold_rel` are provided, the maximum
    of the two is chosen as the minimum intensity threshold of peaks.
    Parameters
    ----------
    image : n-dimensional dask array
        Input image.
    min_distance : int, optional
        Minimum number of pixels separating peaks in a region of `2 *
        min_distance + 1` (i.e. peaks are separated by at least
        `min_distance`).
        To find the maximum number of peaks, use `min_distance=1`.
    threshold_abs : float, optional
        Minimum intensity of peaks. By default, the absolute threshold is
        the minimum intensity of the image.
    threshold_rel : float, optional
        Minimum intensity of peaks, calculated as `max(image) * threshold_rel`.
    exclude_border : int or bool, optional
        If nonzero int, `exclude_border` excludes peaks from
        within `exclude_border`-pixels of the border of the image.
        If True, takes the `min_distance` parameter as value.
        If zero or False, peaks are identified regardless of their
        distance from the border.
    indices : bool, optional
        If True, the output will be an array representing peak
        coordinates.  If False, the output will be a boolean array shaped as
        `image.shape` with peaks present at True elements.
    num_peaks : int, optional
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` peaks based on highest peak intensity.
    footprint : ndarray of bools, optional
        If provided, `footprint == 1` represents the local region within which
        to search for peaks at every point in `image`.  Overrides
        `min_distance` (also for `exclude_border`).
    Returns
    -------
    output : ndarray or dask array of bools
        * If `indices = True`  : (row, column, ...) coordinates of peaks as
         ndarray.
        * If `indices = False` : Boolean dask array shaped like `image`,
        with peaks
          represented by True values.
    Notes
    -----
    The peak local maximum function returns the coordinates of local peaks
    (maxima) in an image. A maximum filter is used for finding local maxima.
    This operation dilates the original image. After comparison of the dilated
    and original image, this function returns the coordinates or a mask of the
    peaks where the dilated image equals the original image.

    Examples
    --------
    >>> img1 = np.zeros((7, 7))
    >>> img1[3, 4] = 1
    >>> img1[3, 2] = 1.5
    >>> img1
    array([[0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 1.5, 0. , 1. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ]])
    >>> peak_local_max(img1, min_distance=1)
    array([[3, 4],
           [3, 2]])
    >>> peak_local_max(img1, min_distance=2)
    array([[3, 2]])
    >>> img2 = np.zeros((20, 20, 20))
    >>> img2[10, 10, 10] = 1
    >>> peak_local_max(img2, exclude_border=0)
    array([[10, 10, 10]])
    """

    # calculate depth based on min_distance and/or footprint
    if not (min_distance or footprint):
        raise ValueError("Either min_distance or footprint must be specified")

    if type(footprint) is np.ndarray:
        depth = footprint.shape
    else:
        depth = 2 * min_distance + 1

    # map_overlap plm without border exclude, labels, indices=False
    mask = image.map_overlap(
        ski_peak_local_max,
        depth=depth,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
        exclude_border=False,
        indices=False,
        num_peaks=np.inf,
        footprint=footprint,
        labels=None,
        num_peaks_per_label=np.inf,
    )

    # if exclude_borders filter out points near borders
    if type(exclude_border) == bool:
        exclude_border = min_distance if exclude_border else 0

    if exclude_border:
        mask = _exclude_border(mask, footprint, exclude_border)

    # Select highest intensities (num_peaks)
    coordinates = _get_high_intensity_peaks(image, mask, num_peaks)

    if indices is True:
        return coordinates
    else:
        out = da.zeros_like(image, dtype=np.bool)
        nd_indices = tuple(coordinates.T)
        out[nd_indices] = True
        return out


def blob_common(blob_func):
    """Decorator for functionality that is conserved between blob_log and
    blob_dog"""

    @wraps(blob_func)
    def wrapped_func(
        image,
        min_sigma=1,
        max_sigma=50,
        num_sigma=10,
        sigma_ratio=1.6,
        threshold=0.2,
        overlap=0.5,
        log_scale=False,
        exclude_border=False,
    ):

        scalar_sigma = (
            True if np.isscalar(max_sigma) and np.isscalar(min_sigma)
            else False
        )

        # Gaussian filter requires that sequence-type sigmas have same
        # dimensionality as image. This broadcasts scalar kernels
        if np.isscalar(max_sigma):
            max_sigma = np.full(image.ndim, max_sigma, dtype=float)
        if np.isscalar(min_sigma):
            min_sigma = np.full(image.ndim, min_sigma, dtype=float)

        # Convert sequence types to array
        min_sigma = np.asarray(min_sigma, dtype=float)
        max_sigma = np.asarray(max_sigma, dtype=float)

        #
        image = _daskarray_to_float(image)

        image_stack, sigma_list = blob_func(
            image=image,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            sigma_ratio=sigma_ratio,
            log_scale=log_scale,
        )

        chunk_shape = image_stack.chunks
        new_shape = chunk_shape[:-1] + ((sum(chunk_shape[-1]),),)
        image_stack = image_stack.rechunk(chunks=new_shape)

        local_maxima = peak_local_max(
            image_stack,
            threshold_abs=threshold,
            footprint=np.ones((3,) * image.ndim + (1,)),
            threshold_rel=0.0,
            exclude_border=exclude_border,
        )

        # Catch no peaks
        if local_maxima.size == 0:
            return np.empty((0, 3))

        # Convert local_maxima to float64
        lm = local_maxima.astype(np.float64)

        # translate final column of lm, which contains the index of the
        # sigma that produced the maximum intensity value, into the sigma
        sigmas_of_peaks = sigma_list[local_maxima[:, -1]]

        if scalar_sigma:
            # select one sigma column, keeping dimension
            sigmas_of_peaks = sigmas_of_peaks[:, 0:1]

        # Remove sigma index and replace with sigmas
        lm = np.hstack([lm[:, :-1], sigmas_of_peaks])

        sigma_dim = sigmas_of_peaks.shape[1]

        return _prune_blobs(lm, overlap, sigma_dim=sigma_dim)

    return wrapped_func


@blob_common
def blob_log(image, min_sigma, max_sigma, num_sigma, log_scale,sigma_ratio = 1.6):
    r"""Finds blobs in the given grayscale image.

    This implementation adapts the skimage.feature.blob_log function for Dask.
    Blobs are found using the Laplacian of Gaussian (LoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.
    Parameters
    ----------
    image : dask array
        Input grayscale image as n-dimensional dask array, blobs are assumed to
        be light on dark background (white on black).
    min_sigma : scalar or sequence of scalars, optional
        the minimum standard deviation for Gaussian kernel. Keep this low to
        detect smaller blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    max_sigma : scalar or sequence of scalars, optional
        The maximum standard deviation for Gaussian kernel. Keep this high to
        detect larger blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    num_sigma : int, optional
        The number of intermediate values of standard deviations to consider
        between `min_sigma` and `max_sigma`.
    threshold : float, optional.
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect blobs with less
        intensities.
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    log_scale : bool, optional
        If set, intermediate values of standard deviations are interpolated
        using a logarithmic scale to the base `10`. If not, linear
        interpolation is used.
    exclude_border : int or bool, optional
        If nonzero int, `exclude_border` excludes blobs from
        within `exclude_border`-pixels of the border of the image.
    Returns
    -------
    A : (m, image.ndim + sigma) ndarray
        A 2d array with each row representing m coordinate values for a
        m-dimensional image plus the sigma(s) used.
        When a single sigma is passed both for min_sigma and max_sigma, the
        last column is the standard deviation of the gaussian that detected the
        blob resulting in an m * (n + 1) array.
        When an anisotropic gaussian is used (sigmas per dimension), the
        detected sigma is returned for each dimension resulting in an m * 2n
        array.
    References
    ----------
    ..[1] https://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_
    of_Gaussian
    Examples
    --------
    >>> from skimage import data, exposure
    >>> from dask_image.ndfeature import blob_log
    >>> img = data.coins()
    >>> img = exposure.equalize_hist(img)  # improves detection
    >>> blob_log(img, threshold = .3)
    array([[266.        , 115.        ,  11.88888889],
           [263.        , 302.        ,  17.33333333],
           [263.        , 244.        ,  17.33333333],
           [260.        , 174.        ,  17.33333333],
           [198.        , 155.        ,  11.88888889],
           [198.        , 103.        ,  11.88888889],
           [197.        ,  44.        ,  11.88888889],
           [194.        , 276.        ,  17.33333333],
           [194.        , 213.        ,  17.33333333],
           [185.        , 344.        ,  17.33333333],
           [128.        , 154.        ,  11.88888889],
           [127.        , 102.        ,  11.88888889],
           [126.        , 208.        ,  11.88888889],
           [126.        ,  46.        ,  11.88888889],
           [124.        , 336.        ,  11.88888889],
           [121.        , 272.        ,  17.33333333],
           [113.        , 323.        ,   1.        ]])
    Notes
    -----
    The radius of each blob is approximately :math:`\sqrt{2}\sigma` for
    a 2-D image and :math:`\sqrt{3}\sigma` for a 3-D image.
    """

    if log_scale:
        # for anisotropic data, we use the "highest resolution/variance" axis
        standard_axis = np.argmax(min_sigma)
        start = np.log10(min_sigma[standard_axis])
        stop = np.log10(max_sigma[standard_axis])
        scale = np.logspace(start, stop, num_sigma)[:, np.newaxis]
        sigma_list = scale * min_sigma / np.max(min_sigma)
    else:
        scale = np.linspace(0, 1, num_sigma)[:, np.newaxis]
        sigma_list = scale * (max_sigma - min_sigma) + min_sigma

    # computing gaussian laplace using dask image functions
    # average s**2 provides scale invariance
    gl_images = [-gaussian_laplace(image, s) * np.mean(s) ** 2
                 for s in sigma_list]

    image_stack = da.stack(gl_images, axis=-1)

    return image_stack, sigma_list

@blob_common
def blob_dog(image, min_sigma, max_sigma, sigma_ratio, num_sigma=0,  log_scale=False):
    r"""Finds blobs in the given grayscale image.
    Blobs are found using the Difference of Gaussian (DoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.
    Parameters
    ----------
    image : 2D or 3D ndarray
        Input grayscale image, blobs are assumed to be light on dark
        background (white on black).
    min_sigma : scalar or sequence of scalars, optional
        The minimum standard deviation for Gaussian kernel. Keep this low to
        detect smaller blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    max_sigma : scalar or sequence of scalars, optional
        The maximum standard deviation for Gaussian kernel. Keep this high to
        detect larger blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    sigma_ratio : float, optional
        The ratio between the standard deviation of Gaussian Kernels used for
        computing the Difference of Gaussians
    threshold : float, optional.
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect blobs with less
        intensities.
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    exclude_border : int or bool, optional
        If nonzero int, `exclude_border` excludes blobs from
        within `exclude_border`-pixels of the border of the image.
    Returns
    -------
    A : (n, image.ndim + sigma) ndarray
        A 2d array with each row representing 2 coordinate values for a 2D
        image, and 3 coordinate values for a 3D image, plus the sigma(s) used.
        When a single sigma is passed, outputs are:
        ``(r, c, sigma)`` or ``(p, r, c, sigma)`` where ``(r, c)`` or
        ``(p, r, c)`` are coordinates of the blob and ``sigma`` is the standard
        deviation of the Gaussian kernel which detected the blob. When an
        anisotropic gaussian is used (sigmas per dimension), the detected sigma
        is returned for each dimension.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_difference_of_
    Gaussians_approach
    Examples
    --------
    >>> from skimage import data, feature
    >>> feature.blob_dog(data.coins(), threshold=.5, max_sigma=40)
    array([[267.      , 359.      ,  16.777216],
           [267.      , 115.      ,  10.48576 ],
           [263.      , 302.      ,  16.777216],
           [263.      , 245.      ,  16.777216],
           [261.      , 173.      ,  16.777216],
           [260.      ,  46.      ,  16.777216],
           [198.      , 155.      ,  10.48576 ],
           [196.      ,  43.      ,  10.48576 ],
           [195.      , 102.      ,  16.777216],
           [194.      , 277.      ,  16.777216],
           [193.      , 213.      ,  16.777216],
           [185.      , 347.      ,  16.777216],
           [128.      , 154.      ,  10.48576 ],
           [127.      , 102.      ,  10.48576 ],
           [125.      , 208.      ,  10.48576 ],
           [125.      ,  45.      ,  16.777216],
           [124.      , 337.      ,  10.48576 ],
           [120.      , 272.      ,  16.777216],
           [ 58.      , 100.      ,  10.48576 ],
           [ 54.      , 276.      ,  10.48576 ],
           [ 54.      ,  42.      ,  16.777216],
           [ 52.      , 216.      ,  16.777216],
           [ 52.      , 155.      ,  16.777216],
           [ 45.      , 336.      ,  16.777216]])
    Notes
    -----
    The radius of each blob is approximately :math:`\sqrt{2}\sigma` for
    a 2-D image and :math:`\sqrt{3}\sigma` for a 3-D image.
    """

    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    k = int(np.mean(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1))

    # a geometric progression of standard deviations for gaussian kernels
    sigma_list = np.array([min_sigma * (sigma_ratio ** i)
                           for i in range(k + 1)])

    gaussian_images = [gaussian_filter(image, s) for s in sigma_list]

    # computing difference between two successive Gaussian blurred images
    # multiplying with average standard deviation provides scale invariance
    dog_images = [
        (gaussian_images[i] - gaussian_images[i + 1]) * np.mean(sigma_list[i])
        for i in range(k)
    ]

    image_stack = np.stack(dog_images, axis=-1)

    return image_stack, sigma_list


def blob_doh(image, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.01,
             overlap=.5, log_scale=False):
    """Finds blobs in the given grayscale image.
    Blobs are found using the Determinant of Hessian method [1]_. For each blob
    found, the method returns its coordinates and the standard deviation
    of the Gaussian Kernel used for the Hessian matrix whose determinant
    detected the blob. Determinant of Hessians is approximated using [2]_.
    Parameters
    ----------
    image : 2D ndarray
        Input grayscale image.Blobs can either be light on dark or vice versa.
    min_sigma : float, optional
        The minimum standard deviation for Gaussian Kernel used to compute
        Hessian matrix. Keep this low to detect smaller blobs.
    max_sigma : float, optional
        The maximum standard deviation for Gaussian Kernel used to compute
        Hessian matrix. Keep this high to detect larger blobs.
    num_sigma : int, optional
        The number of intermediate values of standard deviations to consider
        between `min_sigma` and `max_sigma`.
    threshold : float, optional.
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect less prominent blobs.
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    log_scale : bool, optional
        If set intermediate values of standard deviations are interpolated
        using a logarithmic scale to the base `10`. If not, linear
        interpolation is used.
    Returns
    -------
    A : (n, 3) ndarray
        A 2d array with each row representing 3 values, ``(y,x,sigma)``
        where ``(y,x)`` are coordinates of the blob and ``sigma`` is the
        standard deviation of the Gaussian kernel of the Hessian Matrix whose
        determinant detected the blob.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_determinant_of_the_Hessian
    .. [2] Herbert Bay, Andreas Ess, Tinne Tuytelaars, Luc Van Gool,
           "SURF: Speeded Up Robust Features"
           ftp://ftp.vision.ee.ethz.ch/publications/articles/eth_biwi_00517.pdf
    Examples
    --------
    >>> from skimage import data, feature
    >>> img = data.coins()
    >>> feature.blob_doh(img)
    array([[270.        , 363.        ,  30.        ],
           [265.        , 113.        ,  23.55555556],
           [262.        , 243.        ,  23.55555556],
           [260.        , 173.        ,  30.        ],
           [197.        , 153.        ,  20.33333333],
           [197.        ,  44.        ,  20.33333333],
           [195.        , 100.        ,  23.55555556],
           [193.        , 275.        ,  23.55555556],
           [192.        , 212.        ,  23.55555556],
           [185.        , 348.        ,  30.        ],
           [156.        , 302.        ,  30.        ],
           [126.        , 153.        ,  20.33333333],
           [126.        , 101.        ,  20.33333333],
           [124.        , 336.        ,  20.33333333],
           [123.        , 205.        ,  20.33333333],
           [123.        ,  44.        ,  23.55555556],
           [121.        , 271.        ,  30.        ]])
    Notes
    -----
    The radius of each blob is approximately `sigma`.
    Computation of Determinant of Hessians is independent of the standard
    deviation. Therefore detecting larger blobs won't take more time. In
    methods line :py:meth:`blob_dog` and :py:meth:`blob_log` the computation
    of Gaussians for larger `sigma` takes more time. The downside is that
    this method can't be used for detecting blobs of radius less than `3px`
    due to the box filters used in the approximation of Hessian Determinant.
    """

    if image.ndim != 2:
        raise ValueError('Blob detection with determinant of hessian requires\
         2D array')

    image = _daskarray_to_float(image)
    image = integral_image(image)

    if log_scale:
        start, stop = math.log(min_sigma, 10), math.log(max_sigma, 10)
        sigma_list = np.logspace(start, stop, num_sigma)
    else:
        sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)

    # implement with mask overlap (depth max_sigma * sqrt(2)
    depth = int(np.ceil(max_sigma*math.sqrt(image.ndim)))



    hessian_images = [image.map_overlap(
            _hessian_matrix_det,
            depth=depth,
            sigma=s,
            dtype=image.dtype
        ) for s in sigma_list]

    image_stack = da.dstack(hessian_images)

    chunk_shape = image_stack.chunks
    new_shape = chunk_shape[:-1] + ((sum(chunk_shape[-1]),),)
    image_stack = image_stack.rechunk(chunks=new_shape)

    local_maxima = peak_local_max(image_stack, threshold_abs=threshold,
                                  footprint=np.ones((3,) * image_stack.ndim),
                                  threshold_rel=0.0,
                                  exclude_border=False)

    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, 3))

    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)

    # Convert the last index to its corresponding scale value
    lm[:, -1] = sigma_list[local_maxima[:, -1]]

    return _prune_blobs(lm, overlap)










