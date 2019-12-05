import numpy as np
from scipy.ndimage import gaussian_filter
import math
#from math import sqrt, log
#from scipy import spatial
#from math import sqrt
import dask.array as da
from scipy import spatial, ndimage as ndi
from skimage import img_as_float
from skimage.feature.peak import peak_local_max as ski_peak_local_max
from skimage.feature._hessian_det_appx import _hessian_matrix_det
#from ..transform import integral_image
#from .._shared.utils import check_nD
from ..ndmeasure import labeled_comprehension
from ..ndfilters._gaussian import _get_sigmas, _get_border, gaussian_laplace
from ..ndfilters import _utils
from ._skimage_utils import _exclude_border, _prune_blobs

def _get_high_intensity_peaks(image, mask, num_peaks):
    """
    Return the highest intensity peak coordinates. Adapted from skimage.feature.peak._get_high_intensity_peaks
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

def peak_local_max(image, min_distance=1, threshold_abs=None,
                   threshold_rel=None, exclude_border=True, indices=True,
                   num_peaks=np.inf, footprint=None):

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
        * If `indices = True`  : (row, column, ...) coordinates of peaks as ndarray.
        * If `indices = False` : Boolean dask array shaped like `image`, with peaks
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

    # calculate depth and boundaries based on min_distance and/or footprint
    if not (min_distance or footprint):
        raise ValueError('Either min_distance or footprint must be specified')


    if type(footprint) is np.ndarray:
        depth = footprint.shape
    else:
        depth = 2 * min_distance + 1

    # map_overlap plm without border exclude, labels, indices=False
    print(image.chunks)
    mask = image.map_overlap(
        ski_peak_local_max,
        depth=depth,
        min_distance=min_distance, threshold_abs=threshold_abs,
        threshold_rel=threshold_rel, exclude_border=False, indices=False,
        num_peaks=np.inf, footprint=footprint, labels=None,
        num_peaks_per_label=np.inf
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
        out = da.zeros_like(image,dtype=np.bool)
        nd_indices = tuple(coordinates.T)
        out[nd_indices] = True
        return out



def blob_common(blob_func):
    """Decorator for functionality that is conserved between blob_log and blob_dog"""
    @wraps(blob_func)
    def wrapped_func(*args, **kwargs)
        image = kwargs['image'] if 'image' in kwargs else args[0]
        min_sigma = kwargs['min_sigma'] if 'min_sigma' in kwargs else args[1]
        max_sigma = kwargs['max_sigma'] if 'max_sigma' in kwargs else args[2]

        scalar_sigma = (
            True if np.isscalar(max_sigma) and np.isscalar(min_sigma) else False
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
        kwargs['image'] = _daskarray_to_float(image)
        image_stack = blob_func(*args,**kwargs)
        local_maxima = peak_local_max(image_stack, threshold_abs=kwargs['threshold'],
                                       footprint=np.ones((3,) * (image.ndim + 1)),
                                       threshold_rel=0.0,
                                       exclude_border=kwargs['exclude_border'])

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

        return _prune_blobs(lm, kwargs['overlap'], sigma_dim=sigma_dim)

    return wrapped_func





def blob_log(image, min_sigma=1, max_sigma=50, num_sigma=10, threshold=.2,
             overlap=.5, log_scale=False, *, exclude_border=False):
    r"""Finds blobs in the given grayscale image.

    This implementation adapts the skimage.feature.blob_log function for Dask.
    Blobs are found using the Laplacian of Gaussian (LoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.
    Parameters
    ----------
    image : 2D or 3D ndarray
        Input grayscale image, blobs are assumed to be light on dark
        background (white on black).
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
        If set intermediate values of standard deviations are interpolated
        using a logarithmic scale to the base `10`. If not, linear
        interpolation is used.
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
    .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_of_Gaussian
    Examples
    --------
    >>> from skimage import data, feature, exposure
    >>> img = data.coins()
    >>> img = exposure.equalize_hist(img)  # improves detection
    >>> feature.blob_log(img, threshold = .3)
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
    #image = img_as_float(image) #check this out

    # if both min and max sigma are scalar, function returns only one sigma
    scalar_sigma = (
        True if np.isscalar(max_sigma) and np.isscalar(min_sigma) else False
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

    image_cube = da.stack(gl_images, axis=-1)
    chunk_shape = image_cube.chunks
    new_shape = chunk_shape[:-1] +((sum(chunk_shape[-1]),),)
    image_cube = image_cube.rechunk(chunks=new_shape)

    local_maxima = peak_local_max(image_cube, threshold_abs=threshold,
                                  footprint=np.ones((3,) * (image.ndim + 1)),
                                  threshold_rel=0.0,
                                  exclude_border=exclude_border)

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




