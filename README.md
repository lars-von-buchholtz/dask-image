# Blob Detection for Dask-Image

Travis:
[![Build Status](https://travis-ci.com/lars-von-buchholtz/dask-image.svg?branch=master)](https://travis-ci.com/lars-von-buchholtz/dask-image)

Coveralls:
[![Coverage Status](https://coveralls.io/repos/github/lars-von-buchholtz/dask-image/badge.svg?branch=master)](https://coveralls.io/github/lars-von-buchholtz/dask-image?branch=develop)

Appveyor:
[![Build status](https://ci.appveyor.com/api/projects/status/f04mv9c38sv03eiq?svg=true)](https://ci.appveyor.com/project/lars-von-buchholtz/dask-image)

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Installation](#installation)
- [My contributions to Github Repo](#my-contributions-to-github-repo)
- [Demo](#demo)
- [Documentation of new functions](#documentation-of-new-functions)
  - [dask_image.ndfeature.peak_local_max](#dask_imagendfeaturepeak_local_max)
  - [dask_image.ndfeature.blob_log](#dask_imagendfeatureblob_log)
  - [dask_image.ndfeature.blob_dog](#dask_imagendfeatureblob_dog)
  - [dask_image.ndfeature.blob_doh](#dask_imagendfeatureblob_doh)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Installation

Install from my forked (public) repo on github:

```$ pip install git+git://github.com/lars-von-buchholtz/dask-image.git```

## My contributions to Github Repo

The `/dask-image/ndfeature/` folder and the `/tests/test_dask_image/test_ndfeature/` folders were contributed by me. The rest of the package was left unchanged. I will submit a pull request for the CSCI E-29 class to highlight the contributions I made since the forking.

  After grading of the project, I will change the Readme and submit the state of this repo as a pull request to Dask-Image.

## Demo

A Demo jupyter notebook can be found [here](https://github.com/lars-von-buchholtz/dask-image/blob/master/blob_demo.ipynb).


## Documentation of new functions

### dask_image.ndfeature.peak_local_max

    Find peaks in a dask image as coordinate list or boolean mask.

    Adapted from scikit-image for dask.
    Peaks are the local maxima in a region of `2 * min_distance + 1`
    (i.e. peaks are separated by at least `min_distance`).
    If there are multiple local maxima with identical pixel intensities
    inside the region defined with `min_distance`,
    the coordinates of all such pixels are returned.
    Parameters
    ----------
    image : n-dimensional dask array
        Input image.
    min_distance : int, optional
        Minimum number of pixels separating peaks in a region of `2 *
        min_distance + 1` (i.e. peaks are separated by at least
        `min_distance`).
        To find the maximum number of peaks, use `min_distance=1`.
    threshold : float, optional
        Minimum intensity of peaks. By default, the absolute threshold is
        the minimum intensity of the image.
    exclude_border : int, bool, or sequence of int, optional
        If nonzero int, `exclude_border` excludes peaks from
        within `exclude_border`-pixels of the border of the image.
        If True, takes the `min_distance` parameter as value.
        If Sequence of ints, excluded pixels can be defined for each image
        dimension independently.
        If zero or False, peaks are identified regardless of their
        distance from the border.
    indices : bool, optional
        If True, the output will be an array representing peak
        coordinates.  If False, the output will be a boolean array shaped as
        `image.shape` with peaks present at True elements.
    num_peaks : int, optional
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` peaks based on highest peak intensity. Works only
        if indices is True.
    footprint : ndarray of bools, optional
        If provided, `footprint == 1` represents the local region within which
        to search for peaks at every point in `image`.  Overrides
        `min_distance`.
    Returns
    -------
    output : ndarray or dask array of bools
        * If `indices = True`  : (row, column, ...) coordinates of peaks as
         ndarray.
        * If `indices = False` : Boolean dask array shaped like `image`,
        with peaks
          represented by True values.

    Examples
    --------
    >>> from dask_image.ndfeature import peak_local_max
    >>> import dask.array as da
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
    >>> dsk_img1 = da.from_array(img1)
    >>> peak_local_max(dsk_img1, min_distance=1)
    array([[3, 4],
           [3, 2]])
    >>> peak_local_max(dsk_img1, min_distance=2)
    array([[3, 2]])
    >>> img2 = np.zeros((20, 20, 20))
    >>> img2[10, 10, 10] = 1
    >>> dsk_img2 = da.from_array(img2)
    >>> peak_local_max(dsk_img2, exclude_border=0)
    array([[10, 10, 10]])

### dask_image.ndfeature.blob_log

    Finds blobs in the given grayscale dask array.

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
           
### dask_image.ndfeature.blob_dog

    Finds blobs in the given grayscale dask image.

    Adapted from skimage.feature.blog_dog
    Blobs are found using the Difference of Gaussian (DoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.
    Parameters
    ----------
    image : n-dimensional dask array
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
    .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_difference_of_
    Gaussians_approach
    Examples
    --------
    >>> from skimage import data, feature
    >>> import dask.array as da
    >>> from dask_image.ndfeature import blob_dog
    >>> blob_dog(da.from_array(data.coins()), threshold=.5, max_sigma=40)
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
    
    
### dask_image.ndfeature.blob_doh

    Finds blobs in the given grayscale dask image.

    Adapted for dask from scikit-image.feature.blob_doh
    Blobs are found using the Determinant of Hessian method [1]_. For each blob
    found, the method returns its coordinates and the standard deviation
    of the Gaussian Kernel used for the Hessian matrix whose determinant
    detected the blob. Determinant of Hessians is approximated using [2]_.
    Parameters
    ----------
    image : 2D dask array
        Input grayscale image. Blobs can either be light on dark or vice versa.
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
    .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_
    determinant_of_the_Hessian
    .. [2] Herbert Bay, Andreas Ess, Tinne Tuytelaars, Luc Van Gool,
           "SURF: Speeded Up Robust Features"
           ftp://ftp.vision.ee.ethz.ch/publications/articles/eth_biwi_00517.pdf
    Examples
    --------
    >>> from skimage import data
    >>> import dask.array as da
    >>> from dask_image.ndfeature import blob_doh
    >>> blob_dog(da.from_array(data.coins()))
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
    due to the box filters used in the approximation of Hessian Determinant
    and that the algorithm is currently limited to 2 dimensions.
    
