import numpy as np
import math
from scipy import spatial


def _compute_disk_overlap(d, r1, r2):
    """
    Compute fraction of surface overlap between two disks of radii
    ``r1`` and ``r2``, with centers separated by a distance ``d``.
    Parameters
    ----------
    d : float
        Distance between centers.
    r1 : float
        Radius of the first disk.
    r2 : float
        Radius of the second disk.
    Returns
    -------
    fraction: float
        Fraction of area of the overlap between the two disks.
    """

    ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
    ratio1 = np.clip(ratio1, -1, 1)
    acos1 = math.acos(ratio1)

    ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
    ratio2 = np.clip(ratio2, -1, 1)
    acos2 = math.acos(ratio2)

    a = -d + r2 + r1
    b = d - r2 + r1
    c = d + r2 - r1
    d = d + r2 + r1
    area = (r1 ** 2 * acos1 + r2 ** 2 * acos2 -
            0.5 * math.sqrt(abs(a * b * c * d)))
    return area / (math.pi * (min(r1, r2) ** 2))


def _compute_sphere_overlap(d, r1, r2):
    """
    Compute volume overlap fraction between two spheres of radii
    ``r1`` and ``r2``, with centers separated by a distance ``d``.
    Parameters
    ----------
    d : float
        Distance between centers.
    r1 : float
        Radius of the first sphere.
    r2 : float
        Radius of the second sphere.
    Returns
    -------
    fraction: float
        Fraction of volume of the overlap between the two spheres.
    Notes
    -----
    See for example http://mathworld.wolfram.com/Sphere-SphereIntersection.html
    for more details.
    """
    vol = (math.pi / (12 * d) * (r1 + r2 - d)**2 *
           (d**2 + 2 * d * (r1 + r2) - 3 * (r1**2 + r2**2) + 6 * r1 * r2))
    return vol / (4./3 * math.pi * min(r1, r2) ** 3)


def _blob_overlap(blob1, blob2, *, sigma_dim=1):
    """Finds the overlapping area fraction between two blobs.
    Returns a float representing fraction of overlapped area. Note that 0.0
    is *always* returned for dimension greater than 3.
    Parameters
    ----------
    blob1 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.
    blob2 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.
    sigma_dim : int, optional
        The dimensionality of the sigma value. Can be 1 or the same as the
        dimensionality of the blob space (2 or 3).
    Returns
    -------
    f : float
        Fraction of overlapped area (or volume in 3D).
    """
    ndim = len(blob1) - sigma_dim
    if ndim > 3:
        return 0.0
    root_ndim = math.sqrt(ndim)

    # we divide coordinates by sigma * sqrt(ndim) to rescale space to isotropy,
    # giving spheres of radius = 1 or < 1.
    if blob1[-1] > blob2[-1]:
        max_sigma = blob1[-sigma_dim:]
        r1 = 1
        r2 = blob2[-1] / blob1[-1]
    else:
        max_sigma = blob2[-sigma_dim:]
        r2 = 1
        r1 = blob1[-1] / blob2[-1]
    pos1 = blob1[:ndim] / (max_sigma * root_ndim)
    pos2 = blob2[:ndim] / (max_sigma * root_ndim)

    d = np.sqrt(np.sum((pos2 - pos1)**2))
    if d > r1 + r2:  # centers farther than sum of radii, so no overlap
        return 0.0

    # one blob is inside the other
    if d <= abs(r1 - r2):
        return 1.0

    if ndim == 2:
        return _compute_disk_overlap(d, r1, r2)

    else:  # ndim=3 http://mathworld.wolfram.com/Sphere-SphereIntersection.html
        return _compute_sphere_overlap(d, r1, r2)


def _prune_blobs(blobs_array, overlap, *, sigma_dim=1):
    """Eliminate blobs with area overlap.
    Parameters
    ----------
    blobs_array : ndarray
        A 2d array with each row representing a blob in n-dimensional space,
        with the first n columns representing coordinates of the blob and the
        last ``sigma_dim`` columns representing the standard deviation of the
        Gaussian kernel which detected the blob.
        This array must not have a dimension of size 0.
    overlap : float
        A value between 0 and 1. If the fraction of area overlapping for 2
        blobs is greater than `overlap` the smaller blob is eliminated.
    sigma_dim : int, optional
        The number of columns in ``blobs_array`` corresponding to sigmas rather
        than positions.
    Returns
    -------
    A : ndarray
        `array` with overlapping blobs removed.

    Note
    -------
    Overlapping blobs are only removed in 2D and 3D images. In more dimensions
    all blobs are returned.
    """
    sigma = blobs_array[:, -sigma_dim:].max()
    distance = 2 * sigma * math.sqrt(blobs_array.shape[1] - sigma_dim)
    tree = spatial.cKDTree(blobs_array[:, :-sigma_dim])
    pairs = np.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return blobs_array
    else:
        for (i, j) in pairs:
            blob1, blob2 = blobs_array[i], blobs_array[j]
            if _blob_overlap(blob1, blob2, sigma_dim=sigma_dim) > overlap:
                # note: this test works even in the anisotropic case because
                # all sigmas increase together.
                if blob1[-1] > blob2[-1]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0

    return np.array([b for b in blobs_array if b[-1] > 0])
