#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import numpy as np

from scipy.ndimage.filters import gaussian_filter

import dask
import dask.array as da
import dask.array.utils as dau

import skimage.feature as ski_feat
import dask_image.ndfeature as da_feat

assert dask


def generate_blobimage(shape, blobs):
    """function to generate blob images


    :param shape:shape of image to generate
    :param blobs: array with blob coordinates and sigma as rows"""

    img = np.zeros(shape, dtype=np.float)
    if blobs is None:
        return img
    for blob in blobs:
        tmp_img = np.zeros(shape, dtype=np.float)
        tmp_img[blob[:-1]] = 1.0
        gaussian_filter(tmp_img, blob[-1], output=tmp_img)
        img += tmp_img
    return img


@pytest.mark.parametrize(
    "da_func,ski_func",
    [
        (da_feat.blob_log, ski_feat.blob_log),
        (da_feat.blob_dog, ski_feat.blob_dog),
        (da_feat.blob_doh, ski_feat.blob_doh),
    ],
)
@pytest.mark.parametrize(
    "shape, blobs",
    [
        ((100, 200), None),
        ((100, 200), np.array([[50, 50, 5]])),
        ((100, 200), np.array([[50, 50, 5], [49, 99, 10]])),
        ((100, 200), np.array([[50, 50, 5], [49, 99, 10], [52, 102, 10]])),
    ],
)
@pytest.mark.parametrize(
    "min_sigma, max_sigma, num_sigma, sigma_ratio",
    [
        (1, None),
        (5, None),
        (1, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])),
        (1, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])),
    ],
)
@pytest.mark.parametrize("threshold", [0.001, 0.1, 0.9])
@pytest.mark.parametrize("overlap", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("log_scale", [True, False])
def test_peak_local_max_2d(
    shape, blobs, min_distance, footprint, threshold_abs, num_peaks
):
    a = make_img(shape, points)
    chunks = [e // 2 for e in shape]
    d = da.from_array(a, chunks=chunks)
    ski_r = ski_plm(
        a,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        footprint=footprint,
        num_peaks=num_peaks,
        exclude_border=False,
    )
    # sort the arrays
    ski_r = ski_r[ski_r[:, 1].argsort(kind="mergesort")]
    ski_r = ski_r[ski_r[:, 0].argsort(kind="mergesort")]
    da_r = da_plm(
        d,
        min_distance=min_distance,
        threshold=threshold_abs,
        footprint=footprint,
        num_peaks=num_peaks,
        exclude_border=False,
    )
    da_r = da_r[da_r[:, 1].argsort(kind="mergesort")]
    da_r = da_r[da_r[:, 0].argsort(kind="mergesort")]
    dau.assert_eq(ski_r, da_r)


# def test_2d():

# test 0,1,2,3 blobs are detected with single sigma

# test a few sigma ranges

# test 2 blobs are detected with symmetric sequence sigma

# test num sigma = 1, for blob_log only

# test sigma ratio values for blob_dog only

# test log_scale works

# test overlap eliminates a blob and leaves other intact

# test exclude border eliminates blobs

# test 3D and 4D
