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
    """function to generate blob images from an image shape and blob
    coordinates and sigmas

    :param shape:shape of image to generate
    :param blobs: array with blob coordinates and sigma in last column"""

    img = np.zeros(shape, dtype=np.float32)
    if blobs is None:
        return img
    for blob in blobs:
        tmp_img = np.zeros(shape, dtype=np.float)
        tmp_img[tuple(blob[:-1])] = 1.0
        gaussian_filter(tmp_img, blob[-1], output=tmp_img)
        tmp_img = tmp_img / np.max(tmp_img)
        img += tmp_img
    return img


def sort_array(a):
    a = a[a[:, 1].argsort(kind="mergesort")]
    a = a[a[:, 0].argsort(kind="mergesort")]
    return a


@pytest.mark.parametrize(
    "shape, chunks, blobs", [((60, 30, 30), (30, 30, 30),
                              np.array([[10, 10, 10, 3]]))]
)
def test_blob_log_3d(shape, chunks, blobs):

    a = generate_blobimage(shape, blobs)
    d = da.from_array(a, chunks=chunks)
    ski_r = ski_feat.blob_log(a, min_sigma=1, max_sigma=4, num_sigma=4)

    da_r = da_feat.blob_log(d, min_sigma=1, max_sigma=4, num_sigma=4)

    ski_r = sort_array(ski_r)
    da_r = sort_array(da_r)
    dau.assert_eq(ski_r, da_r)


@pytest.mark.parametrize(
    "shape, chunks, blobs", [((60, 30, 30), (30, 30, 30),
                              np.array([[10, 10, 10, 3]]))]
)
def test_blob_dog_3d(shape, chunks, blobs):
    a = generate_blobimage(shape, blobs)
    d = da.from_array(a, chunks=chunks)
    ski_r = ski_feat.blob_dog(a, min_sigma=1, max_sigma=4, threshold=0.000001)

    da_r = da_feat.blob_dog(d, min_sigma=1, max_sigma=4, threshold=0.000001)

    ski_r = sort_array(ski_r)
    da_r = sort_array(da_r)
    dau.assert_eq(ski_r, da_r)


@pytest.mark.parametrize(
    "shape, blobs",
    [
        ((100, 200), None),
        ((100, 200), np.array([[25, 25, 5]])),
        ((100, 200), np.array([[25, 25, 5], [49, 99, 10]])),
    ],
)
@pytest.mark.parametrize("min_sigma, max_sigma, num_sigma",
                         [(1, 10, 10), (2, 11, 10)])
@pytest.mark.parametrize("threshold", [0.01, 0.9])
@pytest.mark.parametrize("overlap", [0.1])
@pytest.mark.parametrize("log_scale", [True, False])
def test_blob_doh_2d(
    shape, blobs, min_sigma, max_sigma, num_sigma, overlap,
    threshold, log_scale
):
    a = generate_blobimage(shape, blobs)
    chunks = [e // 2 for e in shape]
    d = da.from_array(a, chunks=chunks)
    ski_r = ski_feat.blob_doh(
        a,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap,
        log_scale=log_scale,
    )

    da_r = da_feat.blob_doh(
        d,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap,
        log_scale=log_scale,
    )

    ski_r = sort_array(ski_r)
    da_r = sort_array(da_r)
    print(ski_r)
    print(da_r)
    # coordinates are sometimes off by 1 when using the determinant
    coord_diff = (ski_r - da_r)[:, : a.ndim]
    sigma_diff = (ski_r - da_r)[:, a.ndim:]
    assert np.all(coord_diff <= 1)
    assert np.all(np.abs(sigma_diff) <= 0.01)


@pytest.mark.parametrize(
    "shape, blobs",
    [
        ((100, 200), None),
        ((100, 200), np.array([[50, 50, 5]])),
        ((100, 200), np.array([[50, 50, 5], [49, 99, 6]])),
        ((100, 200), np.array([[50, 50, 5], [49, 99, 6], [52, 102, 3]])),
    ],
)
@pytest.mark.parametrize(
    "min_sigma, max_sigma, sigma_ratio",
    [(1, 10, 1.6), (2, 7, 2.0), ([1, 1], [10, 10], 1.6)],
)
@pytest.mark.parametrize("threshold", [0.001, 0.9])
@pytest.mark.parametrize("overlap", [0.1, 0.9])
def test_blob_dog_2d(
    shape, blobs, min_sigma, max_sigma, sigma_ratio, overlap, threshold
):
    a = generate_blobimage(shape, blobs)
    chunks = [e // 2 for e in shape]
    d = da.from_array(a, chunks=chunks)
    ski_r = ski_feat.blob_dog(
        a,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        sigma_ratio=sigma_ratio,
        threshold=threshold,
        overlap=overlap,
    )

    print(ski_r)
    da_r = da_feat.blob_dog(
        d,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        sigma_ratio=sigma_ratio,
        threshold=threshold,
        overlap=overlap,
    )

    ski_r = sort_array(ski_r)
    da_r = sort_array(da_r)
    dau.assert_eq(ski_r, da_r)


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
    "min_sigma, max_sigma, num_sigma",
    [(1, 10, 10), (2, 11, 10), ([1, 1], [10, 10], 10)],
)
@pytest.mark.parametrize("threshold", [0.001, 0.9])
@pytest.mark.parametrize("overlap", [0.1, 0.9])
@pytest.mark.parametrize("log_scale", [True, False])
def test_blob_log_2d(
    shape, blobs, min_sigma, max_sigma, num_sigma, overlap, threshold,
    log_scale
):
    a = generate_blobimage(shape, blobs)
    chunks = [e // 2 for e in shape]
    d = da.from_array(a, chunks=chunks)
    ski_r = ski_feat.blob_log(
        a,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap,
        log_scale=log_scale,
    )

    da_r = da_feat.blob_log(
        d,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap,
        log_scale=log_scale,
    )

    ski_r = sort_array(ski_r)
    da_r = sort_array(da_r)
    dau.assert_eq(ski_r, da_r)


def test_2d_check():
    img = np.ones((10, 10, 10))
    with pytest.raises(ValueError):
        da_feat.blob_doh(img)
