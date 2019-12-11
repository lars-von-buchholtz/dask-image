#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import numpy as np

import dask
import dask.array as da
import dask.array.utils as dau

from dask_image.ndfeature import peak_local_max as da_plm
from dask_image.ndfeature._utils import _daskarray_to_float
from skimage.feature import peak_local_max as ski_plm


assert dask


def make_img(shape, points):
    img = np.zeros(shape, dtype=np.float)
    if points is not None:
        coord = points[:, :-1].astype(np.int).T
        img[tuple(coord)] = points[:, -1]
    return img


@pytest.mark.parametrize("in_type, out_type", [(np.float16, np.float16),
                                               (np.float32, np.float32),
                                               (np.float32, np.float32),
                                               (np.uint8, np.float16),
                                               (np.uint16, np.float16),
                                               (np.uint32, np.float32),
                                               (np.int8, np.float16),
                                               (np.int16, np.float16),
                                               (np.int32, np.float32)
                                               ])
def test_tofloat(in_type, out_type):
    img = da.from_array(np.ones((5, 5), dtype=in_type))
    img_out = _daskarray_to_float(img)
    assert img_out.dtype == out_type


@pytest.mark.parametrize(
    "shape, points",
    [
        ((100, 200), None),
        ((100, 200), np.array([[50, 50, 0.7]])),
        ((100, 200), np.array([[50, 50, 0.7], [50, 150, 1.0]])),
        ((100, 200), np.array([[2, 2, 1.0], [50, 150, 1.0], [50, 99, 1.0]])),
        (
            (100, 200),
            np.array([[50, 50, 1.0],
                      [50, 150, 1.0],
                      [50, 99, 1.0],
                      [52, 101, 0.8]]),
        ),
    ],
)
@pytest.mark.parametrize(
    "min_distance, footprint",
    [
        (1, None),
        (5, None),
        (1, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])),
        (1, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])),
    ],
)
@pytest.mark.parametrize("threshold_abs", [0.5, 0.75, 0.9])
@pytest.mark.parametrize("num_peaks", [np.inf, 2])
def test_peak_local_max_2d(
    shape, points, min_distance, footprint, threshold_abs, num_peaks,
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


@pytest.mark.parametrize(
    "shape, points",
    [
        ((100, 200), np.array([[50, 50, 0.7]])),
        ((100, 200), np.array([[50, 50, 0.7], [50, 150, 1.0]])),
        ((100, 200), np.array([[50, 50, 1.0], [50, 150, 1.0], [50, 99, 1.0]])),
        (
            (100, 200),
            np.array([[50, 50, 1.0],
                      [50, 150, 1.0],
                      [50, 99, 1.0],
                      [52, 101, 0.8]]),
        ),
    ],
)
@pytest.mark.parametrize(
    "min_distance, footprint",
    [
        (1, None),
        (5, None),
        (1, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])),
        (1, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])),
    ],
)
@pytest.mark.parametrize("threshold_abs", [0.5, 0.75, 0.9])
def test_mask_output(shape, points, min_distance, footprint, threshold_abs):
    a = make_img(shape, points)
    chunks = [e // 2 for e in shape]
    d = da.from_array(a, chunks=chunks)
    ski_r = ski_plm(
        a,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        footprint=footprint,
        exclude_border=False,
        indices=False,
    )

    da_r = da_plm(
        d,
        min_distance=min_distance,
        threshold=threshold_abs,
        footprint=footprint,
        exclude_border=False,
        indices=False,
    )

    dau.assert_eq(ski_r, da_r)


# test 3D and 4D input
@pytest.mark.parametrize(
    "shape, points",
    [
        ((50, 50, 20), np.array([[10, 10, 5, 0.7]])),
        ((50, 50, 10, 10), np.array([[10, 10, 5, 5, 0.7],
                                     [12, 12, 5, 5, 1.0]]))
    ]
)
@pytest.mark.parametrize("min_distance, footprint", [(1, None), (5, None)])
@pytest.mark.parametrize("threshold_abs", [0.5, 0.75])
@pytest.mark.parametrize("num_peaks", [np.inf, 1])
def test_peak_local_max_nd(
    shape, points, min_distance, footprint, threshold_abs, num_peaks
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
