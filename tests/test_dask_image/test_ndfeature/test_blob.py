#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import numpy as np
import skimage.feature as ski_feat
from scipy.ndimage.filters import gaussian_filter

import dask
import dask.array as da
import dask.array.utils as dau

import dask_image.ndfeature as da_feat

assert dask

def generate_blobimage(shape,blobs):
    """function to generate blob images"""
    img = np.zeros(shape,dtype=np.float)
    for blob in blobs:
        tmp_img = np.zeros(shape,dtype=np.float)
        tmp_img[blob[:-1]] = 1.0
        gaussian_filter(tmp_img,blob[-1],output=tmp_img)
        img += tmp_img
    return img

#@pytest.mark.parametrize("da_func,ski_func",[(da_blob_log,ski_blob_log),
#                                             (da_blob_dog,ski_blob_dog)])
# @pytest.mark.parametrize()
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

