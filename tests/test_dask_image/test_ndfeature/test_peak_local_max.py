#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import numpy as np
import skimage.feature as ski_feat

import dask
import dask.array as da
import dask.array.utils as dau

import dask_image.ndfeature as da_feat


assert dask

#@pytest.mark.parametrize()
def test_2d():
