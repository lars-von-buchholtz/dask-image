#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import numpy as np

import dask
import dask.array as da
import dask.array.utils as dau

from dask_image.ndfeature import  peak_local_max as da_plm
from skimage.feature import  peak_local_max as ski_plm


assert dask

def make_img(shape,points):
    img = np.zeros(shape,dtype=np.float)
    coord = points[:,:-1].astype(np.int).T
    img[tuple(coord)] = points[:,-1]
    return img

@pytest.mark.parametrize("shape,points",[((100,200),np.array([[50,50,0.7]])),
                                         ((100,200),np.array([[50,50,0.7],
                                                              [50,150,1.0]])),
                                         ((100,200),np.array([[50,50,1.0],
                                                              [50,150,1.0],
                                                              [50,99,1.0]])),
                                         ((100,200),np.array([[50,50,1.0],
                                                              [50,150,1.0],
                                                              [50,99,1.0],
                                                              [52,101,0.8]]))])
@pytest.mark.parametrize("min_distance,footprint",[(1,None),
                                                   (5,None),
                                                   (1,np.array([[0,1,0],
                                                               [1,1,1],
                                                               [0,1,0]])),
                                                   (1,np.array([[1,1,1],
                                                               [1,1,1],
                                                               [1,1,1]]))
                                                   ])
@pytest.mark.parametrize("threshold_abs,threshold_rel",[(0.5,None),
                                                        (0.75,None),
                                                        (0.9,None),
                                                        (None,0.5),
                                                        (None,0.75),
                                                        (None,0.9)
                                                        ])
@pytest.mark.parametrize("num_peaks",[np.inf,2])
def test_peak_local_max_2d(shape,points,min_distance,footprint,threshold_abs,\
                           threshold_rel,num_peaks):
    a = make_img(shape,points)
    chunks = [e/2 for e in shape]
    d = da.from_array(a,chunks=chunks)
    dau.assert_eq(ski_plm(a,min_distance=min_distance,
                          threshold_abs=threshold_abs,
                          threshold_rel=threshold_rel,
                          footprint=footprint,
                          num_peaks=num_peaks,
                          exclude_border=False),
                  da_plm(d, min_distance=min_distance,
                          threshold_abs=threshold_abs,
                          threshold_rel=threshold_rel,
                          footprint=footprint,
                          num_peaks=num_peaks,
                         exclude_border=False),
                  )


# test indices = False works, too

# test num_peaks limits the number of points

# test 3D and 4D

