import pytest
import numpy as np
import torch
import tomosipo as ts
from ts_algorithms import nag_ls, ATA_max_eigenvalue

def test_ATA_max_eigenvalue():
    # Setup 3D volume and cone_beam projection geometry
    vg = ts.volume(shape=(64, 64, 64), size=(1, 1, 1))
    pg = ts.cone(angles=64, shape=(96, 96), size=(1.5, 1.5), src_orig_dist=3, src_det_dist=3)
    A = ts.operator(vg, pg)

    lower_bounds = []
    upper_bounds = []
    
    for iters in range(1, 5):
        lower_bound, upper_bound, x = ATA_max_eigenvalue(A, stop_iterations=iters)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)
    
    # lower bounds should be increasing
    assert np.all(np.diff(np.array(lower_bounds)) >= 0)
    # upper bounds should be decreasing
    assert np.all(np.diff(np.array(upper_bounds)) <= 0)
    # the lower bound should be lower than the upper bound
    assert lower_bounds[-1] <= upper_bounds[-1]
    

def test_nag():
    vg = ts.volume(shape=32)
    pg = ts.parallel(angles=32, shape=48)
    max_eig = 988.3

    A = ts.operator(vg, pg)

    x = torch.zeros(*A.domain_shape)
    x[4:28, 4:28, 4:28] = 1.0
    x[12:22, 12:22, 12:22] = 0.0

    y = A(x)

    nag_ls(A, y, 10, max_eig)
    
def test_projection_mask():
    vg = ts.volume(shape=32)
    pg = ts.parallel(angles=32, shape=48)
    max_eig = 988.3
    l2_regularization = 10

    A = ts.operator(vg, pg)

    x = torch.zeros(*A.domain_shape)
    x[4:28, 4:28, 4:28] = 1.0
    x[12:22, 12:22, 12:22] = 0.0

    y = A(x)
    # create a mask
    m = torch.zeros_like(y, dtype=torch.bool)
    m[20:24, :, :] = False
    # corrupt the projection data within the masked area
    y_corrupted = y+(~m)
    
    # without using the mask the results should be different
    result_normal = nag_ls(A, y, 10, max_eig, l2_regularization=l2_regularization)
    result_corrupted = nag_ls(A, y_corrupted, 10, max_eig, l2_regularization=l2_regularization)
    assert not torch.equal(result_normal, result_corrupted)

    # with the mask the changes to y should be masked out so the result
    # is the same
    result_normal_masked = nag_ls(A, y, 10, max_eig, projection_mask=m, l2_regularization=l2_regularization)
    result_corrupted_masked = nag_ls(A, y_corrupted, 10, max_eig, projection_mask=m, l2_regularization=l2_regularization)
    assert torch.equal(result_normal_masked, result_corrupted_masked)    
