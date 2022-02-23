import pytest
import numpy as np
import torch
import tomosipo as ts
from ts_algorithms import nag_ls, ATA_max_eigenvalue


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
    
    # test the function once more, with the data on the GPU
    x = x.cuda()
    y = y.cuda()
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
