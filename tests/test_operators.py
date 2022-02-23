import pytest
import numpy as np
import torch
import tomosipo as ts
from ts_algorithms import ATA_max_eigenvalue


def test_ATA_max_eigenvalue():
    # Setup 3D volume and cone_beam projection geometry
    vg = ts.volume(shape=(64, 64, 64), size=(1, 1, 1))
    pg = ts.cone(angles=64, shape=(96, 96), size=(1.5, 1.5), src_orig_dist=3, src_det_dist=3)
    A = ts.operator(vg, pg)

    lower_bounds = []
    upper_bounds = []
    
    for iters in range(1, 5):
        lower_bound, upper_bound = ATA_max_eigenvalue(A, num_iterations=iters)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)
    
    # lower bounds should be increasing
    assert np.all(np.diff(np.array(lower_bounds)) >= 0)
    # upper bounds should be decreasing
    assert np.all(np.diff(np.array(upper_bounds)) <= 0)
    # the lower bound should be lower than the upper bound
    assert lower_bounds[-1] <= upper_bounds[-1]


def test_ATA_max_eigenvalue_masks():
    # Setup 3D volume and cone_beam projection geometry
    vg = ts.volume(shape=(1, 64, 64))
    pg = ts.parallel(angles=64, shape=(1, 96))
    A = ts.operator(vg, pg)

    lower_bounds = []
    upper_bounds = []
    
    volume_mask = torch.ones(A.domain_shape, dtype=torch.bool)
    projection_mask = torch.ones(A.range_shape, dtype=torch.bool)
    
    no_mask_lower_bound, no_mask_upper_bound = ATA_max_eigenvalue(A, num_iterations=5)
    full_mask_lower_bound, full_mask_upper_bound = ATA_max_eigenvalue(A, num_iterations=5)
    
    # Result with no mask should be the same as with a mask that's fully True
    assert no_mask_lower_bound == full_mask_lower_bound
    assert no_mask_upper_bound == full_mask_upper_bound
    
    volume_mask[:, :, ::2] = False
    projection_mask[:, :, ::2] = False
    
    half_mask_lower_bound, half_mask_upper_bound = ATA_max_eigenvalue(A, num_iterations=5)
    
    # Result with a mask where half the volume and half the projection data is
    # masked out should be larger than zero, but smaller than without a mask
    assert half_mask_lower_bound > 0
    assert half_mask_lower_bound <= half_mask_upper_bound
    assert half_mask_upper_bound <= no_mask_upper_bound
    
    
