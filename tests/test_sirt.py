import pytest
import torch
import tomosipo as ts
from ts_algorithms import sirt


def test_sirt():
    vg = ts.volume(shape=32)
    pg = ts.parallel(angles=32, shape=48)

    A = ts.operator(vg, pg)

    x = torch.zeros(*A.domain_shape)
    x[4:28, 4:28, 4:28] = 1.0
    x[12:22, 12:22, 12:22] = 0.0

    y = A(x)

    sirt(A, y, 10)


def test_x_init():
    vg = ts.volume(shape=32)
    pg = ts.parallel(angles=32, shape=48)

    A = ts.operator(vg, pg)

    x = torch.zeros(*A.domain_shape)
    x[4:28, 4:28, 4:28] = 1.0
    x[12:22, 12:22, 12:22] = 0.0

    y = A(x)

    result_together = sirt(A, y, 30)
    result_1 = sirt(A, y, 10)
    result_2 = sirt(A, y, 10, x_init = result_1)
    result_3 = sirt(A, y, 10, x_init = result_2)
    assert torch.equal(result_together, result_3)
    
    result_steps = sirt(A, y, 10)
    result_steps = sirt(A, y, 10, x_init = result_steps)
    result_steps = sirt(A, y, 10, x_init = result_steps)
    assert torch.equal(result_together, result_steps)


def test_projection_mask():
    vg = ts.volume(shape=32)
    pg = ts.parallel(angles=32, shape=48)

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
    result_normal = sirt(A, y, 10)
    result_corrupted = sirt(A, y_corrupted, 10)
    assert not torch.equal(result_normal, result_corrupted)

    # with the mask the changes to y should be masked out so the result
    # is the same
    result_normal_masked = sirt(A, y, 10, projection_mask=m)
    result_corrupted_masked = sirt(A, y_corrupted, 10, projection_mask=m)
    assert torch.equal(result_normal_masked, result_corrupted_masked)    
