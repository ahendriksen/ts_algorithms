import pytest
import torch
import tomosipo as ts
from ts_algorithms import fdk


def make_box_phantom():
    x = torch.ones((32, 32, 32))
    x[4:28, 4:28, 4:28] = 1.0
    x[12:22, 12:22, 12:22] = 0.0
    return x


def test_fdk_reconstruction():
    vg = ts.volume(shape=32, size=1)
    pg = ts.cone(angles=32, shape=(48, 48), size=(2, 2), src_det_dist=2, src_orig_dist=2)

    A = ts.operator(vg, pg)
    x = make_box_phantom()
    y = A(x)

    # very rough test if reconstruction is close to original volume
    rec = fdk(A, y)
    assert torch.allclose(rec, x, atol=1, rtol=0)
    rec_nonPadded = fdk(A, y, padded=False)
    assert torch.allclose(rec_nonPadded, x, atol=1, rtol=0)

    # for a circular geometry recalculating the weights should not change the result
    rec_recalculate = fdk(A, y, recalculate_weights=True)
    assert torch.allclose(rec_recalculate, rec, atol=1e-6, rtol=1e-5)

    # test whether GPU and CPU calculations yield the same result
    assert torch.allclose(fdk(A, y.cuda()).cpu(), fdk(A, y.cpu()), atol=1e-4, rtol=1e-3)


def test_fdk_anisotropic_pixels():
    vg = ts.volume(shape=32, size=1)
    pg = ts.cone(angles=32, shape=(48, 64), size=(2, 2), src_det_dist=2, src_orig_dist=2)

    A = ts.operator(vg, pg)
    x = make_box_phantom()
    y = A(x)

    # very rough test for reconstruction with anisotropic detector pixels
    assert torch.allclose(fdk(A, y), x, atol=1, rtol=0)
