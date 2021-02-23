import pytest
import torch
import tomosipo as ts
from ts_algorithms import fdk
from matplotlib import pyplot as plt


def make_box_phantom():
    x = torch.zeros((64, 64, 64))
    x[8:56, 8:56, 8:56] = 1.0
    x[24:44, 24:44, 24:44] = 0.0
    return x


def test_fdk_reconstruction():
    vg = ts.volume(shape=64, size=1)
    pg = ts.cone(angles=32, shape=(64, 64), size=(2, 2), src_det_dist=2, src_orig_dist=2)

    A = ts.operator(vg, pg)
    x = make_box_phantom()
    y = A(x)

    # rough test if reconstruction is close to original volume
    rec = fdk(A, y)
    assert torch.mean(torch.abs(rec - x)) < 0.15
    rec_nonPadded = fdk(A, y, padded=False)
    assert torch.mean(torch.abs(rec_nonPadded - x)) < 0.15

    # test whether GPU and CPU calculations yield the same result
    rec_cuda = fdk(A, y.cuda()).cpu()
    assert torch.allclose(rec_cuda, rec, atol=1e-3, rtol=1e-2)
    assert torch.mean(torch.abs(rec_cuda - rec)) < 1e-6


def test_fdk_anisotropic_pixels():
    vg = ts.volume(shape=64, size=1)
    pg = ts.cone(angles=32, shape=(64, 80), size=(2, 2), src_det_dist=2, src_orig_dist=2)

    A = ts.operator(vg, pg)
    x = make_box_phantom()
    y = A(x)

    # rough test for reconstruction with anisotropic detector pixels
    rec = fdk(A, y)
    assert torch.mean(torch.abs(rec - x)) < 0.15


def test_fdk_translation_invariance():
    vg = ts.volume(shape=64, size=1)
    pg = ts.cone(angles=32, shape=(64, 64), size=(2, 2), src_det_dist=2, src_orig_dist=2).to_vec()

    # Create sinogram and reconstruct at the origin
    A0 = ts.operator(vg, pg)
    x = make_box_phantom()
    y = A0(x)
    rec0 = fdk(A0, y, src_rot_center_dist=2)
    bp0 = A0.T(y)

    # Move volume and detector far away
    T = ts.translate((1000, 1000, 1000))
    A = ts.operator(T * vg, T * pg)

    rec = fdk(A, y, src_rot_center_dist=2)
    bp = A.T(y)

    # Check that backprojection still matches
    assert torch.allclose(bp, bp0)
    # Check that fdk still matches
    assert torch.allclose(rec, rec0)

 
def test_fdk_scaling_invariance():
    vg = ts.volume(shape=64, size=1)
    pg = ts.cone(angles=32, shape=(64, 64), size=(2, 2), src_det_dist=2, src_orig_dist=2).to_vec()

    # Create sinogram and reconstruct at the origin
    A0 = ts.operator(vg, pg)
    x = make_box_phantom()

    rec0 = fdk(A0, A0(x), src_rot_center_dist=2)
    bp0 = A0.T(A0(x))

    s = 2.0
    S = ts.scale(s)
    A = ts.operator(S * vg, S * pg)

    rec = fdk(A, A(x), src_rot_center_dist=2*s)
    bp = A.T(A(x))

    # Check that backprojection still matches (up to scaling)
    assert torch.allclose(bp / s**2, bp0)
    # Check that fdk still matches
    assert torch.allclose(rec, rec0)
