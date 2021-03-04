import pytest
import torch
import tomosipo as ts
from ts_algorithms import fdk
import numpy as np


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

    # test whether cone and cone_vec geometries yield the same result
    A_vec = ts.operator(vg, pg.to_vec())
    rec_vec = fdk(A_vec, y)
    assert torch.allclose(rec_vec, rec, atol=1e-3, rtol=1e-2)
    assert torch.mean(torch.abs(rec_vec - rec)) < 1e-6

    # test whether GPU and CPU calculations yield the same result
    rec_cuda = fdk(A, y.cuda()).cpu()
    assert torch.allclose(rec_cuda, rec, atol=1e-3, rtol=1e-2)
    assert torch.mean(torch.abs(rec_cuda - rec)) < 1e-6


def test_fdk_off_center_cor():
    """Test that fdk handles non-standard center of rotations correctly
    """

    vg = ts.volume(shape=64).to_vec()
    pg = ts.cone(angles=1, shape=(96, 96), src_det_dist=130).to_vec()

    angles = np.linspace(0, 2 * np.pi, 30)
    R = ts.rotate(pos=(0, -20, -20), axis=(1, 0, 0), rad=angles)

    A1 = ts.operator(vg, R * pg)
    A2 = ts.operator(R.inv * vg, pg)
    x = make_box_phantom()
    y1 = A1(x)
    y2 = A2(x)
    assert torch.allclose(y1, y2)
    bp1 = A1.T(y1)
    bp2 = A2.T(y2)
    assert torch.allclose(bp1, bp2)
    r1 = fdk(A1, y1)
    r2 = fdk(A2, y2)
    print(r1.mean(), r2.mean())
    assert torch.allclose(r1, r2)


def test_magnification_invariance():
    """Test fdk at various magnifications

    The reconstruction from FDK should be the same regardless of
    magnification.
    """

    vg = ts.volume(shape=64)
    angles = np.linspace(0, 2 * np.pi, 30)
    R = ts.rotate(pos=(0, 0, 0), axis=(1, 0, 0), rad=angles)

    # Create pg with various source and detector positions as well as
    # various pixel sizer to achieve various levels of magnification
    # (that are undone by the larger pixels).
    pgs = [
        R * ts.cone_vec(
            shape=(96, 96),
            src_pos=[[0, -130, 0]],
            det_pos=[[0, 130 * (m - 1), 0]],
            det_v=[[m, 0, 0]],
            det_u=[[0, 0, m]],
        )
        for m in [0.5, 1.0, 3.0]
    ]

    # Create operators, phantom, and sinograms
    As = [ts.operator(vg, pg) for pg in pgs]
    x = make_box_phantom()
    ys = [A(x) for A in As]

    # Make sure all sinograms are equal:
    for y1 in ys:
        for y2 in ys:
            assert torch.allclose(y1, y2, atol=1e-2, rtol=1e-5)

    # Make sure that all backprojections are equal:
    bps = [A.T(y) for A, y in zip(As, ys)]
    for bp1 in bps:
        for bp2 in bps:
            assert torch.allclose(bp1, bp2, atol=1e-2, rtol=1e-5)

    # And finally check the reconstructions
    recs = [fdk(A, y) for A, y in zip(As, ys)]
    for r1 in recs:
        for r2 in recs:
            assert torch.allclose(r1, r2, atol=1e-2, rtol=1e-5)


def test_fdk_off_center_cor_subsets():
    """Test that fdk handles non-standard center of rotations correctly

    We rotate around a non-standard center of rotation and also check
    that the reconstruction of a subset equals the subset of a
    reconstruction.
    """

    vg = ts.volume(shape=64)
    pg = ts.cone_vec(
        shape=(96, 96),
        src_pos=[[3, -130, -10]],
        det_pos=[[0, 0, 0]],
        det_v=[[1, 0, 0]],
        det_u=[[0, 0, 1]],
    )
    pg = ts.cone(angles=1, shape=(96, 96), src_det_dist=130).to_vec()

    angles = np.linspace(0, 2 * np.pi, 32)
    R = ts.rotate(pos=(0, -20, 20), axis=(1, 0,  0), rad=angles)

    sub_slice = (slice(0, 32), slice(0, 32), slice(0, 32))
    vg_sub = vg[sub_slice]
    print(vg_sub.shape)
    A = ts.operator(vg, R.inv * pg)
    A_sub = ts.operator(vg_sub, R.inv * pg)

    x = make_box_phantom()
    y = A(x)
    bp = A.T(y)
    bp_sub = A_sub.T(y)
    print(bp.shape, bp_sub.shape, bp[sub_slice].shape)
    assert torch.allclose(bp[sub_slice], bp_sub, atol=1e-1, rtol=1e-6)

    r = fdk(A, y)
    r_sub = fdk(A_sub, y)
    assert torch.allclose(r[sub_slice], r_sub, atol=1e-1, rtol=1e-6)


def test_fdk_rotating_volume():
    """Test that fdk handles volume_vec geometries correctly

    Suppose we have
    - vg: volume geometry
    - pg: cone geometry  (single angle)
    - R: a rotating transform

    Then we must have that the following two operators are equal:

    ts.operator(vg, R * pg) == ts.operator(R.inv * vg, pg)

    in the sense that equal inputs yield equal outputs.

    Let's call these operators A1 (lhs) and A2 (rhs). Then we
    obviously want that

    fbp(A1, y) == fbp(A2, y).

    That is what we are testing here.
    """

    vg = ts.volume(shape=64).to_vec()
    pg = ts.cone(angles=1, shape=(96, 96), src_det_dist=130).to_vec()

    angles = np.linspace(0, 2 * np.pi, 30)
    R = ts.rotate(pos=0, axis=(1, 0, 0), rad=angles)

    A1 = ts.operator(vg, R * pg)
    A2 = ts.operator(R.inv * vg, pg)
    x = make_box_phantom()
    y1 = A1(x)
    y2 = A2(x)
    assert torch.allclose(y1, y2)
    bp1 = A1.T(y1)
    bp2 = A2.T(y2)
    assert torch.allclose(bp1, bp2)
    r1 = fdk(A1, y1)
    r2 = fdk(A2, y2)
    print(r1.mean(), r2.mean())
    assert torch.allclose(r1, r2)


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
    rec0 = fdk(A0, y)
    bp0 = A0.T(y)

    # Move volume and detector far away
    T = ts.translate((1000, 1000, 1000))
    A = ts.operator(T * vg, T * pg)

    rec = fdk(A, y)
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

    rec0 = fdk(A0, A0(x))
    bp0 = A0.T(A0(x))

    s = 2.0
    S = ts.scale(s)
    A = ts.operator(S * vg, S * pg)

    rec = fdk(A, A(x))
    bp = A.T(A(x))

    # Check that backprojection still matches (up to scaling)
    assert torch.allclose(bp / s**2, bp0)
    # Check that fdk still matches
    assert torch.allclose(rec, rec0)


def test_fdk_off_center():
    vg = ts.volume(shape=64, size=1)
    pg = ts.cone(angles=32, shape=(64, 64), size=(2, 2), src_det_dist=2, src_orig_dist=2)

    x = make_box_phantom()
    # Move volume slightly
    T1 = ts.translate((0, 0.2, 0))
    A1 = ts.operator(T1 * vg, pg)
    y1 = A1(x)
    rec1 = fdk(A1, y1)
    # Move detector slightly in opposite direction
    # resulting in the same relative positions
    y2 = A1(x)
    T2 = ts.translate((0, -0.2, 0))
    A2 = ts.operator(vg, T2 * pg.to_vec())
    rec2 = fdk(A2, y2)

    # rough test if reconstruction is close to original volume
    assert torch.mean(torch.abs(rec1 - x)) < 0.15
    assert torch.mean(torch.abs(rec2 - x)) < 0.15
    # test if reconstructions are the same
    assert torch.allclose(rec1, rec2, atol=1e-3, rtol=1e-2)
    assert torch.mean(torch.abs(rec1 - rec2)) < 1e-6


def test_fdk_center_dist_scaling():
    vg1 = ts.volume(shape=64, size=1).to_vec()
    pg1 = ts.cone(angles=32, shape=(64, 64), size=(4, 4), src_det_dist=4, src_orig_dist=2).to_vec()
    A1 = ts.operator(vg1, pg1)

    vg2 = ts.volume(shape=64, size=2).to_vec()
    pg2 = ts.cone(angles=32, shape=(64, 64), size=(8, 8), src_det_dist=8, src_orig_dist=4).to_vec()
    A2 = ts.operator(vg2, pg2)

    x = make_box_phantom()
    y1 = A1(x)
    rec1 = fdk(A1, y1)
    y2 = A2(x)
    rec2 = fdk(A2, y2)

    print(f"mean y1 = {torch.mean(y1)}, mean y2 = {torch.mean(y2)}, mean x = {torch.mean(x)}")

    # rough test if reconstruction is close to original volume
    assert torch.mean(torch.abs(rec1 - x)) < 0.15
    assert torch.mean(torch.abs(rec2 - x)) < 0.15
    # test if reconstructions are the same
    assert torch.allclose(rec1, rec2)
