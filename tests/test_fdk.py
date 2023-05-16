import pytest
import torch
import tomosipo as ts
from ts_algorithms import fdk
from ts_algorithms.fdk import fdk_weigh_projections
import numpy as np


def make_box_phantom():
    x = torch.zeros((64, 64, 64))
    x[8:56, 8:56, 8:56] = 1.0
    x[24:44, 24:44, 24:44] = 0.0
    return x


def astra_fdk(A, y):
    vg, pg = A.astra_compat_vg, A.astra_compat_pg

    vd = ts.data(vg)
    pd = ts.data(pg, y.cpu().numpy())
    ts.astra.fdk(vd, pd)
    # XXX: disgregard clean up of vd and pd (tests are short-lived and
    # small)
    return torch.from_numpy(vd.data.copy()).to(y.device)


# Standard parameters
vg64 = [
    ts.volume(shape=64),          # voxel size == 1
    ts.volume(shape=64, size=1),  # volume size == 1
    ts.volume(shape=64, size=1),  # idem
]
pg64 = [
    ts.cone(angles=96, shape=96, src_det_dist=192),  # pixel size == 1
    ts.cone(angles=96, shape=96, size=1.5, src_det_dist=3),  # detector size == 1 / 64
    ts.cone(angles=96, shape=96, size=3, src_det_dist=3, src_orig_dist=3),  # magnification 2
]
phantom64 = [
    make_box_phantom(),
    make_box_phantom(),
    make_box_phantom(),
]

@pytest.mark.parametrize("vg, pg, x", zip(vg64, pg64, phantom64))
def test_astra_compatibility(vg, pg, x):
    A = ts.operator(vg, pg)
    y = A(x)
    rec_ts = fdk(A, y)
    rec_astra = astra_fdk(A, y)

    print(abs(rec_ts - rec_astra).max())
    assert torch.allclose(rec_ts, rec_astra, atol=5e-4)


def test_fdk_flipped_cone_geometry():
    vg = ts.volume(shape=64)
    angles = np.linspace(0, 2 * np.pi, 96)
    R = ts.rotate(pos=0, axis=(1, 0, 0), angles=angles)
    pg = ts.cone_vec(
            shape=(96, 96),
            src_pos=[[0, 130, 0]],  # usually -130
            det_pos=[[0, 0, 0]],
            det_v=[[1, 0, 0]],
            det_u=[[0, 0, 1]],
    )
    A = ts.operator(vg, R * pg)

    fdk(A, torch.ones(A.range_shape))


@pytest.mark.parametrize("vg, pg, x", zip(vg64, pg64, phantom64))
def test_fdk_inverse(vg, pg, x):
    """Rough test if reconstruction is close to original volume.

    The mean error must be less than 10%. The sharp edges of the box
    phantom make this a difficult test case.
    """
    A = ts.operator(vg, pg)
    y = A(x)

    rec = fdk(A, y)
    assert torch.mean(torch.abs(rec - x)) < 0.1


@pytest.mark.parametrize("vg, pg, x", zip(vg64, pg64, phantom64))
def test_fdk_cone_vec(vg, pg, x):
    """ Test that cone and cone_vec yield same result."""
    A = ts.operator(vg, pg)
    A_vec = ts.operator(vg, pg.to_vec())
    y = A(x)

    rec = fdk(A, y)
    rec_vec = fdk(A_vec, y)
    assert torch.allclose(rec, rec_vec, atol=5e-4)


@pytest.mark.parametrize("vg, pg, x", zip(vg64, pg64, phantom64))
def test_fdk_gpu(vg, pg, x):
    """ Test that cuda and cpu tensors yield same result."""
    A = ts.operator(vg, pg)
    y = A(x)

    rec_cpu = fdk(A, y)
    rec_cuda = fdk(A, y.cuda()).cpu()

    # The atol is necessary because the ASTRA backprojection appears
    # to differ slightly when given cpu and gpu arguments...
    assert torch.allclose(rec_cpu, rec_cuda, atol=5e-4)


def test_fdk_off_center_cor():
    """Test that fdk handles non-standard center of rotations correctly
    """

    vg = ts.volume(shape=64).to_vec()
    pg = ts.cone(angles=1, shape=(96, 96), src_det_dist=130).to_vec()

    angles = np.linspace(0, 2 * np.pi, 30)
    R = ts.rotate(pos=(0, -20, -20), axis=(1, 0, 0), angles=angles)

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
    R = ts.rotate(pos=(0, 0, 0), axis=(1, 0, 0), angles=angles)

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
    R = ts.rotate(pos=(0, -20, 20), axis=(1, 0,  0), angles=angles)

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


@pytest.mark.parametrize("vg, pg, x", zip(vg64, pg64, phantom64))
def test_fdk_split_detector(vg, pg, x):
    """Split detector in four quarters

    Test that pre-weighting each quarter individually is the same as
    pre-weighting the full detector at once.
    """

    pg = pg.to_vec()

    # determine the half-length of the detector shape:
    n, m = np.array(pg.det_shape) // 2

    # Generate slices to split the detector of a projection geometry
    # into four slices.
    pg_slices = [
        np.s_[:, :n, :m],
        np.s_[:, :n, m:],
        np.s_[:, n:, :m],
        np.s_[:, n:, m:],
    ]
    # Change slices to be in 'sinogram' form with angles in the middle.
    sino_slices = [(slice_v, slice_angles, slice_u) for (slice_angles, slice_v, slice_u) in pg_slices]

    A = ts.operator(vg, pg)
    y = A(x)

    As = [ts.operator(vg, pg[pg_slice]) for pg_slice in pg_slices]

    w = fdk_weigh_projections(A, y, False)
    sub_ws = [fdk_weigh_projections(A_sub, y[sino_slice].contiguous(), False) for A_sub, sino_slice in zip(As, sino_slices)]

    for sub_w, sino_slice in zip(sub_ws, sino_slices):
        abs_diff = abs(w[sino_slice] - sub_w)
        print(sub_w.max(), abs_diff.max().item(), abs_diff.mean().item())
        assert torch.allclose(w[sino_slice], sub_w, rtol=1e-2)


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
    R = ts.rotate(pos=0, axis=(1, 0, 0), angles=angles)

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


def test_fdk_errors():
    # 0. uneven voxel sizes
    vg = ts.volume(shape=64, size=(128, 64, 64))
    pg = ts.cone(angles=1, shape=64, src_det_dist=128)
    A = ts.operator(vg, pg)
    with pytest.raises(ValueError):
        fdk(A, torch.zeros(A.range_shape))

    # 1. uneven pixel sizes
    pg = ts.concatenate((
        ts.cone(shape=64, size=64, angles=32, src_det_dist=128),
        ts.cone(shape=64, size=128, angles=32, src_det_dist=128),
    ))
    vg = ts.volume(shape=64)
    A = ts.operator(vg, pg)
    with pytest.raises(ValueError):
        fdk(A, torch.zeros(A.range_shape))

    # 2. non-cone beam geometry
    A = ts.operator(vg, ts.parallel())
    with pytest.raises(TypeError):
        fdk(A, torch.zeros(A.range_shape))

    # 3. varying source-detector distance
    vg = ts.volume(shape=64)
    pg = ts.cone_vec(
        shape=(96, 96),
        src_pos=[[0, -100, 0], [0, -200, 0]],
        det_pos=[[0, 0, 0], [0, 0, 0]],
        det_v=[[1, 0, 0], [1, 0, 0]],
        det_u=[[0, 0, 1], [0, 0, 1]],
    )
    A = ts.operator(vg, pg)
    with pytest.warns(UserWarning):
        fdk(A, torch.zeros(A.range_shape))

    # 4. Rotation center behind source position
    vg = ts.volume(pos=(0, -64, 0), shape=64).to_vec()
    pg = ts.cone(shape=96, angles=1, src_det_dist=128).to_vec()
    angles = np.linspace(0, 2 * np.pi, 90)
    R = ts.rotate(pos=(0, -129, 0), axis=(1, 0, 0), angles=angles)

    A = ts.operator(R * vg, pg)

    with pytest.warns(UserWarning):
        fdk(A, torch.ones(A.range_shape))


def test_fdk_overwrite_y():
    vg = ts.volume(shape=64, size=1)
    pg = ts.cone(angles=32, shape=(64, 64), size=(2, 2), src_det_dist=2, src_orig_dist=2).to_vec()

    # Create sinogram and reconstruct with and without overwrite_y
    A = ts.operator(vg, pg)
    x = make_box_phantom()
    y = A(x)
    rec1 = fdk(A, y, overwrite_y=False)
    rec2 = fdk(A, y, overwrite_y=True)

    # Check that reconstructions with overwrite_y is the same for True and False
    assert torch.allclose(rec1, rec2)


def test_fdk_batch_size():
    vg = ts.volume(shape=64, size=1)
    pg = ts.cone(angles=32, shape=(64, 64), size=(2, 2), src_det_dist=2, src_orig_dist=2).to_vec()

    # Create sinogram and reconstruct with batch size 1 and 32
    A = ts.operator(vg, pg)
    x = make_box_phantom()
    y = A(x)
    rec1 = fdk(A, y, batch_size=1)
    rec2 = fdk(A, y, batch_size=32)

    # Check that reconstructions with different batch sizes are the same
    assert torch.allclose(rec1, rec2)
