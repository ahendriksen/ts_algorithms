import pytest
import torch
import tomosipo as ts
from ts_algorithms import fbp
import numpy as np


def make_box_phantom():
    x = torch.zeros((64, 64, 64))
    x[8:56, 8:56, 8:56] = 1.0
    x[24:44, 24:44, 24:44] = 0.0
    return x


def test_fbp():
    vg = ts.volume(shape=32)
    pg = ts.parallel(angles=32, shape=48)
    A = ts.operator(vg, pg)

    x = torch.zeros(*A.domain_shape)
    x[4:28, 4:28, 4:28] = 1.0
    x[12:22, 12:22, 12:22] = 0.0

    y = A(x)

    rec = fbp(A, y)
    rec = fbp(A, y, padded=False)


def test_fbp_odd_width():
    vg = ts.volume(shape=32)
    # Test also with a projection geometry with *49* pixels:
    pg = ts.parallel(angles=32, shape=49)

    A = ts.operator(vg, pg)

    x = torch.ones(*A.domain_shape)
    y = A(x)

    # Make sure this code runs without errors:
    rec = fbp(A, y)
    rec = fbp(A, y, padded=False)


def test_fbp_filter_error_handling():
    vg = ts.volume(shape=32)
    pg = ts.parallel(angles=32, shape=48)
    A = ts.operator(vg, pg)
    y = torch.ones(*A.range_shape)

    # Unpadded
    fbp(A, y, padded=False, filter=torch.ones(y.shape[-1]))
    with pytest.raises(ValueError):
        fbp(A, y, padded=False, filter=torch.ones(y.shape[-1] + 1))

    # Padded
    fbp(A, y, padded=True, filter=torch.ones(2 * y.shape[-1]))
    with pytest.raises(ValueError):
        fbp(A, y, padded=True, filter=torch.ones(y.shape[-1] + 1))


def test_fbp_rotating_volume():
    """Test that fbp handles volume_vec geometries correctly

    Suppose we have
    - vg: volume geometry
    - pg: projection geometry  (single angle)
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
    pg = ts.parallel(angles=1, shape=(64, 80)).to_vec()

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
    r1 = fbp(A1, y1)
    r2 = fbp(A2, y2)
    print(r1.mean(), r2.mean())
    assert torch.allclose(r1, r2)


def test_fbp_devices():
    vg = ts.volume(shape=32)
    pg = ts.parallel(angles=32, shape=48)
    A = ts.operator(vg, pg)

    x = torch.zeros(*A.domain_shape)
    x[4:28, 4:28, 4:28] = 1.0
    x[12:22, 12:22, 12:22] = 0.0

    y = A(x)

    devices = [torch.device("cpu"), torch.device("cuda")]
    for dev in devices:

        fbp(A, y.to(dev))
        fbp(A, y.to(dev), padded=False)


def test_fbp_overwrite_y():
    vg = ts.volume(shape=64).to_vec()
    pg = ts.parallel(angles=32, shape=(64, 80)).to_vec()

    # Create sinogram and reconstruct with and without overwrite_y
    A = ts.operator(vg, pg)
    x = make_box_phantom()
    y = A(x)
    rec1 = fbp(A, y, overwrite_y=False)
    rec2 = fbp(A, y, overwrite_y=True)

    # Check that reconstructions with overwrite_y is the same for True and False
    assert torch.allclose(rec1, rec2)


def test_fbp_batch_size():
    vg = ts.volume(shape=64).to_vec()
    pg = ts.parallel(angles=32, shape=(64, 80)).to_vec()

    # Create sinogram and reconstruct with batch size 1 and 32
    A = ts.operator(vg, pg)
    x = make_box_phantom()
    y = A(x)
    rec1 = fbp(A, y, batch_size=1)
    rec2 = fbp(A, y, batch_size=32)

    # Check that reconstructions with different batch sizes are the same
    assert torch.allclose(rec1, rec2)
