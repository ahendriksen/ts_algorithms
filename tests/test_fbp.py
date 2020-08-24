import pytest
import torch
import tomosipo as ts
from ts_algorithms import fbp


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
