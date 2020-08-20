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
