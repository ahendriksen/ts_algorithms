import pytest
import torch
import tomosipo as ts
import ts_algorithms.tv_min as tm
import ts_algorithms.operators as op


def test_scale_property():
    vg = ts.volume(shape=(1, 64, 64))
    pg = ts.parallel(angles=96, shape=(1, 96))
    A = ts.operator(vg, pg)

    x = torch.zeros(*A.domain_shape).cuda()
    x[:, 20:50, 20:50] = 1.0  # box
    x[:, 30:40, 30:40] = 0.0  # and hollow

    y = A(x)
    bp = A.T(y)

    for s in [1.0, 0.5, 2.0, 1/3, 3.0, 4.0]:
        print(s)
        S = ts.scale(s)
        A_s = ts.operator(
            S * A.domain,
            S * A.range.to_vec())

        # Relatively large tolerances, because
        # 1. converting to vector geometry incurs a loss;
        # 2. odd-numbered scalings incur heavy floating point inaccuracies
        assert torch.allclose(y * s, A_s(x), atol=1e-4, rtol=1e-3)
        assert torch.allclose(bp, A_s.T(y / s), atol=1e-4, rtol=1e-3)


def test_scale_property_exact():
    # Here, we test the same scaling property, but for a subset of
    # cases, which enables checks on the numerical equality of the
    # results with smaller tolerances. Specifically,
    # 1. We convert pg to a vector geometry immediately
    # 2. We only scale with "even" fractions (representable in base 2)
    # 3. We check with torch default tolerances rtol=1e-5, atol=1e-8

    vg = ts.volume(shape=(1, 64, 64))
    pg = ts.parallel(angles=96, shape=(1, 96)).to_vec()  # <-- vec already
    A = ts.operator(vg, pg)

    x = torch.zeros(*A.domain_shape).cuda()
    x[:, 20:50, 20:50] = 1.0  # box
    x[:, 30:40, 30:40] = 0.0  # and hollow

    y = A(x)
    bp = A.T(y)

    for s in [1.0, 0.5, 2.0, 4.0]:  # only "even" floating point numbers
        print(s)
        S = ts.scale(s)
        A_s = ts.operator(S * A.domain, S * A.range)

        assert torch.allclose(y * s, A_s(x))     # smaller tolerances
        assert torch.allclose(bp, A_s.T(y / s))  # smaller tolerances


def test_devices():
    vg = ts.volume(shape=(1, 64, 64))
    pg = ts.parallel(angles=96, shape=(1, 96))
    A = ts.operator(vg, pg)

    x = torch.zeros(*A.domain_shape)
    x[:, 20:50, 20:50] = 1.0  # box
    x[:, 30:40, 30:40] = 0.0  # and hollow

    y = A(x)

    tm.tv_min2d(A, y.cuda(), 0.1, num_iterations=10)
    tm.tv_min2d(A, y, 0.1, num_iterations=10)
