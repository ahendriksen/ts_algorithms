import tomosipo as ts
import torch
import math


def sirt(A, y, num_iterations, min_constraint=None, max_constraint=None):
    """Execute the SIRT algorithm

    If `y` is located on GPU, the entire algorithm is executed on a single GPU.

    IF `y` is located in RAM (CPU in PyTorch parlance), then only the
    foward and backprojection are executed on GPU.

    :param A: `tomosipo.operator`
    :param y: `torch.tensor`
    :param num_iterations: `int`
    :param min_constraint: `float`
        Minimum value enforced at each iteration. Setting to None skips this step.
    :param max_constraint: `float`
        Maximum value enforced at each iteration. Setting to None skips this step.

    :returns:
    :rtype:

    """
    dev = y.device

    # Compute C
    y_tmp = torch.ones(A.range_shape, device=dev)
    C = A.T(y_tmp)
    C[C < ts.epsilon] = math.inf
    C.reciprocal_()
    # Compute R
    x_tmp = torch.ones(A.domain_shape, device=dev)
    R = A(x_tmp)
    R[R < ts.epsilon] = math.inf
    R.reciprocal_()

    x_cur = torch.zeros(A.domain_shape, device=dev)

    for _ in range(num_iterations):
        A(x_cur, out=y_tmp)
        y_tmp -= y
        y_tmp *= R
        A.T(y_tmp, out=x_tmp)
        x_tmp *= C
        x_cur -= x_tmp
        if min_constraint or max_constraint:
            x_cur.clamp_(min_constraint, max_constraint)

    return x_cur
