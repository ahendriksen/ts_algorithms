"""An implementatation of Total-Variation minimization using Chambolle-Pock

Described in more detail at:

https://blog.allardhendriksen.nl/cwi-ci-group/chambolle_pock_using_tomosipo/

Based on the article:

Sidky, Emil Y, Jakob H Jørgensen, and Xiaochuan Pan. 2012. “Convex
Optimization Problem Prototyping for Image Reconstruction in Computed
Tomography with the Chambolle-Pock Algorithm". Physics in Medicine and
Biology 57 (10). IOP
Publishing:3065–91. https://doi.org/10.1088/0031-9155/57/10/3065.

"""

import tomosipo as ts
import torch
import math
from .operators import operator_norm


def grad_2D(x):
    weight = x.new_zeros(2, 1, 3, 3)
    weight[0, 0] = torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
    weight[1, 0] = torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    x = x[:, None]              # Add channel dimension
    out = torch.conv2d(x, weight, padding=1)
    return out[:, :, :, :]


def grad_2D_T(y):
    weight = y.new_zeros(2, 1, 3, 3)
    weight[0, 0] = torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
    weight[1, 0] = torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    out = torch.conv_transpose2d(y, weight, padding=1)
    return out[:, 0, :, :]      # Remove channel dimension


def operator_norm_plus_grad(A, num_iter=10):
    x = torch.randn(A.domain_shape)
    operator_norm_estimate = 0.0
    for i in range(num_iter):
        y_A = A(x)
        y_TV = grad_2D(x)
        x_new = A.T(y_A) + grad_2D_T(y_TV)
        operator_norm_estimate = torch.norm(x_new) / torch.norm(x)
        x = x_new / torch.norm(x_new)

    norm_ATA = operator_norm_estimate.item()
    return math.sqrt(norm_ATA)


def magnitude(z):
    return torch.sqrt(z[:, 0:1] ** 2 + z[:, 1:2] ** 2)


def clip(z, lamb):
    return z * torch.clamp(lamb / magnitude(z), min=None, max=1.0)


def tv_min2d(A, y, lam, num_iterations=500, L=None, non_negativity=False):
    """Computes the total-variation minimization using Chambolle-Pock

    Assumes that the data is a single 2D slice. A 3D version with 3D
    gradients is work in progress.

    :param A: `tomosipo.operator`
    :param y: `torch.tensor`
    :param lam: `float`
        regularization parameter lambda.
    :param num_iterations: `int`
    :returns:
    :rtype:

    """

    dev = y.device

    # It is preferable that the operator norm of `A` is roughly equal
    # to one. First, this makes the `lam` parameter comparable between
    # different geometries. Second, without this trick I cannot get
    # the algorithm to converge.

    # The operator norm of `A` scales with the scale of the
    # geometry. Therefore, it is easiest to rescale the geometry and
    # to divide the measurement y by the scale to preserve the
    # intensity. The validity of this appraoch is checked in the
    # tests, see `test_scale_property` in test_tv_min.py.
    scale = operator_norm(A)
    S = ts.scale(1 / scale, pos=A.volume_geometry.pos)
    A = ts.operator(S * A.volume_geometry, S * A.projection_geometry.to_vec())
    y = y / scale

    if L is None:
        L = operator_norm_plus_grad(A, num_iter=100)
    t = 1.0 / L
    s = 1.0 / L
    theta = 1

    u = torch.zeros(A.domain_shape, device=dev)
    p = torch.zeros(A.range_shape, device=dev)
    q = grad_2D(u)                  # contains zeros (and has correct shape)
    u_avg = torch.clone(u)

    for n in range(num_iterations):
        p = (p + s * (A(u_avg) - y)) / (1 + s)
        q = clip(q + s * grad_2D(u_avg), lam)
        u_new = u - (t * A.T(p) + t * grad_2D_T(q))
        if non_negativity:
            u_new = torch.clamp(u_new, min=0.0, max=None)
        u_avg = u_new + theta * (u_new - u)
        u = u_new

    return u
