import tomosipo as ts
import torch
import math
import tqdm
from .callbacks import call_all_callbacks


def em(A, y, num_iterations, x_init=None, volume_mask=None,
    projection_mask=None, progress_bar=False, callbacks=()):
    """Execute the (Maximum Likelihood) Expectation-Maximization algorithm

    If `y` is located on GPU, the entire algorithm is executed on a single GPU.

    IF `y` is located in RAM (CPU in PyTorch parlance), then only the
    forward and backprojection are executed on GPU.

    :param A: `tomosipo.Operator`
        Projection operator
    :param y: `torch.Tensor`
        Projection data
    :param num_iterations: `int`
        Number of iterations
    :param x_init: `torch.Tensor`
        Initial value for the solution. Setting to None will start with ones.
        Setting x_init to a previously found solution can be useful to
        continue with more iterations of EM.
    :param volume_mask: `torch.Tensor`
        Mask for the reconstruction volume. All voxels outside of the mask will
        be assumed to not contribute to the projection data.
        Setting to None will result in using the whole volume.
    :param projection_mask: `torch.Tensor`
        Mask for the projection data. All pixels outside of the mask will
        be assumed to not contribute to the reconstruction.
        Setting to None will result in using the whole projection data.
    :param progress_bar: `bool`
        Whether to show a progress bar on the command line interface.
        Default: False
    :param callbacks:
        Iterable containing functions or callable objects. Each callback will
        be called every iteration with the current estimate and iteration
        number as arguments. If any callback returns True, the algorithm stops
        after this iteration. This can be used for logging, tracking or
        alternative stopping conditions.
    :returns: `torch.Tensor`
        A reconstruction of the volume using num_iterations iterations of EM
    :rtype:

    """
    dev = y.device

    # Compute C
    y_tmp = torch.ones(A.range_shape, device=dev)
    C = A.T(y_tmp)
    C[C < ts.epsilon] = math.inf
    C.reciprocal_()

    if x_init is None:
        x_cur = torch.ones(A.domain_shape, device=dev)
    else:
        with torch.cuda.device_of(y):
            x_cur = x_init.clone()

    if volume_mask is not None:
        x_cur *= volume_mask

    x_tmp = torch.empty(A.domain_shape, device=dev)
    for iteration in tqdm.trange(num_iterations, disable=not progress_bar):
        A(x_cur, out=y_tmp)
        if projection_mask is not None:
            y_tmp *= projection_mask
        y_tmp[y_tmp < ts.epsilon] = math.inf
        torch.div(y, y_tmp, out=y_tmp)
        A.T(y_tmp, out=x_tmp)
        x_cur *= x_tmp
        x_cur *= C

        # Call all callbacks and stop iterating if one of the callbacks
        # indicates to stop
        if call_all_callbacks(callbacks, x_cur, iteration):
            break

    return x_cur
