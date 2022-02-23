import tomosipo as ts
import torch
import math
import tqdm
from .callbacks import call_all_callbacks


def sirt(A, y, num_iterations, min_constraint=None, max_constraint=None, x_init=None, volume_mask=None,
    projection_mask=None, progress_bar=False, callbacks=()):
    """Execute the SIRT algorithm

    If `y` is located on GPU, the entire algorithm is executed on a single GPU.

    IF `y` is located in RAM (CPU in PyTorch parlance), then only the
    foward and backprojection are executed on GPU.

    :param A: `tomosipo.Operator`
        Projection operator
    :param y: `torch.Tensor`
        Projection data
    :param num_iterations: `int`
        Number of iterations
    :param min_constraint: `float`
        Minimum value enforced at each iteration. Setting to None skips this step.
    :param max_constraint: `float`
        Maximum value enforced at each iteration. Setting to None skips this step.
    :param x_init: `torch.Tensor`
        Initial value for the solution. Setting to None will start with zeros.
        Setting x_init to a previously found solution can be useful to
        continue with more iterations of SIRT.
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
        A reconstruction of the volume using num_iterations iterations of SIRT
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
    
    if x_init is None:
        x_cur = torch.zeros(A.domain_shape, device=dev)
    else:
        with torch.cuda.device_of(y):
            x_cur = x_init.clone()

    if volume_mask is not None:
        x_cur *= volume_mask
        C *= volume_mask
        
    if projection_mask is not None:
        R *= projection_mask

    for iteration in tqdm.trange(num_iterations, disable=not progress_bar):
        A(x_cur, out=y_tmp)
        y_tmp -= y
        y_tmp *= R
        A.T(y_tmp, out=x_tmp)
        x_tmp *= C
        x_cur -= x_tmp
        if (min_constraint is not None) or (max_constraint is not None):
            x_cur.clamp_(min_constraint, max_constraint)
            
        # Call all callbacks and stop iterating if one of the callbacks
        # indicates to stop
        if call_all_callbacks(callbacks, x_cur, iteration):
            break

    return x_cur
