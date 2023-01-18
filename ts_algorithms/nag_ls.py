import warnings
import tomosipo as ts
import torch
import math
import tqdm
import time
from .callbacks import call_all_callbacks
from .operators import ATA_max_eigenvalue


def nag_ls(A, y, num_iterations, max_eigen=None, min_eigen=0, l2_regularization=0,
    min_constraint=None, max_constraint=None, x_init=None,
    volume_mask=None, projection_mask=None, progress_bar=False, callbacks=()):
    """Apply nesterov accelerated gradient descent (nag) [1](chapter 2.2) to
       solve the (possibly l2-regularized) least squares (ls) problem:
       minimize over x: ||Ax - y||^2 + l2_regularization * ||x||^2

    If `y` is located on GPU, the entire algorithm is executed on a single GPU.
    If `y` is located in RAM (CPU in PyTorch parlance), then only the
    foward and backprojection are executed on GPU.
    
    [1] Nesterov, Y. (2003). Introductory lectures on convex optimization: A
    basic course (Vol. 87). Springer Science & Business Media.
    [2] https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/

    :param A: `tomosipo.Operator`
        Projection operator
    :param y: `torch.Tensor`
        Projection data
    :param num_iterations: `int`
        Number of iterations
    :param max_eigen: `float`
        largest eigenvalue of A.T*A / spectral radius of A.T*A / Lipschitz
        constant of the gradient of the least squares problem not considering
        regularization. Can be calculated using ts.algorithms.ATA_max_eigenvalue
    :param min_eigen: `float`
        *Currently unused* Smallest eigenvalue of A.T*A / strict convexity
        parameter not considering regularization
    :param l2_regularization: `float`
        amount of l2-regularization
    :param min_constraint: `float`
        Minimum value enforced at each iteration. Setting to None skips this step.
    :param max_constraint: `float`
        Maximum value enforced at each iteration. Setting to None skips this step.
    :param x_init: `torch.Tensor`
        Initial value for the solution. Setting to None will start with zeros.
        This can be used to improve upon a previously found solution. However,
        the momentum parameters will be re-initialized every time this
        function is called, so contrary to SIRT calling this function twice for
        10 iterations will not give the same result as calling this function
        once for 20 iterations.
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
        A reconstruction of the volume using num_iterations iterations of
        Nesterov accelerated gradient descent.
    :rtype:

    """
    
    # Calculate the maximum eigenvalue of A.T*A if it wasn't provided and show
    # a warning when this takes more than 10 seconds
    if max_eigen is None:
        warning_timeout = 10
        start_time = time.time()
        _, max_eigen = ATA_max_eigenvalue(A, num_iterations=50, stop_ratio=1.01)
        calculation_time = time.time() - start_time
        
        if calculation_time >= warning_timeout:
            warnings.warn(
            "The max_eigen parameter was not set when calling the nag_ls"
            " function, so it had to be calculated, which took"
            f" {calculation_time:.1f} seconds. Calculate the maximum"
            " eigenvalue once for your geometry with"
            " ts_algorithms.ATA_max_eigenvalue and then store it to skip this"
            " step when using nag_ls on the same geometry multiple times. \n"
            f" The calulated max_eigen parameter was {max_eigen}"
        )
    
    L = 2*(max_eigen + l2_regularization)   # Lipschitz constant of the gradient

    # Allocate memory on the right device for all used vectors
    dev = y.device
    y_tmp = torch.zeros(A.range_shape, device=dev, dtype=torch.float32)
    if x_init is None:
        x_cur = torch.zeros(A.domain_shape, device=dev, dtype=torch.float32)
    else:
        with torch.cuda.device_of(y):
            x_cur = x_init.clone()
    x_prev = x_cur.clone()
    z = torch.zeros(A.domain_shape, device=dev, dtype=torch.float32)
    z_half_grad = torch.zeros(A.domain_shape, device=dev, dtype=torch.float32)


    # Initialize the variables related to the step size 
    step_lambda = 0
    next_step_lambda = (1 + math.sqrt(1 + 4 * step_lambda * step_lambda)) / 2


    # Main loop of the algorithm
    # tqdm is used to show a progress bar
    for iteration in tqdm.trange(num_iterations, disable=not progress_bar):
        # take a step in the direction of the gradient
        A(z, out=y_tmp)
        y_tmp -= y
        if projection_mask is not None:
            y_tmp *= projection_mask
        A.T(y_tmp, out=z_half_grad)
        z_half_grad += l2_regularization * z
        if volume_mask is not None:
            z_half_grad *= volume_mask
        
        x_cur, x_prev = x_prev, x_cur
        x_cur[...] = z - (2/L) * z_half_grad
        if (min_constraint is not None) or (max_constraint is not None):
            x_cur.clamp_(min_constraint, max_constraint)
        
        # Apply momentum
        step_lambda = next_step_lambda
        next_step_lambda = (1 + math.sqrt(1 + 4 * step_lambda * step_lambda)) / 2
        step_gamma = (1 - step_lambda) / next_step_lambda
        z[...] = ((1 - step_gamma) * x_cur) + (step_gamma * x_prev)
            
        # Call all callbacks and stop iterating if one of the callbacks
        # indicates to stop
        if call_all_callbacks(callbacks, x_cur, iteration):
            break
            
    return x_cur
    
