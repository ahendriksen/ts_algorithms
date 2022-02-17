import tomosipo as ts
import torch
import math
import tqdm


def ATA_max_eigenvalue(A, stop_iterations=10, stop_ratio=1):
    """Calculate the largest eigenvalue of A.T*A given a Tomosipo operator A
    
    The estimate is improved iteratively [1], and for each iteration both a
    lower and an upper bound for the largest eigenvalue are calculated. When
    stop_iterations iterations have been executed or upper_bound/lower_bound
    <= stop_ratio, the algorithm is stopped and both the lower and the upper
    bound estimates are returned.
    
    Disclaimer: The method requires A.T*A to be irreducible, primitive and
    non-negative. The only property I'm sure of is non-negativity, but the
    method seems to work well for all operators I've tried it with.
    
    [1] Wood, R. J., & O'Neill, M. J. (2003). An always convergent method for
    finding the spectral radius of an irreducible non-negative matrix. ANZIAM
    Journal, 45, C474-C485.
    """
    x = torch.ones(A.domain_shape)
    x_next = torch.zeros(A.domain_shape)
    y_tmp = torch.zeros(A.range_shape)
    
    for i in range(stop_iterations):
        A(x, out=y_tmp)
        A.T(y_tmp, out=x_next)
        
        ratio = x_next/x
        lower_bound = torch.min(ratio)
        upper_bound = torch.max(ratio)
        x, x_next = x_next, x
        x /= (upper_bound + lower_bound)/2
        
        if upper_bound/lower_bound <= stop_ratio:
            break
    
    return lower_bound, upper_bound, x


def nag_ls(A, y, num_iterations, max_eigen, min_eigen=0, l2_regularization=0,
    min_constraint=None, max_constraint=None, x_init=None,
    volume_mask=None, projection_mask=None, progress_bar=False):
    """Apply nesterov accelerated gradient descent (nag) [1](chapter 2.2) to
       solve the (possibly l2-regularized) least squares (ls) problem:
       minimize over x: ||Ax - y||^2 + l2_regularization * ||x||^2
    
    Which variant of the algorithm is used depends on whether the regularized
    problem is strictly convex or not. When both min_eigen and 
    l2_regularization are 0 a non-strictly convex version [2] of nag is used.
    Otherwise a strictly convex version [3] is used.

    If `y` is located on GPU, the entire algorithm is executed on a single GPU.
    If `y` is located in RAM (CPU in PyTorch parlance), then only the
    foward and backprojection are executed on GPU.
    
    [1] Nesterov, Y. (2003). Introductory lectures on convex optimization: A
    basic course (Vol. 87). Springer Science & Business Media.
    [2] https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/
    [3] https://blogs.princeton.edu/imabandit/2014/03/06/nesterovs-accelerated-gradient-descent-for-smooth-and-strongly-convex-optimization/

    :param A: `tomosipo.Operator`
        Projection operator
    :param y: `torch.Tensor`
        Projection data
    :param num_iterations: `int`
        Number of iterations
    :param max_eigen: `float`
        largest eigenvalue of A.T*A / spectral radius of A.T*A / Lipschitz
        constant of the gradient of the least squares problem not considering
        regularization
    :param min_eigen: `float`
        Smallest eigenvalue of A.T*A / strict convexity parameter not
        considering regularization
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
    :returns: `torch.Tensor`
        A reconstruction of the volume using num_iterations iterations of
        Nesterov accelerated gradient descent.
    :rtype:

    """
    
    L = 2*(max_eigen + l2_regularization)   # Lipschitz constant of the gradient
    mu = 2*(min_eigen + l2_regularization)  # (strong) convexity parameter
    
    dev = y.device
    residual = torch.zeros(A.range_shape, device=dev)
    if x_init is None:
        x_cur = torch.zeros(A.domain_shape, device=dev)
    else:
        with torch.cuda.device_of(y):
            x_cur = x_init.clone()
    x_prev = x_cur.clone()
    z = torch.zeros(A.domain_shape, device=dev)
    z_half_grad = torch.zeros(A.domain_shape, device=dev)

    if mu == 0:
        step_lambda = 0
        next_step_lambda = (1 + math.sqrt(1 + 4 * step_lambda * step_lambda)) / 2
    else:
        Q_sqrt = math.sqrt(L/mu)
        frac = (Q_sqrt - 1) / (Q_sqrt + 1)
        
    for _ in tqdm.trange(num_iterations, disable=not progress_bar):
        A(z, out=residual)
        residual -= y
        if projection_mask is not None:
            residual *= projection_mask
        A.T(residual, out=z_half_grad)
        z_half_grad += l2_regularization * z
        if volume_mask is not None:
            z_half_grad *= volume_mask
        
        x_cur, x_prev = x_prev, x_cur
        x_cur[...] = z - (2/L) * z_half_grad
        if (min_constraint is not None) or (max_constraint is not None):
            x_cur.clamp_(min_constraint, max_constraint)
        
        if mu == 0:
            step_lambda = next_step_lambda
            next_step_lambda = (1 + math.sqrt(1 + 4 * step_lambda * step_lambda)) / 2
            step_gamma = (1 - step_lambda) / next_step_lambda
            z[...] = ((1 - step_gamma) * x_cur) + (step_gamma * x_prev)
        else:
            z[...] = ((1 + frac) * x_cur) - (frac * x_prev)
            
    return x_cur
    
