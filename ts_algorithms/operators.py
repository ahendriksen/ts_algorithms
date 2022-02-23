import torch
import math


def operator_norm(A, num_iter=10):
    x = torch.randn(A.domain_shape)
    for i in range(num_iter):
        x = A.T(A(x))
        x /= torch.norm(x)  # L2 vector-norm

    norm_ATA = (torch.norm(A.T(A(x))) / torch.norm(x)).item()
    return math.sqrt(norm_ATA)


def ATA_max_eigenvalue(A, num_iterations=10, stop_ratio=1, volume_mask=None, projection_mask=None):
    """Calculate the largest eigenvalue of A.T*A given a Tomosipo operator A
    
    The estimate is improved iteratively [1], and for each iteration both a
    lower and an upper bound for the largest eigenvalue are calculated. When
    num_iterations iterations have been executed or upper_bound/lower_bound
    <= stop_ratio, the algorithm is stopped and both the lower and the upper
    bound estimates are returned.
    
    Disclaimer: The method requires A.T*A to be irreducible, primitive and
    non-negative. The only property I'm sure of is non-negativity, but the
    method seems to work well for all operators I've tried it with.
    
    [1] Wood, R. J., & O'Neill, M. J. (2003). An always convergent method for
    finding the spectral radius of an irreducible non-negative matrix. ANZIAM
    Journal, 45, C474-C485.
    
    :param A: `tomosipo.Operator`
        Projection operator
    :param num_iterations: `int`
        Number of iterations
    :param stop_ratio: `float`
        If upper_bound/lower_bound <= stop_ratio the algorithm is stopped
    :param volume_mask: `torch.Tensor`
        Mask for the reconstruction volume. All voxels outside of the mask will
        be assumed to not contribute to the projection data.
        Setting to None will result in using the whole volume.
    :param projection_mask: `torch.Tensor`
        Mask for the projection data. All pixels outside of the mask will
        be assumed to not contribute to the reconstruction.
        Setting to None will result in using the whole projection data.
    :returns: `(float, float)`
        lower and upper bounds on the largest eigenvalue of A.T*A
    """
    x = torch.ones(A.domain_shape, dtype=torch.float32)
    if volume_mask is not None:
            x *= volume_mask
    x_next = torch.zeros(A.domain_shape, dtype=torch.float32)
    y_tmp = torch.zeros(A.range_shape, dtype=torch.float32)
    
    for i in range(num_iterations):
        A(x, out=y_tmp)
        if projection_mask is not None:
            y_tmp *= projection_mask
        A.T(y_tmp, out=x_next)
        if volume_mask is not None:
            x_next *= volume_mask
        
        ratio = x_next/x
        lower_bound = torch.min(torch.nan_to_num(ratio, nan=float("inf")))
        upper_bound = torch.max(torch.nan_to_num(ratio, nan=float("-inf")))
        x, x_next = x_next, x
        x /= (upper_bound + lower_bound)/2
        
        if upper_bound/lower_bound <= stop_ratio:
            break
    
    return lower_bound, upper_bound
