import torch
import math


def operator_norm(A, num_iter=10):
    x = torch.randn(A.domain_shape)
    for i in range(num_iter):
        x = A.T(A(x))
        x /= torch.norm(x)  # L2 vector-norm

    norm_ATA = (torch.norm(A.T(A(x))) / torch.norm(x)).item()
    return math.sqrt(norm_ATA)
