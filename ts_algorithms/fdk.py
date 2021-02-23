import numpy as np
import torch
import tomosipo as ts
from .fbp import fbp


def fdk_weigh_projections(op, projections, src_rot_center_dist):
    # The distance of each pixel to the source is calulated in a coordinate
    # system based on the normalized u, v and normal vectors of the detector
    # because that way the distance along the v axis is constant along each
    # row and the distance along the u axis is constant along each column
    # resulting in fewer computations
    
    src_to_det = torch.from_numpy(op.range.det_pos[0] - op.range.src_pos[0])
    src_det_dist = torch.norm(src_to_det)
    # Load u, v vectors:
    det_u = torch.from_numpy(op.range.det_u[0])
    det_v = torch.from_numpy(op.range.det_v[0])
    # Calculate u, v vector lengths:
    det_u_norm = torch.norm(det_u)
    det_v_norm = torch.norm(det_v)
    # Transform source to detector center vector to the u, v system
    src_det_u_dist = torch.dot(src_to_det, det_u) / det_u_norm
    src_det_v_dist = torch.dot(src_to_det, det_v) / det_v_norm

    #calculate the distance of each row and column of pixels along the u or v axis
    u_num_pixels = op.range_shape[2]
    u_space = (torch.arange(u_num_pixels, dtype=torch.float64)+0.5-(u_num_pixels/2))
    v_num_pixels = op.range_shape[0]
    v_space = (torch.arange(v_num_pixels, dtype=torch.float64)+0.5-(v_num_pixels/2))    
    
    # Calculate the minimal source to detector distance divided by the
    # pixel to detector distance for each pixel. The pixel to detector 
    # distances are calculated using the pythagorean theorem
    u_pos_squared = (u_space * det_u_norm + src_det_u_dist)**2
    v_pos_squared = (v_space * det_v_norm + src_det_v_dist)**2

    weights_mat = src_det_dist / torch.sqrt(
          u_pos_squared[None, :].expand([v_num_pixels, u_num_pixels])
          + v_pos_squared[:, None].expand([v_num_pixels, u_num_pixels])
          + src_det_dist**2
    )
    
    # Multiply with extra scaling factor to account for detector distance
    weights_mat *= (src_rot_center_dist / src_det_dist)
    weights_mat = weights_mat.float().to(projections.device)

    result = projections.new_empty(projections.shape).to(projections.device)

    for i in range(op.range_shape[1]):
        result[:, i, :] = projections[:, i, :] * weights_mat

    return result


def fdk(A, y, padded=True, filter=None, reject_acyclic_filter=True, src_rot_center_dist=None):
    """Approximately reconstruct volumes in a circular cone beam geometry using
    the Feldkamp, Davis and Kress(FDK) algorithm [1]. Arbtrary shifts, scaling
    and rotations along the vertical axis are also supported.

    If `y` is located on GPU, the entire algorithm is executed on a single GPU.

    If `y` is located in RAM (CPU in PyTorch parlance), then only the
    foward and backprojection are executed on GPU.

    :param A: `tomosipo.operator` The projection geometry of the operator must
    be either cone or cone_vec. If it is cone_vec parameter src_rot_center_dist
    must be provided too
    :param y: `torch.tensor` sinogram stack with the following layout:
    (num_vertical_pixels, num_angles, num_horizontal_pixels)
    :param padded: bool, is passed to ts_algorithms.fbp
    :param filter: bool, is passed to ts_algorithms.fbp
    :param reject_acyclic_filter: bool, is passed to ts_algorithms.fbp
    :param src_rot_center_dist: the distance from the source to the
    center of rotation of the object. Only used and required when the
    projection geometry is cone_vec
    :returns: reconstruction of a volume
    :rtype: `torch.tensor`
    
    [1] Feldkamp, L. A., Davis, L. C., & Kress, J. W. (1984). Practical Cone-Beam
    Algorithm. Journal of the Optical Society of America A, 1(6), 612.
    http://dx.doi.org/10.1364/josaa.1.000612
    """
    voxel_sizes = np.array(A.volume_geometry.size) / np.array(A.volume_geometry.shape)
    if np.ptp(voxel_sizes) > ts.epsilon:
        raise ValueError(
            "The voxels in the volume are required to have the same size in every dimension.\n"
            f"This is not the case: voxel size = {voxel_sizes}."
        )
    pg = A.range
    if isinstance(pg, ts.geometry.cone_vec.ConeVectorGeometry):
        if src_rot_center_dist is None:
            raise ValueError(
                "When pg is a cone vector geometry parameter src_rot_center_dist needs to be provided"
            )
    elif isinstance(pg, ts.geometry.cone.ConeGeometry):
        src_rot_center_dist = pg.src_orig_dist
    else:
        raise ValueError(
            "The provided operator A needs to describe a cone beam geometry."
        )
        
    return fbp(
        A=A,
        y=fdk_weigh_projections(A, y, src_rot_center_dist),
        padded=padded,
        filter=filter,
        reject_acyclic_filter=reject_acyclic_filter
    )
