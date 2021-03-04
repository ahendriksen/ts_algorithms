import numpy as np
import torch
import tomosipo as ts
from .fbp import fbp


def fdk_weigh_projections(A, y, src_rot_center_dist):
    # The distance of each pixel to the source is calulated in a coordinate
    # system based on the normalized u, v and normal vectors of the detector
    # because that way the distance along the v axis is constant along each
    # row and the distance along the u axis is constant along each column
    # resulting in fewer computations

    src_to_det = torch.from_numpy(A.range.det_pos[0] - A.range.src_pos[0])
    src_det_dist = torch.norm(src_to_det)
    # Load u, v vectors:
    det_u = torch.from_numpy(A.range.det_u[0])
    det_v = torch.from_numpy(A.range.det_v[0])
    # Calculate u, v vector lengths:
    det_u_norm = torch.norm(det_u)
    det_v_norm = torch.norm(det_v)
    # Transform source to detector center vector to the u, v system
    src_det_u_dist = torch.dot(src_to_det, det_u) / det_u_norm
    src_det_v_dist = torch.dot(src_to_det, det_v) / det_v_norm

    # calculate the distance of each row and column of pixels along the u or v axis
    u_num_pixels = A.range_shape[2]
    u_space = (torch.arange(u_num_pixels, dtype=torch.float64)+0.5-(u_num_pixels/2))
    v_num_pixels = A.range_shape[0]
    v_space = (torch.arange(v_num_pixels, dtype=torch.float64)+0.5-(v_num_pixels/2))

    # Calculate the minimal source to detector distance divided by the
    # pixel to detector distance for each pixel. The pixel to detector
    # distances are calculated using the pythagorean theorem
    u_pos_squared = (u_space * det_u_norm + src_det_u_dist)**2
    v_pos_squared = (v_space * det_v_norm + src_det_v_dist)**2

    weights_mat = src_det_dist / torch.sqrt(
          u_pos_squared[None, :] + v_pos_squared[:, None] + src_det_dist**2
    )

    # Multiply with extra scaling factor to account for detector distance
    weights_mat *= (src_rot_center_dist / src_det_dist)
    weights_mat = weights_mat.float().to(y.device)

    return y * weights_mat[:, None, :]


def fit_src_rot_center_dist(A):
    vg = A.domain
    pg = A.range

    T = ts.from_perspective(
        pos=pg.det_pos,
        w=pg.det_v/np.linalg.norm(pg.det_v[0]),
        v=pg.det_normal/np.linalg.norm(pg.det_normal[0]),
        u=pg.det_u/np.linalg.norm(pg.det_u[0])
    )

    det_to_obj = T.transform_point(vg.pos)
    det_to_src = T.transform_point(pg.src_pos)

    if np.ptp(det_to_obj[:, 0]) > ts.epsilon:
        print("Warning: Vertical object movement detected.")
    if np.ptp(det_to_src[:, 1]) > ts.epsilon:
        print("Warning: The source detector distance is changing over time.")

    return np.mean(det_to_obj[:, 1]-det_to_src[:, 1])


def fdk(A, y, padded=True, filter=None, reject_acyclic_filter=True, src_rot_center_dist=None):
    """Approximately reconstruct volumes in a circular cone beam geometry using
    the Feldkamp, Davis and Kress(FDK) algorithm [1]. Transformations on the
    geometry are allowed as long as the trajectory of the volume relative to
    the source consists of a circle parallel to the horizontal plane.

    If `y` is located on GPU, the entire algorithm is executed on a single GPU.

    If `y` is located in RAM (CPU in PyTorch parlance), then only the
    foward and backprojection are executed on GPU.

    :param A: `tomosipo.operator` The projection geometry of the operator must
    be either cone or cone_vec.
    :param y: `torch.tensor` sinogram stack with the following layout:
    (num_vertical_pixels, num_angles, num_horizontal_pixels)
    :param padded: bool, is passed to ts_algorithms.fbp
    :param filter: bool, is passed to ts_algorithms.fbp
    :param reject_acyclic_filter: bool, is passed to ts_algorithms.fbp
    :param src_rot_center_dist: optional, the distance from the source to the
    center of rotation of the object. If not provided and A doesn't contain a
    vector geometry A.range.src_orig_dist is used. If A does contain a vector
    geometry, the average source to object distance is used.
    :rtype: `torch.tensor`

    [1] Feldkamp, L. A., Davis, L. C., & Kress, J. W. (1984). Practical Cone-Beam
    Algorithm. Journal of the Optical Society of America A, 1(6), 612.
    http://dx.doi.org/10.1364/josaa.1.000612
    """
    voxel_sizes = np.array(A.volume_geometry.size) / np.array(A.volume_geometry.shape)
    if np.ptp(voxel_sizes) > ts.epsilon:
        raise ValueError(
            "The voxels in the volume must have the same size in every dimension.\n"
            f"This is not the case: voxel size = {voxel_sizes}."
        )

    pg = A.range
    vg = A.domain
    if not (isinstance(pg, ts.geometry.cone.ConeGeometry)
        or isinstance(pg, ts.geometry.cone_vec.ConeVectorGeometry)
    ):
        raise ValueError(
            "The provided operator A must describe a cone beam geometry."
        )

    if src_rot_center_dist is None:
        if (isinstance(pg, ts.geometry.cone.ConeGeometry)
            and isinstance(vg, ts.geometry.volume.VolumeGeometry)
        ):
            src_rot_center_dist = pg.src_orig_dist
        else:
            src_rot_center_dist = fit_src_rot_center_dist(A)

    return fbp(
        A=A,
        y=fdk_weigh_projections(A, y, src_rot_center_dist),
        padded=padded,
        filter=filter,
        reject_acyclic_filter=reject_acyclic_filter
    )
