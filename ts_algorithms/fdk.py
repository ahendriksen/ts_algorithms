import warnings
import numpy as np
import torch
import tomosipo as ts
from tomosipo.geometry import (
    ConeGeometry,
    ConeVectorGeometry,
)
from .fbp import fbp


def fdk_weigh_projections(A, y):
    # Move to perspective of detector without changing the scale of
    # the coordinate axes.
    vg, pg = A.domain, A.range

    T = ts.from_perspective(
        pos=pg.det_pos,
        w=pg.det_v/np.linalg.norm(pg.det_v[0]),
        v=pg.det_normal/np.linalg.norm(pg.det_normal[0]),
        u=pg.det_u/np.linalg.norm(pg.det_u[0])
    )

    vg_fixed, pg_fixed = T * vg.to_vec(), T * pg.to_vec()

    ###########################################################################
    #                    Determine source-detector distance                   #
    ###########################################################################
    # Read of source-detector distance in the y coordinate of the
    # transformed source position.
    src_det_dists = pg_fixed.src_pos[:, 1]
    src_det_dist = src_det_dists.mean()

    if np.ptp(src_det_dists) > ts.epsilon:
        warnings.warn(
            f"The source to detector distance is not constant. "
            f"It has a variation of {np.ptp(src_det_dists): 0.2e}. "
            f"This may cause unexpected results in the reconstruction. "
            f"The mean source to detector distance ({src_det_dist: 0.2e}) "
            "has been used to compute the reconstruction. "
        )

    ###########################################################################
    #                Determine source-rotation center distance                #
    ###########################################################################
    # Read of source-object distance in the y coordinate of the
    # transformed volume position.
    src_obj_dists = vg_fixed.pos[:, 1] - src_det_dists

    # Take the rotation center as the mean of the volume positions.
    src_rot_center_dist = src_obj_dists.mean()

    if src_rot_center_dist < 0.0:
        raise ValueError(
            "Rotation center is behind source position. "
            "Consider adjusting your geometry to obtain a reconstruction. "
        )

    ###########################################################################
    #                            Create pixel grid                            #
    ###########################################################################
    num_v, num_u = pg.det_shape
    v_size, u_size = np.array(pg.det_size) / np.array(pg.det_shape)

    v_range = torch.arange(num_v, dtype=torch.float64) - (num_v - 1) / 2
    u_range = torch.arange(num_u, dtype=torch.float64) - (num_u - 1) / 2
    u_pos_squared = (u_size * u_range) ** 2
    v_pos_squared = (v_size * v_range) ** 2

    # Determine source-pixel distance for each pixel on the detector.
    src_pixel_dist = torch.sqrt(
          u_pos_squared[None, :] + v_pos_squared[:, None] + src_det_dist**2
    )

    ###########################################################################
    #                           Determine weighting                           #
    ###########################################################################
    weights_mat = src_det_dist / src_pixel_dist

    # Multiply with extra scaling factor to account for detector distance
    weights_mat *= (src_rot_center_dist / src_det_dist)
    weights_mat = weights_mat.float().to(y.device)

    return y * weights_mat[:, None, :]


def fdk(A, y, padded=True, filter=None, reject_acyclic_filter=True):
    """Compute FDK reconstruction

    Approximately reconstruct volumes in a circular cone beam geometry using
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

    vg = A.astra_compat_vg
    pg = A.astra_compat_pg

    voxel_sizes = np.array(vg.size) / np.array(vg.shape)
    voxel_size_var = np.ptp(voxel_sizes)
    if voxel_size_var > ts.epsilon:
        # XXX: Consider making this a warning.. This use case is could
        # be worthwile even if the reconstruction is not perfect.
        raise ValueError(
            "The voxels in the volume must have the same size in every dimension. "
            f"Found variation of {voxel_size_var:0.2e}."
        )

    det_size_var = np.ptp(pg.det_sizes, axis=0).max()
    if det_size_var > ts.epsilon:
        raise ValueError(
            "The size of the detector is not constant. "
            f"Found variation of {det_size_var:0.2e}."
        )

    if not (isinstance(pg, ConeGeometry) or isinstance(pg, ConeVectorGeometry)):
        raise TypeError(
            "The provided operator A must describe a cone beam geometry."
        )

    # Pre-weigh projections by the inverse of the source-to-pixel distance
    y_weighted = fdk_weigh_projections(A, y)

    # Compute a normal FBP reconstruction
    return fbp(
        A=A,
        y=y_weighted,
        padded=padded,
        filter=filter,
        reject_acyclic_filter=reject_acyclic_filter
    )
