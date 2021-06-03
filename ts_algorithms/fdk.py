import warnings
import numpy as np
import torch
import tomosipo as ts
from tomosipo.geometry import (
    ConeGeometry,
    ConeVectorGeometry,
)
from .fbp import fbp


def fdk_weigh_projections(A, y, overwrite_y):
    # Move to perspective of detector without changing the scale of
    # the coordinate axes.
    vg, pg = A.domain, A.range

    T = ts.from_perspective(
        pos=pg.src_pos,
        w=pg.det_v/np.linalg.norm(pg.det_v[0]),
        v=pg.det_normal/np.linalg.norm(pg.det_normal[0]),
        u=pg.det_u/np.linalg.norm(pg.det_u[0])
    )
    # We have:
    # - the source position is placed at the origin
    # - the z axis is parallel to the detector v axis
    # - the y axis is orthogonal to the detecor plane
    # - the x axis is parallel to the detector u axis
    vg_fixed, pg_fixed = T * vg.to_vec(), T * pg.to_vec()

    ###########################################################################
    #                    Determine source-detector distance                   #
    ###########################################################################
    # The source is located on the origin. So the source-detector
    # distance can be read of from the detector position. We warn if
    # the detector position is not constant (and use the mean detector
    # position regardless).
    det_positions = pg_fixed.det_pos
    det_pos = det_positions.mean(axis=0)

    if np.ptp(det_positions, axis=0).max() > ts.epsilon:
        warnings.warn(
            f"The source to detector distance is not constant. "
            f"It has a variation of {np.ptp(det_positions, axis=0)}. "
            f"This may cause unexpected results in the reconstruction. "
            f"The mean source to detector distance ({det_pos}) "
            "has been used to compute the reconstruction. "
        )

    ###########################################################################
    #                Determine source-rotation center distance                #
    ###########################################################################
    # Read of source-object distance in the y coordinate of the
    # transformed volume position.
    obj_positions = vg_fixed.pos[:, 1]

    # Take the rotation center as the mean of the volume positions.
    rot_center_pos = obj_positions.mean()
    src_rot_center_dist = abs(rot_center_pos)

    # Check that the center of rotation is "in front" of the source
    # beam. Warn otherwise. We want to avoid the situation:
    # rot_center  src ----> det
    #    ⊙         .  ---->  |   or    | <----  .    ⊙
    if np.sign(rot_center_pos) != np.sign(det_pos[1]):
        warnings.warn(
            "Rotation center of volume is behind source position. "
            "Adjust your geometry to obtain a better reconstruction. "
        )

    ###########################################################################
    #                            Create pixel grid                            #
    ###########################################################################
    num_v, num_u = pg.det_shape
    v_size, u_size = np.array(pg.det_size) / np.array(pg.det_shape)

    v_range = torch.arange(num_v, dtype=torch.float64) - (num_v - 1) / 2
    u_range = torch.arange(num_u, dtype=torch.float64) - (num_u - 1) / 2
    u_pos_squared = (det_pos[2] + u_size * u_range) ** 2
    v_pos_squared = (det_pos[0] + v_size * v_range) ** 2

    # Determine source-pixel distance for each pixel on the detector.
    src_pixel_dist = torch.sqrt(
          u_pos_squared[None, :] + v_pos_squared[:, None] + det_pos[1]**2
    )

    ###########################################################################
    #                           Determine weighting                           #
    ###########################################################################
    # Multiply with extra scaling factor to account for detector distance
    weights_mat = src_rot_center_dist / src_pixel_dist
    weights_mat = weights_mat.float().to(y.device)

    if overwrite_y:
        y *= weights_mat[:, None, :]
        return y
    else:
        return y * weights_mat[:, None, :]


def fdk(A, y, padded=True, filter=None, batch_size=10, overwrite_y=False):
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
    :param batch_size: int, is passed to ts_algorithms.fbp
    :param overwrite_y: bool, Specifies whether to overwrite y with the
    filtered version while running this function. If overwrite_y==False an
    extra block of memory with the size of y needs to be allocated, so use
    overwrite_y==True if you would otherwise run out of memory. Choose
    overwrite_y==False if you still want to use y after calling this function.

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
    y_weighted = fdk_weigh_projections(A, y, overwrite_y)

    # Compute a normal FBP reconstruction
    return fbp(
        A=A,
        y=y_weighted,
        padded=padded,
        filter=filter,
        batch_size=batch_size,
        overwrite_y=True
    )
