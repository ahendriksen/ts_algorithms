import numpy as np
import torch
import tomosipo as ts
from fbp import fbp


def fdk_weigh_projections(op, projections, recalculate_weights):
    dev = projections.device
    u_num_pixels = op.range_shape[2]
    u_space = (torch.arange(u_num_pixels, dtype=torch.float, device=dev)+0.5-(u_num_pixels/2))
    v_num_pixels = op.range_shape[0]
    v_space = (torch.arange(v_num_pixels, dtype=torch.float, device=dev)+0.5-(v_num_pixels/2))

    result = torch.empty(projections.shape, dtype=torch.float, device=dev)

    for i in range(op.range_shape[1]):
        if recalculate_weights or i == 0:
            src_to_det = torch.from_numpy(op.range.det_pos[i] - op.range.src_pos[i]).float().to(dev)
            src_to_obj = torch.from_numpy(-op.range.src_pos[i]).float().to(dev)
            det_n = torch.from_numpy(op.range.det_normal[i]).float().to(dev)
            det_u = torch.from_numpy(op.range.det_u[i]).float().to(dev)
            det_v = torch.from_numpy(op.range.det_v[i]).float().to(dev)
            det_n_norm = torch.norm(det_n)
            det_u_norm = torch.norm(det_u)
            det_v_norm = torch.norm(det_v)
            src_det_n_dist = torch.dot(src_to_det, det_n) / det_n_norm
            src_det_u_dist = torch.dot(src_to_det, det_u) / det_u_norm
            src_det_v_dist = torch.dot(src_to_det, det_v) / det_v_norm
            src_to_obj_n_dist = torch.dot(src_to_obj, det_n) / det_n_norm

            u_pos_squared = (u_space * det_u_norm + src_det_u_dist)**2
            v_pos_squared = (v_space * det_v_norm + src_det_v_dist)**2

            weights_mat = src_det_n_dist / torch.sqrt(
                  u_pos_squared[None, :].expand([v_num_pixels, u_num_pixels])
                  + v_pos_squared[:, None].expand([v_num_pixels, u_num_pixels])
                  + src_det_n_dist**2
            )
            weights_mat *= (src_to_obj_n_dist / src_det_n_dist)  # scale detector to origin

        result[:, i, :] = projections[:, i, :] * weights_mat

    return result


def fdk(A, y, padded=True, filter=None, reject_acyclic_filter=True, recalculate_weights=False):
    voxel_sizes = np.array(A.volume_geometry.size) / np.array(A.volume_geometry.shape)
    if (np.ptp(voxel_sizes) > ts.epsilon):
        raise ValueError(
            "The voxels in the volume are required to have the same size in every dimension.\n"
            f"This is not the case: voxel size = {voxel_sizes}."
        )
    return fbp(
        A=A,
        y=fdk_weigh_projections(A, y, recalculate_weights),
        padded=padded,
        filter=filter,
        reject_acyclic_filter=reject_acyclic_filter
    )
