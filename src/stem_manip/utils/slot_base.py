"""Slot-base kinematics: pose and geometric Jacobian of the slot base in the robot base frame (tensor math only).

The slot base is a fixed offset (position, quaternion (x, y, z, w)) in the end-effector body, e.g.
`stem_manip.assets.fr3.slot_base_offset("fork_v2")`. `slot_base_pose` and `slot_base_jacobian` are pure tensor
functions (tests: `tests/test_slot_base.py`); the push task's `mdp/slot_base.py` (`SlotBaseKinematics`) reads their
inputs from the robot articulation for the action term and the scripts, the observations call `slot_base_pose`.
Batched over the first dimension (num_envs); the Jacobian rows are linear velocity then angular velocity, as in
`stem_manip.utils.impedance`.

`isaaclab.utils.math` is plain torch math (no simulator); it is used instead of `stem_manip.utils.quat` so that the
action term and the observations stay bit-exact with the formulas they used before this module.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply,
    quat_inv,
    quat_mul,
    skew_symmetric_matrix,
    subtract_frame_transforms,
)


def slot_base_pose(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    offset_pos: torch.Tensor,
    offset_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slot-base position (n, 3) and orientation (n, 4) in the robot base frame.

    Args:
        root_pos_w, root_quat_w: robot base pose in the world (n, 3), (n, 4).
        body_pos_w, body_quat_w: end-effector body pose in the world (n, 3), (n, 4).
        offset_pos, offset_quat: slot base in the body frame (n, 3), (n, 4).
    """
    pos, quat = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    return combine_frame_transforms(pos, quat, offset_pos, offset_quat)


def slot_base_jacobian(
    body_jacobian_w: torch.Tensor, root_quat_w: torch.Tensor, body_quat_w: torch.Tensor, offset_pos: torch.Tensor
) -> torch.Tensor:
    """Geometric Jacobian of the slot base in the robot base frame (n, 6, num_joints).

    Args:
        body_jacobian_w: geometric Jacobian of the end-effector body origin in world axes (n, 6, num_joints).
        root_quat_w: robot base orientation in the world (n, 4).
        body_quat_w: end-effector body orientation in the world (n, 4).
        offset_pos: slot-base position in the body frame (n, 3).
    """
    base_quat_inv = quat_inv(root_quat_w)
    base_rot = matrix_from_quat(base_quat_inv)  # world -> base
    jacobian = torch.cat((base_rot @ body_jacobian_w[:, :3], base_rot @ body_jacobian_w[:, 3:]), dim=1)
    # shift the linear rows from the body origin to the slot base: v_slot_base = v_body + w x r = v_body - [r]x w
    body_quat_b = quat_mul(base_quat_inv, body_quat_w)
    lever = quat_apply(body_quat_b, offset_pos)
    jacobian[:, :3] += torch.bmm(-skew_symmetric_matrix(lever), jacobian[:, 3:])
    return jacobian
