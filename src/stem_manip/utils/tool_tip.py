"""Tool-tip kinematics: pose and geometric Jacobian of the tool tip in the robot base frame.

The tool tip is a fixed offset (position, quaternion (x, y, z, w)) in the end-effector body, e.g.
`stem_manip.assets.fr3.tool_tip_offset("fork_v2")`. Used by the push task's action term and observations and by
the scripts that drive the robot directly (`check_contact.py`, `sweep_impedance.py`).

`tool_tip_pose` and `tool_tip_jacobian` are pure tensor functions (tests: `tests/test_tool_tip.py`);
`ToolTipKinematics` reads their inputs from an Isaac Lab `Articulation`. Batched over the first dimension
(num_envs); the Jacobian rows are linear velocity then angular velocity, as in `stem_manip.utils.impedance`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


def tool_tip_pose(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    offset_pos: torch.Tensor,
    offset_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tool-tip position (n, 3) and orientation (n, 4) in the robot base frame.

    Args:
        root_pos_w, root_quat_w: robot base pose in the world (n, 3), (n, 4).
        body_pos_w, body_quat_w: end-effector body pose in the world (n, 3), (n, 4).
        offset_pos, offset_quat: tool tip in the body frame (n, 3), (n, 4).
    """
    pos, quat = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    return combine_frame_transforms(pos, quat, offset_pos, offset_quat)


def tool_tip_jacobian(
    body_jacobian_w: torch.Tensor, root_quat_w: torch.Tensor, body_quat_w: torch.Tensor, offset_pos: torch.Tensor
) -> torch.Tensor:
    """Geometric Jacobian of the tool tip in the robot base frame (n, 6, num_joints).

    Args:
        body_jacobian_w: geometric Jacobian of the end-effector body origin in world axes (n, 6, num_joints).
        root_quat_w: robot base orientation in the world (n, 4).
        body_quat_w: end-effector body orientation in the world (n, 4).
        offset_pos: tool-tip position in the body frame (n, 3).
    """
    base_quat_inv = quat_inv(root_quat_w)
    base_rot = matrix_from_quat(base_quat_inv)  # world -> base
    jacobian = torch.cat((base_rot @ body_jacobian_w[:, :3], base_rot @ body_jacobian_w[:, 3:]), dim=1)
    # shift the linear rows from the body origin to the tool tip: v_tip = v_body + w x r = v_body - [r]x w
    body_quat_b = quat_mul(base_quat_inv, body_quat_w)
    lever = quat_apply(body_quat_b, offset_pos)
    jacobian[:, :3] += torch.bmm(-skew_symmetric_matrix(lever), jacobian[:, 3:])
    return jacobian


class ToolTipKinematics:
    """Tool-tip pose and Jacobian of an articulation's end-effector body, read from its current data.

    Args:
        robot: the robot articulation.
        body_name: end-effector body that carries the tool.
        offset: tool tip in that body, (position, quaternion (x, y, z, w)).
        joint_ids: articulation joint indices of the Jacobian's columns (the arm joints).
    """

    def __init__(
        self,
        robot: Articulation,
        body_name: str,
        offset: tuple[Sequence[float], Sequence[float]],
        joint_ids: Sequence[int],
    ):
        body_ids, body_names = robot.find_bodies(body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one body for '{body_name}', found {body_names}")
        self._robot = robot
        self._body_idx = body_ids[0]
        # PhysX's Jacobians omit the fixed root body and put the floating base's dofs first
        self._jacobi_body_idx = self._body_idx - 1 if robot.is_fixed_base else self._body_idx
        self._jacobi_joint_ids = [j + robot.num_base_dofs for j in joint_ids]
        self._offset_pos, self._offset_quat = (
            torch.tensor(v, device=robot.device, dtype=torch.float32).repeat(robot.num_instances, 1) for v in offset
        )

    def pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Tool-tip position (n, 3) and orientation (n, 4) in the robot base frame."""
        data = self._robot.data
        return tool_tip_pose(
            data.root_pos_w.torch,
            data.root_quat_w.torch,
            data.body_pos_w.torch[:, self._body_idx],
            data.body_quat_w.torch[:, self._body_idx],
            self._offset_pos,
            self._offset_quat,
        )

    def jacobian(self) -> torch.Tensor:
        """Geometric Jacobian of the tool tip in the robot base frame (n, 6, len(joint_ids))."""
        data = self._robot.data
        return tool_tip_jacobian(
            data.body_link_jacobian_w.torch[:, self._jacobi_body_idx, :, self._jacobi_joint_ids],
            data.root_quat_w.torch,
            data.body_quat_w.torch[:, self._body_idx],
            self._offset_pos,
        )
