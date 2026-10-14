"""Tool-tip kinematics of the robot articulation: pose and geometric Jacobian in the robot base frame.

Reads the inputs of `stem_manip.utils.tool_tip` (the formulas) from the robot's current data. Used by the action
term (`actions.py`) and by `scripts/check_contact.py` and `scripts/sweep_impedance.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from stem_manip.utils.tool_tip import tool_tip_jacobian, tool_tip_pose

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


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
        self.jacobian_joint_ids = [j + robot.num_base_dofs for j in joint_ids]
        """Columns of `joint_ids` in PhysX's Jacobian and mass matrix (shifted by the floating base's dofs)."""
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
            data.body_link_jacobian_w.torch[:, self._jacobi_body_idx, :, self.jacobian_joint_ids],
            data.root_quat_w.torch,
            data.body_quat_w.torch[:, self._body_idx],
            self._offset_pos,
        )
