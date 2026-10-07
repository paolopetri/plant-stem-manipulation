"""Observation terms (privileged stem state in simulation).

Requirements:
- End-effector (tool_tip) pose, stem segment positions (and orientations) in the robot root frame,
  point-of-interest position, target position, last action.
- Shapes fixed per task; values in meters / normalized quaternions.

Implemented: tool-tip position and orientation. See docs/TODO.md -> M4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def tool_tip_pose_b(
    env: ManagerBasedEnv, body_name: str, offset: tuple[tuple[float, ...], tuple[float, ...]], asset_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tool-tip position (num_envs, 3) and orientation (num_envs, 4) in the robot root frame.

    The tool tip is `offset` = (pos, quat) in the body `body_name` (see `stem_manip.assets.fr3.tool_tip_offset`).
    """
    robot = env.scene[asset_cfg.name]
    body = robot.body_names.index(body_name)
    pos_b, quat_b = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        robot.data.body_pos_w.torch[:, body],
        robot.data.body_quat_w.torch[:, body],
    )
    offset_pos, offset_quat = (torch.tensor(v, device=env.device).repeat(env.num_envs, 1) for v in offset)
    return combine_frame_transforms(pos_b, quat_b, offset_pos, offset_quat)


def tool_tip_pos(
    env: ManagerBasedEnv,
    body_name: str,
    offset: tuple[tuple[float, ...], tuple[float, ...]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Tool-tip position in the robot root frame (num_envs, 3) [m]."""
    return tool_tip_pose_b(env, body_name, offset, asset_cfg)[0]


def tool_tip_rot6d(
    env: ManagerBasedEnv,
    body_name: str,
    offset: tuple[tuple[float, ...], tuple[float, ...]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Tool-tip orientation as the first two columns of its rotation matrix (num_envs, 6): the tool's x and y axes in
    the robot base frame. Continuous, unlike quaternions (q and -q); Zhou et al., CVPR 2019 (decision 2026-10-07)."""
    rotation = matrix_from_quat(tool_tip_pose_b(env, body_name, offset, asset_cfg)[1])
    return torch.cat((rotation[..., 0], rotation[..., 1]), dim=-1)
