"""Observation terms (privileged stem state in simulation).

Requirements:
- End-effector (tool_tip) pose, stem segment positions (and orientations) in the robot root frame,
  point-of-interest position, target position, last action.
- Shapes fixed per task; values in meters / normalized quaternions.

Implemented: tool-tip position and orientation, the applied action step and the target offset (the action term's
state: with the step-change limit and the target clamp the raw action alone does not tell the policy what was
executed; a limiter the policy cannot see breaks learning, Aljalbout et al., RA-L 2024). See docs/TODO.md -> M4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat, subtract_frame_transforms

from stem_manip.utils.impedance import rotvec_between

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions import ToolTipImpedanceAction


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


def applied_step(env: ManagerBasedEnv, action_name: str = "tool_tip") -> torch.Tensor:
    """Step the action term applied in the last policy step (num_envs, 6): translation / max_step and rotation
    vector / max_rot_step, so in [-1, 1] except where the clamp dragged the target along with a tool that moved
    faster than the cap (the applied step is the target's actual motion). Differs from the raw action where the
    step-change limit or the clamp acted."""
    term: ToolTipImpedanceAction = env.action_manager.get_term(action_name)
    step = term.processed_actions
    return torch.cat((step[:, :3] / term.cfg.max_step, step[:, 3:] / term.cfg.max_rot_step), dim=-1)


def target_offset(env: ManagerBasedEnv, action_name: str = "tool_tip") -> torch.Tensor:
    """Target pose relative to the measured tool tip (num_envs, 6), robot base frame: position offset /
    max_target_offset and rotation vector (tool -> target) / max_target_rot_offset, so about [-1, 1]: the clamp holds
    the bound at its physics step, this is measured after it (the tool may have moved slightly since). The spring
    force of the impedance law is proportional to it."""
    term: ToolTipImpedanceAction = env.action_manager.get_term(action_name)
    pos, quat = term.tool_pose()
    rot = rotvec_between(term.target_quat(), quat)
    return torch.cat(
        ((term.target() - pos) / term.cfg.max_target_offset, rot / term.cfg.max_target_rot_offset), dim=-1
    )
