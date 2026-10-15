"""Termination terms (decided values: user, 2026-10-09; thresholds in the env cfg, damage limits from `stem.yaml`).

- Damage limits (no-damage constraint): bending curvature of any joint above `damage.max_curvature`; contact force
  of the robot on the stem (mean over the policy step) above `damage.max_contact_force`. Twist and tension once
  their limits are set (#81, #82).
- Joint margin: any FR3 joint closer than `min_margin` to its limit (the real FR3 stops with a reflex there).
- Arm contact (user, 2026-10-15, #160): any robot body other than the fork touches the stem, its contact force (mean
  over the policy step, as the contact-force limit) above `max_force`. A contact rule, not a damage limit
  (`GLOSSARY.md`): the policy may not push with the arm.
- Time out: Isaac Lab's `time_out`. No slot-base workspace bound (user, 2026-10-09: it would limit the motion too
  much; the time out and the speed cap bound an episode).

Each term is shown to fire in `scripts/check_push_terms.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .contact import arm_contact_force, robot_contact_force
from .stem_state import stem_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def curvature_limit(env: ManagerBasedRLEnv, max_curvature: float) -> torch.Tensor:
    """True where any joint of the stem bends more than `max_curvature` [1/m]."""
    return (stem_state(env).curvature() > max_curvature).any(dim=-1)


def contact_force_limit(env: ManagerBasedRLEnv, max_force: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """True where the robot's contact force on the stem (`contact.robot_contact_force`) exceeds `max_force` [N]."""
    return robot_contact_force(env, sensor_cfg) > max_force


def arm_contact(env: ManagerBasedRLEnv, max_force: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """True where the contact force of the robot bodies other than the fork on the stem
    (`contact.arm_contact_force`) exceeds `max_force` [N]."""
    return arm_contact_force(env, sensor_cfg) > max_force


def joint_margin(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Smallest distance of the joints in `asset_cfg` to their position limits (num_envs,) [rad]."""
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos.torch[:, asset_cfg.joint_ids]
    limits = robot.data.joint_pos_limits.torch[:, asset_cfg.joint_ids]
    return torch.minimum(q - limits[..., 0], limits[..., 1] - q).min(dim=-1).values


def joint_limit_margin(env: ManagerBasedRLEnv, min_margin: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """True where any joint in `asset_cfg` is closer than `min_margin` [rad] to a position limit."""
    return joint_margin(env, asset_cfg) < min_margin
