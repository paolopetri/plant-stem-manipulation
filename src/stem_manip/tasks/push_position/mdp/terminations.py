"""Termination terms (decided values: user, 2026-10-09; thresholds in the env cfg, damage limits from `stem.yaml`).

- Damage limits (no-damage constraint): bending curvature of any joint above `damage.max_curvature`; contact force
  of the robot on the stem (mean over the policy step) above `damage.max_contact_force`. Twist and tension once
  their limits are set.
- Joint margin: any FR3 joint closer than `min_margin` to its limit (the real FR3 stops with a reflex there).
- Time out: Isaac Lab's `time_out`. No tool-tip workspace bound (user, 2026-10-09: it would limit the motion too
  much; the time out and the speed cap bound an episode).

Each term is shown to fire in `scripts/check_push_terms.py`. See docs/TODO.md -> M4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .rewards import robot_contact_force
from .stem_state import stem_curvature

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def curvature_limit(
    env: ManagerBasedRLEnv, model: str, max_curvature: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("stem")
) -> torch.Tensor:
    """True where any joint of the stem bends more than `max_curvature` [1/m]."""
    return (stem_curvature(env, model, asset_cfg) > max_curvature).any(dim=-1)


def contact_force_limit(env: ManagerBasedRLEnv, max_force: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """True where the robot's contact force on the stem (`rewards.robot_contact_force`) exceeds `max_force` [N]."""
    return robot_contact_force(env, sensor_cfg) > max_force


def joint_margin(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Smallest distance of the joints in `asset_cfg` to their position limits (num_envs,) [rad]."""
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos.torch[:, asset_cfg.joint_ids]
    limits = robot.data.joint_pos_limits.torch[:, asset_cfg.joint_ids]
    return torch.minimum(q - limits[..., 0], limits[..., 1] - q).min(dim=-1).values


def joint_limit_margin(env: ManagerBasedRLEnv, min_margin: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """True where any joint in `asset_cfg` is closer than `min_margin` [rad] to a position limit."""
    return joint_margin(env, asset_cfg) < min_margin
