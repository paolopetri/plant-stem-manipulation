"""Reset and randomization events.

Requirements:
- Reset robot and stem to their default states (stem: `reset_cable_state_uniform` or own term).
- Later: domain randomization of stem parameters (stiffness, length, diameter, damping) once the
  per-env randomization path is clarified. Ranges in the env cfg.

Implemented: base clamping at startup; the reset uses Isaac Lab's `reset_scene_to_default`, then
`reset_stem_base_uniform` moves the stem base within the spawn area (env cfg).
See docs/TODO.md -> M4 (randomization: Later).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

from stem_manip.assets.stem import stem_model

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def fix_stem_base(
    env: ManagerBasedEnv, env_ids: torch.Tensor | None, model: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("stem")
) -> None:
    """Startup event: clamp the stem base with the stem model's `fix_stem_base` (`chain`: nothing to do)."""
    stem_model(model).fix_stem_base(env.scene[asset_cfg.name])


def reset_stem_base_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("stem"),
) -> None:
    """Reset event: stem base uniformly in the rectangle `x_range` x `y_range` [m, env frame], absolute (independent
    of the stem's `init_state.pos`); height and orientation from the default root pose, at rest. Writes the root pose
    of an articulated stem (`chain`); joint state is left to the default reset before it."""
    stem = env.scene[asset_cfg.name]
    pose = stem.data.default_root_pose.torch[env_ids].clone()
    ranges = torch.tensor((x_range, y_range), device=env.device)
    pose[:, :2] = sample_uniform(ranges[:, 0], ranges[:, 1], (pose.shape[0], 2), device=env.device)
    pose[:, :3] += env.scene.env_origins[env_ids]
    stem.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    stem.write_root_velocity_to_sim_index(
        root_velocity=stem.data.default_root_vel.torch[env_ids].clone(), env_ids=env_ids
    )
