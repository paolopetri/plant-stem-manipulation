"""Stem state for the MDP terms: the one place where the stem is read (privileged state in simulation).

On the real robot the stem state will come from cameras (perception, with Alessio Caporali), so everything the policy
observes about the stem is defined here and can be changed to what the perception delivers (decision 2026-10-06).
No contact forces in the observations (decision 2026-10-06).

- `stem_segment_poses(env, model)`: the stem accessor of `docs/architecture.md` (Stem interface); the only call into
  the stem model.
- `stem_points_b(env, arc_lengths, model)`: points on the centre line at given arc lengths, robot base frame
  (`to_robot_base`: world -> robot base frame).
- Observation term `stem_points`: the same, flattened (num_envs, 3 * len(arc_lengths)) [m]. Used for the stem base
  (arc length 0) and the 5 points along the stem (0.08 ... 0.40 m, the last one is the tip; user, 2026-10-08).
  The target is observed with Isaac Lab's `generated_commands` (robot base frame, `mdp/commands.py`).

Verify: `tests/test_stem_geometry.py` (point helper), `scripts/check_push_obs.py` (in the env).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from stem_manip.assets.stem import stem_model, stem_params
from stem_manip.utils import stem_geometry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def stem_segment_poses(
    env: ManagerBasedEnv, model: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("stem")
) -> torch.Tensor:
    """Segment poses (num_envs, num_segments, 7): position + quaternion (x, y, z, w), world frame."""
    return stem_model(model).segment_poses(env.scene[asset_cfg.name])


@lru_cache
def segment_length(model: str) -> float:
    """Rest length of one segment of the stem model [m] (all segments equally long). Cached: read from the yaml
    once, not at every policy step (one value for all envs; per-env lengths: docs/TODO.md, randomization)."""
    geometry = stem_params(model)["geometry"]
    return geometry["length"] / geometry["num_segments"]


def stem_points_b(
    env: ManagerBasedEnv,
    arc_lengths: Sequence[float],
    model: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("stem"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Points on the stem's centre line at `arc_lengths` [m], robot base frame, shape (num_envs, len, 3) [m]."""
    points_w = stem_geometry.stem_points(stem_segment_poses(env, model, asset_cfg), arc_lengths, segment_length(model))
    return to_robot_base(env, points_w, robot_cfg)


def to_robot_base(
    env: ManagerBasedEnv, points_w: torch.Tensor, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """World-frame points (num_envs, num_points, 3) [m] in the robot base frame, same shape."""
    robot = env.scene[robot_cfg.name]
    num_points = points_w.shape[1]
    root_pos = robot.data.root_pos_w.torch.repeat_interleave(num_points, dim=0)
    root_quat = robot.data.root_quat_w.torch.repeat_interleave(num_points, dim=0)
    points_b, _ = subtract_frame_transforms(root_pos, root_quat, points_w.reshape(-1, 3))
    return points_b.reshape(-1, num_points, 3)


def stem_points(
    env: ManagerBasedEnv,
    arc_lengths: Sequence[float],
    model: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("stem"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Observation: points on the stem at `arc_lengths` [m], robot base frame, (num_envs, 3 * len) [m], raw meters
    (x, y, z of the first point, then the next)."""
    return stem_points_b(env, arc_lengths, model, asset_cfg, robot_cfg).flatten(start_dim=1)
