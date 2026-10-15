"""Reset and randomization events.

Base clamping at startup; the reset uses Isaac Lab's `reset_scene_to_default`, then `reset_arm_start_pose` puts the arm
at a start pose picked from a table, and `reset_stem_base_uniform` moves the stem base within the spawn area, clear of
the fork and the arm (env cfg). Domain randomization of the stem parameters: #89, #90.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, sample_uniform

from stem_manip.assets.fr3 import ARM_LINKS, arm_surface_points
from stem_manip.assets.stem import stem_model, stem_params
from stem_manip.utils.stem_geometry import distance_to_centre_line

from .slot_base import slot_base
from .stem_state import STEM

MAX_STEM_DRAWS = 100  # draws of a stem position clear of the fork and the arm before the reset gives up
FORK_POINT_SPACING = 0.01  # [m] points along the fork's centre line
ARM_MESH_POINTS = 4000  # surface points per arm collision mesh (as the start-pose table's arm clearance)
ARM_ENV_CHUNK = 256  # envs per batch of the arm clearance (bounds its memory: 256 x 32000 points)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def fix_stem_base(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
    """Startup event: clamp the stem base with the `fix_stem_base` of the stem model named in `env.cfg.stem_model`
    (`chain`: nothing to do)."""
    stem_model(env.cfg.stem_model).fix_stem_base(env.scene[STEM])


def reset_arm_start_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    poses: Sequence[Sequence[float]],
    joint_noise: float,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Reset event: each env in `env_ids` gets one of `poses` at random (joint angles [rad] of `asset_cfg`'s joints, in
    their order), plus uniform noise in [-`joint_noise`, `joint_noise`] [rad] on every joint, at rest."""
    env_ids = _env_id_tensor(env, env_ids)
    robot = env.scene[asset_cfg.name]
    table = torch.tensor(poses, dtype=torch.float32, device=env.device)
    picked = table[torch.randint(len(table), (len(env_ids),), device=env.device)]
    q = picked + sample_uniform(-joint_noise, joint_noise, picked.shape, device=env.device)
    joint_ids = asset_cfg.joint_ids
    robot.write_joint_position_to_sim_index(position=q, joint_ids=joint_ids, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=joint_ids, env_ids=env_ids)


def reset_stem_base_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    fork_extent: tuple[float, float],
    min_fork_distance: float,
    min_arm_clearance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("stem"),
) -> None:
    """Reset event: stem base uniformly in the rectangle `x_range` x `y_range` [m, env frame], absolute (independent
    of the stem's `init_state.pos`); height and orientation from the default root pose, at rest. Writes the root pose
    of an articulated stem (`chain`); joint state is left to the default reset before it.

    Clear of the fork: a position where the upright stem's axis comes closer than `min_fork_distance` [m] to the fork's
    centre line (the slot base's x axis from `fork_extent[0]` to `fork_extent[1]` [m], at the robot's pose after the
    reset events before this one) is drawn again; so is a position where the stem's surface comes closer than
    `min_arm_clearance` [m] to an arm body (any robot body but the end-effector: surface points of the FR3 collision
    meshes, `ARM_MESH_POINTS` per body, at the link poses after those events). Assumes robot base frame = env frame
    (the spawn area's convention).
    """
    env_ids = _env_id_tensor(env, env_ids)
    stem = env.scene[asset_cfg.name]
    pose = stem.data.default_root_pose.torch[env_ids].clone()
    ranges = torch.tensor((x_range, y_range), device=env.device)
    pose[:, :2] = sample_uniform(ranges[:, 0], ranges[:, 1], (pose.shape[0], 2), device=env.device)
    fork = _fork_centre_line(env, env_ids, fork_extent)
    geometry = stem_params(env.cfg.stem_model)["geometry"]
    length, radius = geometry["length"], geometry["diameter"] / 2
    axis = torch.zeros(len(env_ids), 1, 7, device=env.device)  # the upright stem as one segment
    axis[..., 2], axis[..., 6] = length / 2, 1.0
    close = torch.ones(len(env_ids), dtype=torch.bool, device=env.device)  # envs whose position is (re)checked
    for _ in range(MAX_STEM_DRAWS):
        axis[:, 0, :2] = pose[:, :2]
        fork_close = distance_to_centre_line(axis[close], fork[close], length).amin(dim=-1) < min_fork_distance
        arm_clearance = _arm_stem_distance(env, env_ids[close], pose[close, :2], length) - radius
        close[close.clone()] = fork_close | (arm_clearance < min_arm_clearance)
        if not close.any():
            break
        pose[close, :2] = sample_uniform(ranges[:, 0], ranges[:, 1], (int(close.sum()), 2), device=env.device)
    else:
        raise RuntimeError(f"No stem position clear of the fork and the arm after {MAX_STEM_DRAWS} draws")
    pose[:, :3] += env.scene.env_origins[env_ids]
    stem.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    stem.write_root_velocity_to_sim_index(
        root_velocity=stem.data.default_root_vel.torch[env_ids].clone(), env_ids=env_ids
    )


def _fork_centre_line(env: ManagerBasedEnv, env_ids: torch.Tensor, extent: tuple[float, float]) -> torch.Tensor:
    """Points (len(env_ids), k, 3) on the slot base's x axis from `extent[0]` to `extent[1]` [m], robot base frame."""
    pos, quat = (v[env_ids] for v in slot_base(env).pose())
    num_points = int((extent[1] - extent[0]) / FORK_POINT_SPACING) + 1
    s = torch.linspace(*extent, num_points, device=env.device)
    x_axis = quat_apply(quat, torch.tensor((1.0, 0.0, 0.0), device=env.device).expand(len(env_ids), 3))
    return pos[:, None] + s[None, :, None] * x_axis[:, None]


def _arm_stem_distance(
    env: ManagerBasedEnv, env_ids: torch.Tensor, stem_xy: torch.Tensor, length: float
) -> torch.Tensor:
    """Smallest distance (len(env_ids),) [m] of the arm bodies' surface points to the axis of an upright stem of
    `length` [m] standing at `stem_xy` (len(env_ids), 2) [m, robot base frame], in batches of `ARM_ENV_CHUNK` envs."""
    robot = env.scene["robot"]
    body_ids = [robot.body_names.index(name) for name in ARM_LINKS]
    local = [p.to(env.device) for p in arm_surface_points(ARM_MESH_POINTS)]
    distances = []
    for chunk in torch.split(torch.arange(len(env_ids), device=env.device), ARM_ENV_CHUNK):
        ids = env_ids[chunk]
        pos = robot.data.body_pos_w.torch[ids][:, body_ids] - env.scene.env_origins[ids, None]
        quat = robot.data.body_quat_w.torch[ids][:, body_ids]
        points = torch.cat(
            [quat_apply(quat[:, [b]].expand(-1, len(p), -1), p.expand(len(ids), -1, -1)) + pos[:, [b]]
             for b, p in enumerate(local)],
            dim=1,
        )  # fmt: skip
        horizontal = (points[..., :2] - stem_xy[chunk, None]).norm(dim=-1)
        vertical = torch.maximum(points[..., 2] - length, -points[..., 2]).clamp(min=0.0)  # beyond the base or the tip
        distances.append(torch.hypot(horizontal, vertical).amin(dim=-1))
    return torch.cat(distances) if distances else torch.zeros(0, device=env.device)


def _env_id_tensor(env: ManagerBasedEnv, env_ids: torch.Tensor | slice) -> torch.Tensor:
    """The env ids as a tensor (the event manager passes a slice for all envs)."""
    if isinstance(env_ids, slice):
        return torch.arange(env.scene.num_envs, device=env.device)[env_ids]
    return env_ids
