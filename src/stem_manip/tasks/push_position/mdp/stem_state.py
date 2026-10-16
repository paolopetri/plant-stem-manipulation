"""Stem state for the MDP terms: the one place where the stem is read (privileged state in simulation).

On the real robot the stem state will come from cameras (perception, with Alessio Caporali), so everything the policy
observes about the stem is defined here and can be changed to what the perception delivers (decision 2026-10-06).
No contact forces in the observations (decision 2026-10-06).

- `StemState`: the stem accessor of `docs/architecture.md` (Stem interface): points at given arc lengths, the
  bending curvature of every joint, the distance of points to the centre line, all in the robot base frame, and the
  axial compression of every joint, the arc length of every segment. It owns the stem model, the segment length
  (read from the stem asset) and the world-to-robot-base transform.
- `stem_state(env)`: the env's one instance, built from `env.cfg.stem_model` and the scene entities `STEM` and
  `ROBOT` (cached by `stem_manip.utils.per_env`).
- Observation term `stem_points`: points at arc lengths, flattened (num_envs, 3 * len(arc_lengths)) [m]. Used for the
  stem base (arc length 0) and the 5 points along the stem (0.08 ... 0.40 m, the last one is the tip; user,
  2026-10-08). The target is observed with Isaac Lab's `generated_commands` (robot base frame, `mdp/commands.py`).
- The curvature and the distance are for the rewards and terminations, the compression for the metrics (privileged,
  not observed).

Verify: `tests/test_stem_state.py` (with a fake stem model), `scripts/check_push_obs.py` (in the env).
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

from stem_manip.assets.stem import stem_model
from stem_manip.utils import stem_geometry
from stem_manip.utils.per_env import per_env

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, CableObject
    from isaaclab.envs import ManagerBasedEnv

STEM = "stem"
"""Scene name of the stem."""
ROBOT = "robot"
"""Scene name of the robot whose base frame the stem state is in."""


class StemState:
    """The stem read through a stem model, in the robot base frame.

    Args:
        model: stem model module (`stem_manip.assets.stem.stem_model(name)`): `segment_poses(stem)`,
            `segment_length(stem)`.
        stem: the stem asset (an `Articulation` for `chain`, a `CableObject` for `cable`).
        robot: the robot asset whose base frame the results are in.
    """

    def __init__(self, model: ModuleType, stem: Articulation | CableObject, robot: Articulation):
        self._model = model
        self._stem = stem
        self._robot = robot
        self._segment_length = model.segment_length(stem)

    def points_b(self, arc_lengths: Sequence[float]) -> torch.Tensor:
        """Points on the stem's centre line at `arc_lengths` [m] from the base (0: the stem base), robot base frame,
        shape (num_envs, len(arc_lengths), 3) [m]."""
        points_w = stem_geometry.stem_points(self._segment_poses(), arc_lengths, self._segment_length)
        return _world_to_base(self._robot, points_w)

    def curvature(self) -> torch.Tensor:
        """Bending curvature of every stem joint, shape (num_envs, num_segments - 1) [1/m]."""
        poses = self._segment_poses()
        lengths = torch.full((poses.shape[1],), self._segment_length, dtype=poses.dtype, device=poses.device)
        return stem_geometry.joint_curvature(poses, lengths)

    def distance_b(self, points_b: torch.Tensor) -> torch.Tensor:
        """Distance of points (num_envs, k, 3), robot base frame [m], to the stem's centre line, shape (num_envs, k)
        [m]."""
        points_w = _base_to_world(self._robot, points_b)
        return stem_geometry.distance_to_centre_line(self._segment_poses(), points_w, self._segment_length)

    def axial_compression(self) -> torch.Tensor:
        """Axial force in every stem joint, + = compression, - = tension, shape (num_envs, num_segments - 1) [N].

        Read from the model's joint wrenches (`joint_wrenches(stem)`, `chain` only; the parked `cable` has none), i.e.
        the force each segment receives through its incoming joint along the stem axis. The base clamp is left out.
        The force is the same in every frame, so no transform to the robot base.
        """
        return self._model.joint_wrenches(self._stem)[:, 1:, 0]

    def segment_arc_lengths(self) -> torch.Tensor:
        """Arc length of every segment's centre from the stem base, shape (num_segments,) [m]: (i + 1/2) x the segment
        length (0 at the base, the stem length at the tip, as `points_b`). Places a per-segment reading, e.g. the
        fork's contact force, on the stem (the contact point, `metrics.py`)."""
        poses = self._segment_poses()
        index = torch.arange(poses.shape[1], dtype=poses.dtype, device=poses.device)
        return (index + 0.5) * self._segment_length

    def _segment_poses(self) -> torch.Tensor:
        """Segment poses (num_envs, num_segments, 7), world frame; read anew at every call (a reset can write the stem
        within a step)."""
        return self._model.segment_poses(self._stem)


def _world_to_base(robot: Articulation, points_w: torch.Tensor) -> torch.Tensor:
    """World-frame points (num_envs, num_points, 3) [m] in `robot`'s base frame, same shape."""
    num_points = points_w.shape[1]
    root_pos = robot.data.root_pos_w.torch.repeat_interleave(num_points, dim=0)
    root_quat = robot.data.root_quat_w.torch.repeat_interleave(num_points, dim=0)
    points_b, _ = subtract_frame_transforms(root_pos, root_quat, points_w.reshape(-1, 3))
    return points_b.reshape(-1, num_points, 3)


def _base_to_world(robot: Articulation, points_b: torch.Tensor) -> torch.Tensor:
    """Points (num_envs, num_points, 3) [m] in `robot`'s base frame in the world frame, same shape."""
    num_points = points_b.shape[1]
    root_pos = robot.data.root_pos_w.torch.repeat_interleave(num_points, dim=0)
    root_quat = robot.data.root_quat_w.torch.repeat_interleave(num_points, dim=0)
    points_w, _ = combine_frame_transforms(root_pos, root_quat, points_b.reshape(-1, 3))
    return points_w.reshape(-1, num_points, 3)


@per_env
def stem_state(env: ManagerBasedEnv) -> StemState:
    """The env's stem state, built on first use from the stem model named in `env.cfg.stem_model` and the scene
    entities `STEM` and `ROBOT`."""
    return StemState(stem_model(env.cfg.stem_model), env.scene[STEM], env.scene[ROBOT])


def stem_points(env: ManagerBasedEnv, arc_lengths: Sequence[float]) -> torch.Tensor:
    """Observation: points on the stem at `arc_lengths` [m], robot base frame, (num_envs, 3 * len) [m], raw meters
    (x, y, z of the first point, then the next)."""
    return stem_state(env).points_b(arc_lengths).flatten(start_dim=1)
