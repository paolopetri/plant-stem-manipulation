"""Target command for the stem point (stage 1): a position for the stem's tip, sampled once per episode.

The stem point is given by a segment index + an offset along that segment (task cfg; the tip, user 2026-10-08).
At reset the target is sampled around the point's rest position (`stem_manip.utils.stem_target`): horizontal
distance and direction uniform, height on or below the "bowl" the point reaches when pushed, down to the deepest
stem shape within `curvature_budget` x `damage.max_curvature` (user, 2026-10-09). The rest position is the
stem base plus the point's arc length straight up (assumes an upright stem and an upright robot base; the tilted
stem in docs/TODO.md "Later" needs a change here). The base is read from the clamped segment 0, so it follows the
spawn randomization (Isaac Lab resets the events before the command manager).

The command is the target position in the robot base frame (num_envs, 3) [m]; metric `position_error` [m] (stem
point to target). Debug markers: target (red, green within `success_distance`) and the stem point (blue).
Starting point: `CableUniformPoseCommand` (IsaacLab core/lift mdp). Verify: `scripts/check_push_obs.py`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms

from stem_manip.assets.stem import stem_params
from stem_manip.utils.stem_geometry import point_pose
from stem_manip.utils.stem_target import sample_targets

from .stem_state import segment_length, stem_points_b, stem_segment_poses, to_robot_base

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class StemTipTargetCommand(CommandTerm):
    """Target position of the stem point, robot base frame, sampled at reset."""

    cfg: StemTipTargetCommandCfg

    def __init__(self, cfg: StemTipTargetCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self._robot_cfg = SceneEntityCfg(cfg.asset_name)
        self._stem_cfg = SceneEntityCfg(cfg.stem_asset_name)
        self.arc_length = (cfg.segment_index + 0.5) * segment_length(cfg.stem_model) + cfg.offset
        """Arc length of the stem point from the base [m] (for its rest position)."""
        self.max_curvature = cfg.curvature_budget * stem_params(cfg.stem_model)["damage"]["max_curvature"]
        """Curvature of the deepest target shape [1/m]."""
        self.target_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            f"StemTipTargetCommand:\n\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tStem point: arc length {self.arc_length:.3f} m\n"
        )

    @property
    def command(self) -> torch.Tensor:
        """Target position of the stem point in the robot base frame (num_envs, 3) [m]."""
        return self.target_b

    def stem_point_b(self) -> torch.Tensor:
        """Current position of the stem point in the robot base frame (num_envs, 3) [m]."""
        poses = stem_segment_poses(self._env, self.cfg.stem_model, self._stem_cfg)
        point_w = point_pose(poses, self.cfg.segment_index, self.cfg.offset)[:, :3]
        return to_robot_base(self._env, point_w.unsqueeze(1), self._robot_cfg)[:, 0]

    def _resample_command(self, env_ids: Sequence[int]):
        base = stem_points_b(self._env, (0.0,), self.cfg.stem_model, self._stem_cfg, self._robot_cfg)[:, 0]
        rest = base[env_ids]  # stem base (clamped segment 0)
        rest[:, 2] += self.arc_length  # upright stem: the point's rest position straight above the base
        self.target_b[env_ids] = sample_targets(
            rest, self.arc_length, self.cfg.distance_range, self.cfg.angle_range, self.max_curvature
        )
        # world frame for the markers: the first render after a reset comes before the next `_update_metrics`
        self.target_w[env_ids] = self._to_world(self.target_b[env_ids], env_ids)

    def _to_world(self, points_b: torch.Tensor, env_ids: Sequence[int] | slice = slice(None)) -> torch.Tensor:
        return combine_frame_transforms(
            self.robot.data.root_pos_w.torch[env_ids], self.robot.data.root_quat_w.torch[env_ids], points_b
        )[0]

    def _update_command(self):
        pass  # the target stays fixed during the episode

    def _update_metrics(self):
        self.target_w[:] = self._to_world(self.target_b)
        self.metrics["position_error"] = (self.stem_point_b() - self.target_b).norm(dim=-1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                self.target_visualizer = VisualizationMarkers(self.cfg.target_visualizer_cfg)
                self.point_visualizer = VisualizationMarkers(self.cfg.point_visualizer_cfg)
            self.target_visualizer.set_visibility(True)
            self.point_visualizer.set_visibility(True)
        elif hasattr(self, "target_visualizer"):
            self.target_visualizer.set_visibility(False)
            self.point_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        point_w = self._to_world(self.stem_point_b())
        near = ((point_w - self.target_w).norm(dim=-1) < self.cfg.success_distance).int()
        self.target_visualizer.visualize(self.target_w, marker_indices=near)
        self.point_visualizer.visualize(point_w)


def _sphere(radius: float, color: tuple[float, float, float]) -> sim_utils.SphereCfg:
    return sim_utils.SphereCfg(radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))


@configclass
class StemTipTargetCommandCfg(CommandTermCfg):
    """Target position of the stem point; the defaults are the decided values (user, 2026-10-08)."""

    class_type: type = StemTipTargetCommand

    asset_name: str = "robot"
    """Robot whose base frame the command is expressed in."""

    stem_asset_name: str = "stem"
    """Scene name of the stem."""

    stem_model: str = MISSING
    """Stem model name (`stem_manip.assets.stem.STEM_MODELS`)."""

    segment_index: int = 19
    """Segment that carries the stem point (19 = tip segment of the 20-segment chain)."""

    offset: float = 0.01
    """Distance of the stem point from the segment centre along the stem [m] (+L/2 = the tip)."""

    distance_range: tuple[float, float] = (0.03, 0.10)
    """Horizontal distance of the target from the stem point's rest position [m]."""

    curvature_budget: float = 0.8
    """The deepest target shape bends the stem to at most this fraction of `damage.max_curvature` (user, 2026-10-09:
    0.8, i.e. 4 1/m, for clear C / S bends that need the fork's slot). The drop is sampled uniformly between the
    bowl and that shape."""

    angle_range: tuple[float, float] = (-math.pi, math.pi)
    """Horizontal direction of the target from the rest position [rad] (from +x towards +y): full circle."""

    success_distance: float = 0.01
    """Debug markers only: the target turns green within this distance of the stem point [m]."""

    target_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/stem_target",
        markers={"far": _sphere(0.01, (1.0, 0.0, 0.0)), "near": _sphere(0.01, (0.0, 1.0, 0.0))},
    )
    point_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/stem_point", markers={"point": _sphere(0.007, (0.1, 0.4, 1.0))}
    )
