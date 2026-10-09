"""Reward terms (decided values: user, 2026-10-09; weights and parameters in the env cfg).

- Task: distance of the stem point (the tip) to the target, `1 - tanh(d / std)`, coarse (5 cm) and fine (1 cm);
  height error alone (`stem_point_height_tanh`, weight 0 for the baseline: against a side-push optimum, a prong's
  side only reaches the bowl above deep targets). Same point as the command's `position_error`.
- Approach: `1 - tanh(d / std)` of the tool tip's distance to the nearest point on the stem (exploration: the task
  terms are flat until the fork touches the stem; saturates in contact, does not say where to push).
- Damage below the hard limits: curvature, quadratic above `soft_fraction` x `damage.max_curvature`; contact force
  of the robot on the stem (largest single contact), quadratic above a free threshold. Twist and tension once their
  limits are set.
- Smooth motion: Isaac Lab's `action_rate_l2`; early terminations: Isaac Lab's `is_terminated`.

Contact forces are used here and in the terminations only, never observed (decision 2026-10-06).
Verify: `scripts/check_push_terms.py`. See docs/TODO.md -> M4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import tool_tip_pos
from .stem_state import stem_curvature, stem_distance_b

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import StemTipTargetCommand


def stem_point_error(env: ManagerBasedRLEnv, command_name: str = "stem_target") -> torch.Tensor:
    """Stem point minus target (num_envs, 3), robot base frame [m]."""
    term: StemTipTargetCommand = env.command_manager.get_term(command_name)
    return term.stem_point_b() - term.command


def stem_point_distance_tanh(env: ManagerBasedRLEnv, std: float, command_name: str = "stem_target") -> torch.Tensor:
    """`1 - tanh(d / std)` of the stem point's distance d to the target [m], in (0, 1]."""
    return 1.0 - torch.tanh(stem_point_error(env, command_name).norm(dim=-1) / std)


def stem_point_height_tanh(env: ManagerBasedRLEnv, std: float, command_name: str = "stem_target") -> torch.Tensor:
    """`1 - tanh(|dz| / std)` of the stem point's height error dz [m] (robot base frame z), in (0, 1]."""
    return 1.0 - torch.tanh(stem_point_error(env, command_name)[:, 2].abs() / std)


def approach_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    model: str,
    body_name: str,
    offset: tuple[tuple[float, ...], tuple[float, ...]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("stem"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """`1 - tanh(d / std)` of the tool tip's distance d to the stem's centre line [m], in (0, 1]."""
    tip_b = tool_tip_pos(env, body_name, offset, robot_cfg)
    return 1.0 - torch.tanh(stem_distance_b(env, tip_b.unsqueeze(1), model, asset_cfg, robot_cfg)[:, 0] / std)


def curvature_penalty(
    env: ManagerBasedRLEnv,
    model: str,
    max_curvature: float,
    soft_fraction: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("stem"),
) -> torch.Tensor:
    """Sum over the joints of `max(0, kappa - soft_fraction * max_curvature)^2` [1/m^2]; zero for every target shape
    (they stay within 0.8 x the limit)."""
    excess = (stem_curvature(env, model, asset_cfg) - soft_fraction * max_curvature).clamp(min=0.0)
    return excess.square().sum(dim=-1)


def robot_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Largest single contact force of the robot on the stem, (num_envs,) [N]: at each physics step the magnitude of
    the force (normal + friction) between each stem segment and each robot body (fork, arm links), its maximum over
    the pairs, averaged over the sensor's history (one policy step). The maximum stands for the local load (no-damage
    constraint; opposing contacts do not cancel as in a net force; user, 2026-10-09); the average keeps single-step
    contact spikes from deciding a termination."""
    data = env.scene.sensors[sensor_cfg.name].data
    force = data.normal_force_matrix_w_history.torch + data.friction_force_matrix_w_history.torch
    # (envs, history, segments, bodies, 3) -> magnitude per pair -> max over the pairs -> mean over the history
    return force.norm(dim=-1).amax(dim=(2, 3)).mean(dim=1)


def contact_force_penalty(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """`max(0, F - threshold)^2` of the robot's contact force F on the stem [N^2] (`robot_contact_force`)."""
    return (robot_contact_force(env, sensor_cfg) - threshold).clamp(min=0.0).square()
