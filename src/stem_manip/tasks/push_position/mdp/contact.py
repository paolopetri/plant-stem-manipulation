"""Robot-stem contact forces: the one place where they are read (rewards, terminations and metrics only, never
observed; user, 2026-10-06).

The stem contact sensor reports, at every physics step of the policy step (its history), the force between each stem
segment and each robot body of its filter. The filter lists the fork first, then the other robot bodies (the arm
links; env cfg), so the robot-body axis splits into the fork (`FORK_BODY`) and the arm (arm contact, `GLOSSARY.md`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

FORK_BODY = 0  # index of the fork on the sensor's robot-body axis: first in the filter


def contact_force_pairs(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Magnitude of the force (normal + friction) between each stem segment and each robot body at each physics step
    of the policy step, (num_envs, history, segments, bodies) [N]; bodies in the filter's order (`FORK_BODY` first)."""
    data = env.scene.sensors[sensor_cfg.name].data
    force = data.normal_force_matrix_w_history.torch + data.friction_force_matrix_w_history.torch
    return force.norm(dim=-1)


def fork_segment_force_history(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Force of the fork on each stem segment at each physics step of the policy step, (num_envs, history,
    segments) [N]."""
    return contact_force_pairs(env, sensor_cfg)[..., FORK_BODY]


def _policy_step_force(pairs: torch.Tensor) -> torch.Tensor:
    """Largest single contact of `pairs` (num_envs, history, segments, bodies) per physics step, averaged over the
    policy step, (num_envs,) [N]."""
    return pairs.amax(dim=(2, 3)).mean(dim=1)


def robot_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Largest single contact force of the robot on the stem, (num_envs,) [N]: at each physics step the magnitude of
    the force (normal + friction) between each stem segment and each robot body (fork, arm links), its maximum over
    the pairs, averaged over the sensor's history (one policy step). The maximum stands for the local load (no-damage
    constraint; opposing contacts do not cancel as in a net force; user, 2026-10-09); the average keeps single-step
    contact spikes from deciding a termination."""
    return _policy_step_force(contact_force_pairs(env, sensor_cfg))


def fork_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Largest single contact force of the fork on the stem, (num_envs,) [N]: the reading of `robot_contact_force`
    over the fork only."""
    return _policy_step_force(contact_force_pairs(env, sensor_cfg)[..., FORK_BODY : FORK_BODY + 1])


def arm_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Largest single contact force of the robot bodies other than the fork on the stem, (num_envs,) [N]: the reading
    of `robot_contact_force` over the arm's bodies only."""
    pairs = contact_force_pairs(env, sensor_cfg)
    arm = torch.ones(pairs.shape[-1], dtype=torch.bool, device=pairs.device)
    arm[FORK_BODY] = False
    return _policy_step_force(pairs[..., arm])
