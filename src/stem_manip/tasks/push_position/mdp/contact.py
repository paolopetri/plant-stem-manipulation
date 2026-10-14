"""Robot-stem contact forces: the one place where they are read (rewards and terminations only, never observed;
decision 2026-10-06)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
