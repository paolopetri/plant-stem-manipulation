"""Observation terms of the slot base and the action term (the stem-state terms are in `stem_state.py`).

Slot-base position and orientation, the applied action step and the setpoint offset (the action term's state: with the
step-change limit and the setpoint clamp the raw action alone does not tell the policy what was executed; a limiter the
policy cannot see breaks learning, Aljalbout et al., RA-L 2024).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import matrix_from_quat

from stem_manip.utils.impedance import rotvec_between

from .slot_base import slot_base

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions import SlotBaseImpedanceAction


def slot_base_pos(env: ManagerBasedEnv) -> torch.Tensor:
    """Slot-base position in the robot root frame (num_envs, 3) [m]."""
    return slot_base(env).pose()[0]


def slot_base_rot6d(env: ManagerBasedEnv) -> torch.Tensor:
    """Slot-base orientation as the first two columns of its rotation matrix (num_envs, 6): the tool's x and y axes in
    the robot base frame. Continuous, unlike quaternions (q and -q); Zhou et al., CVPR 2019 (decision 2026-10-07)."""
    rotation = matrix_from_quat(slot_base(env).pose()[1])
    return torch.cat((rotation[..., 0], rotation[..., 1]), dim=-1)


def applied_step(env: ManagerBasedEnv, action_name: str = "slot_base") -> torch.Tensor:
    """Step the action term applied in the last policy step (num_envs, 6): translation / max_step and rotation
    vector / max_rot_step, so in [-1, 1] except where the clamp dragged the setpoint along with a tool that moved
    faster than the cap (the applied step is the setpoint's actual motion). Differs from the raw action where the
    step-change limit or the clamp acted."""
    term: SlotBaseImpedanceAction = env.action_manager.get_term(action_name)
    step = term.processed_actions
    return torch.cat((step[:, :3] / term.cfg.max_step, step[:, 3:] / term.cfg.max_rot_step), dim=-1)


def setpoint_offset(env: ManagerBasedEnv, action_name: str = "slot_base") -> torch.Tensor:
    """Setpoint pose relative to the measured slot base (num_envs, 6), robot base frame: position offset /
    max_setpoint_offset and rotation vector (slot base -> setpoint) / max_setpoint_rot_offset, so about [-1, 1]:
    the clamp holds the bound at its physics step, this is measured after it (the tool may have moved slightly since).
    The spring force of the impedance law is proportional to it."""
    term: SlotBaseImpedanceAction = env.action_manager.get_term(action_name)
    pos, quat = term.slot_base_pose()
    rot = rotvec_between(term.setpoint_quat(), quat)
    return torch.cat(
        ((term.setpoint() - pos) / term.cfg.max_setpoint_offset, rot / term.cfg.max_setpoint_rot_offset), dim=-1
    )
