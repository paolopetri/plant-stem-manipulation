"""Action term: relative slot-base pose (position and rotation step), executed by Franka's Cartesian impedance law.

The policy outputs a slot-base translation and rotation per policy step (robot base frame); the term integrates them
into a target pose (IndustReal's policy-level action integrator) and, every physics step, computes the joint torques
with the same law the real FR3 will run (`stem_manip.utils.impedance`). Per policy step the step's size is capped
(speed), its change from the previous step is limited (acceleration: keeps the tracking error small), and the target
is kept within a bound of the measured tool pose at every physics step (bounds the force when the tool is blocked;
inactive in free motion); the applied step is the target's actual motion after the clamp. The target moves linearly
(position) and turns at constant rate (orientation) through the policy step, and its linear and angular velocities
enter the damping term (feedforward; while the clamp holds the target, it moves with the tool and its velocity is the
tool's). The per-physics-step part is `stem_manip.utils.impedance.substep_command`, the same function the real
robot's torque loop is ported from.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm

from stem_manip.utils.impedance import (
    apparent_mass_damping,
    cartesian_impedance_torque,
    integrate_orientation,
    integrate_target,
    limit_torque_rate,
    rotvec_between,
    substep_command,
)

from .actions_cfg import SlotBaseImpedanceActionCfg
from .slot_base import SlotBaseKinematics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SlotBaseImpedanceAction(ActionTerm):
    """Relative slot-base pose action with a Cartesian impedance controller (see the module docstring)."""

    cfg: SlotBaseImpedanceActionCfg
    _asset: Articulation

    def __init__(self, cfg: SlotBaseImpedanceActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._joint_ids, _ = self._asset.find_joints(cfg.joint_names)
        self._tool = SlotBaseKinematics(self._asset, cfg.body_name, cfg.tool_offset, self._joint_ids)

        n, device = self.num_envs, self.device
        self._q_ns = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        self._raw_actions = torch.zeros(n, 6, device=device)
        self._step = torch.zeros(n, 6, device=device)  # processed action: translation [m] and rotation vector [rad]
        self._target = torch.zeros(n, 3, device=device)  # target position at the end of the policy step
        self._target_start = torch.zeros(n, 3, device=device)  # target position at the start of the policy step
        self._target_quat = torch.zeros(n, 4, device=device)  # target orientation at the end of the policy step
        self._target_quat_start = torch.zeros(n, 4, device=device)
        self._command_pos = torch.zeros(n, 3, device=device)  # target the law used at the latest physics step
        self._command_quat = torch.zeros(n, 4, device=device)
        self._clamp_held = torch.zeros(n, 2, dtype=torch.bool, device=device)  # (position, orientation) this step
        self._tau_prev = torch.zeros(n, len(self._joint_ids), device=device)
        self._decimation = env.cfg.decimation
        self._policy_dt = env.cfg.decimation * env.physics_dt
        self._substep = 0
        self._max_torque_change = cfg.torque_rate_limit * env.physics_dt
        if cfg.damping not in ("franka", "apparent_mass"):
            raise ValueError(f"Unknown damping '{cfg.damping}', expected 'franka' or 'apparent_mass'")

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._step

    def tool_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Slot-base position (n, 3) and orientation (n, 4) in the robot base frame."""
        return self._tool.pose()

    def target(self) -> torch.Tensor:
        """Current target position (n, 3) in the robot base frame (end of the current policy step)."""
        return self._target

    def target_quat(self) -> torch.Tensor:
        """Current target orientation (n, 4) (x, y, z, w) in the robot base frame (end of the current policy step)."""
        return self._target_quat

    def command_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Target position (n, 3) and orientation (n, 4) the law used at the latest physics step (base frame)."""
        return self._command_pos, self._command_quat

    def clamp_held(self) -> torch.Tensor:
        """Whether the clamp held the target in any physics step of the latest policy step (n, 2) bool: position,
        orientation. Where it held, the applied step may exceed the speed cap (the clamp drags the target along)."""
        return self._clamp_held

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._target_start[:] = self._target
        self._target_quat_start[:] = self._target_quat
        cfg = self.cfg
        # commanded step; the clamp acts per physics step (`apply_actions`), which sets the applied step at the end
        self._target[:], self._step[:, :3] = integrate_target(
            self._target, actions[:, :3], cfg.max_step, self._step[:, :3], cfg.max_step_change
        )
        self._target_quat[:], self._step[:, 3:] = integrate_orientation(
            self._target_quat, actions[:, 3:], cfg.max_rot_step, self._step[:, 3:], cfg.max_rot_step_change
        )
        self._substep = 0
        self._clamp_held[:] = False

    def apply_actions(self):
        self._substep = min(self._substep + 1, self._decimation)
        cfg = self.cfg
        data = self._asset.data
        pos, quat = self.tool_pose()
        jacobian = self._tool.jacobian()
        dq = data.joint_vel.torch[:, self._joint_ids]
        twist = (jacobian @ dq.unsqueeze(-1)).squeeze(-1)
        target_pos, target_quat, target_vel, target_ang_vel, held = substep_command(
            self._target_start,
            self._target_quat_start,
            self._step,
            self._substep / self._decimation,
            pos,
            quat,
            twist[:, :3],
            twist[:, 3:],
            self._policy_dt,
            cfg.max_target_offset,
            cfg.max_target_rot_offset,
        )
        self._command_pos[:], self._command_quat[:] = target_pos, target_quat
        self._clamp_held |= held
        if self._substep == self._decimation:
            # end of the policy step: the target stays where the clamp held it; applied step = its actual motion
            self._target[:], self._target_quat[:] = target_pos, target_quat
            self._step[:, :3] = self._target - self._target_start
            self._step[:, 3:] = rotvec_between(self._target_quat, self._target_quat_start)
        damping = None
        if cfg.damping == "apparent_mass":
            # PhysX's mass matrix excludes the joint armature (motor inertia); add it
            joints = self._tool.jacobian_joint_ids
            mass_matrix = data.mass_matrix.torch[:, joints][:, :, joints]
            mass_matrix = mass_matrix + torch.diag_embed(data.joint_armature.torch[:, self._joint_ids])
            damping = apparent_mass_damping(jacobian, mass_matrix, cfg.stiffness_pos, cfg.stiffness_rot)
        tau = cartesian_impedance_torque(
            jacobian,
            pos,
            quat,
            twist,
            data.joint_pos.torch[:, self._joint_ids],
            dq,
            target_pos,
            target_vel,
            target_quat,
            cfg.stiffness_pos,
            cfg.stiffness_rot,
            cfg.stiffness_nullspace,
            self._q_ns,
            damping,
            target_ang_vel,
        )
        tau = limit_torque_rate(tau, self._tau_prev, self._max_torque_change)
        self._tau_prev[:] = tau
        self._asset.set_joint_effort_target_index(target=tau, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Start the target at the tool's current pose (link poses are up to date after the reset event)."""
        env_ids = slice(None) if env_ids is None else env_ids
        pos, quat = self.tool_pose()
        self._target[env_ids] = pos[env_ids]
        self._target_start[env_ids] = pos[env_ids]
        self._target_quat[env_ids] = quat[env_ids]
        self._target_quat_start[env_ids] = quat[env_ids]
        self._command_pos[env_ids] = pos[env_ids]
        self._command_quat[env_ids] = quat[env_ids]
        self._step[env_ids] = 0.0
        self._clamp_held[env_ids] = False
        self._raw_actions[env_ids] = 0.0
        self._tau_prev[env_ids] = 0.0
