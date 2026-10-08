"""Action term: relative tool-tip pose (position and rotation step), executed by Franka's Cartesian impedance law.

The policy outputs a tool-tip translation and rotation per policy step (robot base frame); the term integrates them
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
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply,
    quat_inv,
    quat_mul,
    skew_symmetric_matrix,
    subtract_frame_transforms,
)

from stem_manip.utils.impedance import (
    apparent_mass_damping,
    cartesian_impedance_torque,
    integrate_orientation,
    integrate_target,
    limit_torque_rate,
    rotvec_between,
    substep_command,
)

from .actions_cfg import ToolTipImpedanceActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ToolTipImpedanceAction(ActionTerm):
    """Relative tool-tip pose action with a Cartesian impedance controller (see the module docstring)."""

    cfg: ToolTipImpedanceActionCfg
    _asset: Articulation

    def __init__(self, cfg: ToolTipImpedanceActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._joint_ids, _ = self._asset.find_joints(cfg.joint_names)
        body_ids, body_names = self._asset.find_bodies(cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one body for '{cfg.body_name}', found {body_names}")
        self._body_idx = body_ids[0]
        self._jacobi_body_idx = self._body_idx - 1 if self._asset.is_fixed_base else self._body_idx
        self._jacobi_joint_ids = [j + self._asset.num_base_dofs for j in self._joint_ids]

        n, device = self.num_envs, self.device
        self._offset_pos, self._offset_quat = (torch.tensor(v, device=device).repeat(n, 1) for v in cfg.tool_offset)
        self._q_ns = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        self._raw_actions = torch.zeros(n, 6, device=device)
        self._step = torch.zeros(n, 6, device=device)  # processed action: translation [m] and rotation vector [rad]
        self._target = torch.zeros(n, 3, device=device)  # target position at the end of the policy step
        self._target_start = torch.zeros(n, 3, device=device)  # target position at the start of the policy step
        self._target_quat = torch.zeros(n, 4, device=device)  # target orientation at the end of the policy step
        self._target_quat_start = torch.zeros(n, 4, device=device)
        self._command_pos = torch.zeros(n, 3, device=device)  # target the law used at the latest physics step
        self._command_quat = torch.zeros(n, 4, device=device)
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
        """Tool-tip position (n, 3) and orientation (n, 4) in the robot base frame."""
        data = self._asset.data
        pos, quat = subtract_frame_transforms(
            data.root_pos_w.torch,
            data.root_quat_w.torch,
            data.body_pos_w.torch[:, self._body_idx],
            data.body_quat_w.torch[:, self._body_idx],
        )
        return combine_frame_transforms(pos, quat, self._offset_pos, self._offset_quat)

    def target(self) -> torch.Tensor:
        """Current target position (n, 3) in the robot base frame (end of the current policy step)."""
        return self._target

    def target_quat(self) -> torch.Tensor:
        """Current target orientation (n, 4) (x, y, z, w) in the robot base frame (end of the current policy step)."""
        return self._target_quat

    def command_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Target position (n, 3) and orientation (n, 4) the law used at the latest physics step (base frame)."""
        return self._command_pos, self._command_quat

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

    def apply_actions(self):
        self._substep = min(self._substep + 1, self._decimation)
        cfg = self.cfg
        data = self._asset.data
        pos, quat = self.tool_pose()
        jacobian = self._tool_jacobian()
        dq = data.joint_vel.torch[:, self._joint_ids]
        twist = (jacobian @ dq.unsqueeze(-1)).squeeze(-1)
        target_pos, target_quat, target_vel, target_ang_vel = substep_command(
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
        if self._substep == self._decimation:
            # end of the policy step: the target stays where the clamp held it; applied step = its actual motion
            self._target[:], self._target_quat[:] = target_pos, target_quat
            self._step[:, :3] = self._target - self._target_start
            self._step[:, 3:] = rotvec_between(self._target_quat, self._target_quat_start)
        damping = None
        if cfg.damping == "apparent_mass":
            # PhysX's mass matrix excludes the joint armature (motor inertia); add it
            joints = self._jacobi_joint_ids
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
        self._raw_actions[env_ids] = 0.0
        self._tau_prev[env_ids] = 0.0

    def _tool_jacobian(self) -> torch.Tensor:
        """Geometric Jacobian of the tool tip in the base frame (n, 6, num_joints)."""
        data = self._asset.data
        jacobian = data.body_link_jacobian_w.torch[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        base_quat_inv = quat_inv(data.root_quat_w.torch)
        base_rot = matrix_from_quat(base_quat_inv)  # world -> base
        jacobian = torch.cat((base_rot @ jacobian[:, :3], base_rot @ jacobian[:, 3:]), dim=1)
        # shift the linear rows from the body origin to the tool tip: v_tip = v_body + w x r = v_body - [r]x w
        body_quat_b = quat_mul(base_quat_inv, data.body_quat_w.torch[:, self._body_idx])
        lever = quat_apply(body_quat_b, self._offset_pos)
        jacobian[:, :3] += torch.bmm(-skew_symmetric_matrix(lever), jacobian[:, 3:])
        return jacobian
