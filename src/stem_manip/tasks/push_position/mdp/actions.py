"""Action term: relative slot-base pose (position and rotation step), executed by Franka's Cartesian impedance law.

The policy outputs a slot-base translation and rotation per policy step (robot base frame); the term integrates them
into a setpoint pose (IndustReal's policy-level action integrator) and, every physics step, computes the joint torques
with the same law the real FR3 will run (`stem_manip.utils.impedance`). Per policy step the step's size is capped
(speed), its change from the previous step is limited (acceleration: keeps the tracking error small), and the setpoint
is kept within a bound of the measured slot base pose at every physics step (bounds the force when the tool is blocked;
inactive in free motion); the applied step is the setpoint's actual motion after the clamp. The setpoint moves linearly
(position) and turns at constant rate (orientation) through the policy step, and its linear and angular velocities
enter the damping term (feedforward; while the clamp holds the setpoint, it moves with the tool and its velocity is the
tool's). The null-space rest posture is the arm's joint position at each reset (the start pose, with its noise), as
in Franka's example controller (`q_d_nullspace_ = q_initial`). The per-physics-step part is
`stem_manip.utils.impedance.substep_command`, the same function the real robot's torque loop is ported from.
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
    integrate_position,
    limit_torque_rate,
    rotvec_between,
    substep_command,
)

from .actions_cfg import SlotBaseImpedanceActionCfg
from .slot_base import slot_base

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SlotBaseImpedanceAction(ActionTerm):
    """Relative slot-base pose action with a Cartesian impedance controller (see the module docstring)."""

    cfg: SlotBaseImpedanceActionCfg
    _asset: Articulation

    def __init__(self, cfg: SlotBaseImpedanceActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._joint_ids, _ = self._asset.find_joints(cfg.joint_names)
        self._slot_base = slot_base(env)

        n, device = self.num_envs, self.device
        self._q_ns = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()  # set at every reset
        self._raw_actions = torch.zeros(n, 6, device=device)
        self._step = torch.zeros(n, 6, device=device)  # processed action: translation [m] and rotation vector [rad]
        self._setpoint = torch.zeros(n, 3, device=device)  # setpoint position at the end of the policy step
        self._setpoint_start = torch.zeros(n, 3, device=device)  # setpoint position at the start of the policy step
        self._setpoint_quat = torch.zeros(n, 4, device=device)  # setpoint orientation at the end of the policy step
        self._setpoint_quat_start = torch.zeros(n, 4, device=device)
        self._command_pos = torch.zeros(n, 3, device=device)  # setpoint the law used at the latest physics step
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

    def slot_base_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Slot-base position (n, 3) and orientation (n, 4) in the robot base frame."""
        return self._slot_base.pose()

    def setpoint(self) -> torch.Tensor:
        """Current setpoint position (n, 3) in the robot base frame (end of the current policy step)."""
        return self._setpoint

    def setpoint_quat(self) -> torch.Tensor:
        """Current setpoint orientation (n, 4) (x, y, z, w) in the robot base frame (end of the current policy step)."""
        return self._setpoint_quat

    def command_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Setpoint position (n, 3) and orientation (n, 4) the law used at the latest physics step (base frame)."""
        return self._command_pos, self._command_quat

    def nullspace_posture(self) -> torch.Tensor:
        """Null-space rest posture (n, joints) [rad]: the arm joints the law's null-space torque pulls towards."""
        return self._q_ns

    def clamp_held(self) -> torch.Tensor:
        """Whether the clamp held the setpoint in any physics step of the latest policy step (n, 2) bool: position,
        orientation. Where it held, the applied step may exceed the speed cap (the clamp drags the setpoint along)."""
        return self._clamp_held

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._setpoint_start[:] = self._setpoint
        self._setpoint_quat_start[:] = self._setpoint_quat
        cfg = self.cfg
        # commanded step; the clamp acts per physics step (`apply_actions`), which sets the applied step at the end
        self._setpoint[:], self._step[:, :3] = integrate_position(
            self._setpoint, actions[:, :3], cfg.max_step, self._step[:, :3], cfg.max_step_change
        )
        self._setpoint_quat[:], self._step[:, 3:] = integrate_orientation(
            self._setpoint_quat, actions[:, 3:], cfg.max_rot_step, self._step[:, 3:], cfg.max_rot_step_change
        )
        self._substep = 0
        self._clamp_held[:] = False

    def apply_actions(self):
        self._substep = min(self._substep + 1, self._decimation)
        cfg = self.cfg
        data = self._asset.data
        pos, quat = self.slot_base_pose()
        jacobian = self._slot_base.jacobian()
        dq = data.joint_vel.torch[:, self._joint_ids]
        twist = (jacobian @ dq.unsqueeze(-1)).squeeze(-1)
        setpoint_pos, setpoint_quat, setpoint_vel, setpoint_ang_vel, held = substep_command(
            self._setpoint_start,
            self._setpoint_quat_start,
            self._step,
            self._substep / self._decimation,
            pos,
            quat,
            twist[:, :3],
            twist[:, 3:],
            self._policy_dt,
            cfg.max_setpoint_offset,
            cfg.max_setpoint_rot_offset,
        )
        self._command_pos[:], self._command_quat[:] = setpoint_pos, setpoint_quat
        self._clamp_held |= held
        if self._substep == self._decimation:
            # end of the policy step: the setpoint stays where the clamp held it; applied step = its actual motion
            self._setpoint[:], self._setpoint_quat[:] = setpoint_pos, setpoint_quat
            self._step[:, :3] = self._setpoint - self._setpoint_start
            self._step[:, 3:] = rotvec_between(self._setpoint_quat, self._setpoint_quat_start)
        damping = None
        if cfg.damping == "apparent_mass":
            # PhysX's mass matrix excludes the joint armature (motor inertia); add it
            joints = self._slot_base.jacobian_joint_ids
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
            setpoint_pos,
            setpoint_vel,
            setpoint_quat,
            cfg.stiffness_pos,
            cfg.stiffness_rot,
            cfg.stiffness_nullspace,
            self._q_ns,
            damping,
            setpoint_ang_vel,
        )
        tau = limit_torque_rate(tau, self._tau_prev, self._max_torque_change)
        self._tau_prev[:] = tau
        self._asset.set_joint_effort_target_index(target=tau, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Start the setpoint at the slot base's current pose and the null-space rest posture at the arm's current
        joint position (link poses and joints are up to date after the reset events)."""
        env_ids = slice(None) if env_ids is None else env_ids
        self._q_ns[env_ids] = self._asset.data.joint_pos.torch[env_ids][:, self._joint_ids]
        pos, quat = self.slot_base_pose()
        self._setpoint[env_ids] = pos[env_ids]
        self._setpoint_start[env_ids] = pos[env_ids]
        self._setpoint_quat[env_ids] = quat[env_ids]
        self._setpoint_quat_start[env_ids] = quat[env_ids]
        self._command_pos[env_ids] = pos[env_ids]
        self._command_quat[env_ids] = quat[env_ids]
        self._step[env_ids] = 0.0
        self._clamp_held[env_ids] = False
        self._raw_actions[env_ids] = 0.0
        self._tau_prev[env_ids] = 0.0
