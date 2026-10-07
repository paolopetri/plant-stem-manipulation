"""Franka's Cartesian impedance law and the policy's target integration (torch only, no simulator).

The same functions are meant to run in simulation (action term of the push task) and to be ported 1:1 to the
real FR3 (libfranka torque loop), so that the policy sees the same controller on both sides. Background, gains and
the sweep that chose the values: docs/overleaf_folder/open_questions/impedance_action_study.tex.

Conventions: all poses, velocities and Jacobians in the robot base frame; quaternions (x, y, z, w); batched over
the first dimension (num_envs). The Jacobian is the geometric Jacobian of the tool tip, (n, 6, num_joints), rows
linear velocity then angular velocity.
"""

from __future__ import annotations

import torch


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a * b, quaternions (x, y, z, w)."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((-q[..., :3], q[..., 3:]), dim=-1)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (n, 3) by unit quaternions q (n, 4)."""
    xyz, w = q[..., :3], q[..., 3:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def integrate_target(target: torch.Tensor, action: torch.Tensor, max_step: float) -> tuple[torch.Tensor, torch.Tensor]:
    """New target = previous target + step (IndustReal's policy-level action integrator, PLAI).

    The action (n, 3) is clipped to [-1, 1] per axis and scaled by `max_step` [m]; the step's length is then
    limited to `max_step`, so that the tool can follow every step (max_step from the sweep, see module docstring).
    Returns (new target, step).
    """
    step = action.clamp(-1.0, 1.0) * max_step
    length = step.norm(dim=-1, keepdim=True)
    step = step * (max_step / length.clamp(min=max_step))
    return target + step, step


def _sqrtm_spd(matrix: torch.Tensor) -> torch.Tensor:
    """Square root of symmetric positive (semi-)definite matrices (..., k, k), via eigendecomposition."""
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    return eigenvectors @ torch.diag_embed(eigenvalues.clamp(min=0.0).sqrt()) @ eigenvectors.transpose(-1, -2)


def apparent_mass_damping(
    jacobian: torch.Tensor, mass_matrix: torch.Tensor, k_pos: float, k_rot: float, regularization: float = 1e-4
) -> torch.Tensor:
    """Task-space damping matrix (n, 6, 6) that makes the impedance critically damped in every direction and pose.

    Factorization design (Albu-Schaeffer et al., ICRA 2003): with the task-space inertia
    Lambda = (J M^-1 J^T)^-1 = A A and the stiffness K = K1 K1 (K1 = diag(sqrt K)), D = A K1 + K1 A. For an isotropic
    stiffness k this is 2 sqrt(k) Lambda^(1/2): critical damping for the inertia the tool actually feels, instead of
    Franka's D = 2 sqrt(K), which is critical only for a 1 kg (1 kg m^2) tool. `regularization` (relative to the
    largest eigenvalue of J M^-1 J^T) keeps Lambda finite near singular poses.
    """
    inverse_lambda = jacobian @ torch.linalg.solve(mass_matrix, jacobian.transpose(-1, -2))
    inverse_lambda = 0.5 * (inverse_lambda + inverse_lambda.transpose(-1, -2))
    scale = torch.linalg.eigvalsh(inverse_lambda)[..., -1:].unsqueeze(-1)
    eye6 = torch.eye(6, device=jacobian.device, dtype=jacobian.dtype)
    task_inertia = torch.linalg.inv(inverse_lambda + regularization * scale * eye6)
    sqrt_inertia = _sqrtm_spd(0.5 * (task_inertia + task_inertia.transpose(-1, -2)))
    sqrt_stiffness = torch.diag(
        torch.tensor([k_pos] * 3 + [k_rot] * 3, device=jacobian.device, dtype=jacobian.dtype).sqrt()
    )
    return sqrt_inertia @ sqrt_stiffness + sqrt_stiffness @ sqrt_inertia


def cartesian_impedance_torque(
    jacobian: torch.Tensor,
    pos: torch.Tensor,
    quat: torch.Tensor,
    twist: torch.Tensor,
    q: torch.Tensor,
    dq: torch.Tensor,
    target_pos: torch.Tensor,
    target_vel: torch.Tensor,
    target_quat: torch.Tensor,
    k_pos: float,
    k_rot: float,
    k_ns: float,
    q_ns: torch.Tensor,
    damping: torch.Tensor | None = None,
) -> torch.Tensor:
    """Joint torques (n, num_joints) of Franka's Cartesian impedance example controller.

    tau = J^T [-K_p (x - x_d) - D_p (v - v_d);  -K_o e_o - D_o w]
          + (I - J^T J^T+) (k_ns (q_ns - q) - 2 sqrt(k_ns) dq)
    with D = 2 sqrt(K) as in franka_ros / libfranka, e_o from the quaternion error as in their code (vector part
    of q^-1 q_d, rotated to the base frame, sign flipped), and v_d the target velocity (0 in Franka's example,
    the feedforward term here). Gravity and Coriolis are not included: the robot compensates gravity, and the
    Coriolis term is added on the real robot from libfranka's model (not available from the simulator).

    Args:
        jacobian: tool-tip geometric Jacobian (n, 6, m). pos, quat: tool-tip pose (n, 3), (n, 4).
        twist: tool-tip linear and angular velocity (n, 6). q, dq: joint positions and velocities (n, m).
        target_pos, target_vel, target_quat: target position (n, 3), velocity (n, 3), orientation (n, 4).
        k_pos [N/m], k_rot [N m/rad], k_ns [N m/rad]: stiffnesses. q_ns: null-space rest posture (n, m).
        damping: task-space damping matrix (n, 6, 6), e.g. `apparent_mass_damping`; None = Franka's 2 sqrt(K).
    """
    d_pos, d_rot, d_ns = 2.0 * k_pos**0.5, 2.0 * k_rot**0.5, 2.0 * k_ns**0.5
    quat = torch.where((quat * target_quat).sum(-1, keepdim=True) < 0.0, -quat, quat)  # same hemisphere
    error_quat = _quat_mul(_quat_conj(quat), target_quat)
    error_rot = -_quat_apply(quat, error_quat[..., :3])
    velocity_error = torch.cat((twist[..., :3] - target_vel, twist[..., 3:]), dim=-1)
    if damping is None:
        damping_force = torch.cat((d_pos * velocity_error[..., :3], d_rot * velocity_error[..., 3:]), dim=-1)
    else:
        damping_force = (damping @ velocity_error.unsqueeze(-1)).squeeze(-1)
    force = -k_pos * (pos - target_pos) - damping_force[..., :3]
    moment = -k_rot * error_rot - damping_force[..., 3:]
    jacobian_t = jacobian.transpose(-1, -2)
    tau_task = (jacobian_t @ torch.cat((force, moment), dim=-1).unsqueeze(-1)).squeeze(-1)
    eye = torch.eye(jacobian.shape[-1], device=jacobian.device, dtype=jacobian.dtype)
    projector = eye - jacobian_t @ torch.linalg.pinv(jacobian_t)
    tau_ns = (projector @ (k_ns * (q_ns - q) - d_ns * dq).unsqueeze(-1)).squeeze(-1)
    return tau_task + tau_ns


def limit_torque_rate(tau: torch.Tensor, tau_prev: torch.Tensor, max_change: float) -> torch.Tensor:
    """Limit the change of each joint torque per control step to `max_change` [N m] (Franka: 1 N m per 1 ms)."""
    return tau_prev + (tau - tau_prev).clamp(-max_change, max_change)
