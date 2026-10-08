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


def quat_from_rotvec(rotvec: torch.Tensor) -> torch.Tensor:
    """Unit quaternions (n, 4) (x, y, z, w) of rotation vectors (n, 3) (axis * angle [rad])."""
    angle = rotvec.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    # sin(a/2)/a -> 1/2 for a -> 0 (series), keeps small and zero rotations exact
    scale = torch.where(angle > 1e-6, torch.sin(half) / angle.clamp(min=1e-12), 0.5 - angle**2 / 48.0)
    return torch.cat((rotvec * scale, torch.cos(half)), dim=-1)


def rotvec_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Rotation vectors (n, 3) (axis * angle [rad], angle in [0, pi]) of unit quaternions (n, 4) (x, y, z, w)."""
    quat = torch.where(quat[..., 3:] < 0.0, -quat, quat)  # w >= 0: the shorter rotation
    sin_half = quat[..., :3].norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, quat[..., 3:])
    # angle / sin(a/2) -> 2 for a -> 0, keeps small and zero rotations exact
    scale = torch.where(sin_half > 1e-6, angle / sin_half.clamp(min=1e-12), 2.0 + angle**2 / 12.0)
    return quat[..., :3] * scale


def rotvec_between(quat_a: torch.Tensor, quat_b: torch.Tensor) -> torch.Tensor:
    """Rotation vector (n, 3) [rad], base frame, that turns orientation b into orientation a (a = R b)."""
    return rotvec_from_quat(_quat_mul(quat_a, _quat_conj(quat_b)))


def _limit_norm(vector: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Vectors (n, 3) scaled down to length `max_norm` where they are longer (direction kept)."""
    return vector * (max_norm / vector.norm(dim=-1, keepdim=True).clamp(min=max_norm))


def _limited_step(
    action: torch.Tensor, max_step: float, prev_step: torch.Tensor | None, max_change: float | None
) -> torch.Tensor:
    """Action (n, 3) clipped to [-1, 1] per axis, scaled by max_step, its length limited to max_step; then, if
    `prev_step` and `max_change` are given, its change from `prev_step` limited to length `max_change`. The cap has
    priority: the result is at most max_step long even after a longer `prev_step` (e.g. where the clamp dragged the
    target)."""
    step = _limit_norm(action.clamp(-1.0, 1.0) * max_step, max_step)
    if prev_step is not None and max_change is not None:
        step = _limit_norm(prev_step + _limit_norm(step - prev_step, max_change), max_step)
    return step


def integrate_target(
    target: torch.Tensor,
    action: torch.Tensor,
    max_step: float,
    prev_step: torch.Tensor | None = None,
    max_change: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """New target = previous target + step (IndustReal's policy-level action integrator, PLAI).

    The action (n, 3) is clipped to [-1, 1] per axis and scaled by `max_step` [m]; the step's length is then
    limited to `max_step` (speed cap). With `prev_step` (n, 3) [m] and `max_change` the change of the step per policy
    step is limited to length `max_change` [m] (acceleration limit: keeps the tracking error small; Overleaf
    controller section). Returns (new target, step).
    """
    step = _limited_step(action, max_step, prev_step, max_change)
    return target + step, step


def integrate_orientation(
    target_quat: torch.Tensor,
    action: torch.Tensor,
    max_rot_step: float,
    prev_step: torch.Tensor | None = None,
    max_change: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """New target orientation = rotation step applied to the previous target orientation (PLAI for rotation).

    The action (n, 3) is a rotation vector in the robot base frame, clipped to [-1, 1] per axis and scaled by
    `max_rot_step` [rad]; its angle is then limited to `max_rot_step`. With `prev_step` (n, 3) [rad] and `max_change`
    the change of the rotation vector per policy step is limited to `max_change` [rad]. Returns (new target
    quaternion, rotation step (n, 3) [rad]).
    """
    step = _limited_step(action, max_rot_step, prev_step, max_change)
    quat = _quat_mul(quat_from_rotvec(step), target_quat)  # base-frame rotation: applied from the left
    return quat / quat.norm(dim=-1, keepdim=True), step


def clamp_target_offset(
    target: torch.Tensor, pos: torch.Tensor, max_offset: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Target position (n, 3) moved towards the tool position `pos` so that it is at most `max_offset` [m] away.

    Bounds the spring force of the law to K_p * max_offset when the tool is blocked (e.g. pushed against something).
    Returns (clamped target, held (n,) bool: where the clamp acted).
    """
    offset = target - pos
    distance = offset.norm(dim=-1, keepdim=True)
    held = distance > max_offset
    return torch.where(held, pos + offset * (max_offset / distance), target), held.squeeze(-1)


def clamp_target_rot_offset(
    target_quat: torch.Tensor, quat: torch.Tensor, max_angle: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Target orientation (n, 4) turned towards the tool orientation `quat` so that the rotation between them is at
    most `max_angle` [rad] (bounds the spring moment to K_o * max_angle). Returns (clamped target, held (n,) bool)."""
    offset = rotvec_between(target_quat, quat)  # base frame: tool -> target
    angle = offset.norm(dim=-1, keepdim=True)
    held = angle > max_angle
    clamped = _quat_mul(quat_from_rotvec(offset * (max_angle / angle.clamp(min=max_angle))), quat)
    return torch.where(held, clamped / clamped.norm(dim=-1, keepdim=True), target_quat), held.squeeze(-1)


def substep_command(
    target_start: torch.Tensor,
    target_quat_start: torch.Tensor,
    step: torch.Tensor,
    fraction: float,
    pos: torch.Tensor,
    quat: torch.Tensor,
    tool_vel: torch.Tensor,
    tool_ang_vel: torch.Tensor,
    policy_dt: float,
    max_offset: float,
    max_rot_offset: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Target pose and velocity the law uses at one physics (torque-loop) step within a policy step.

    The target moves linearly (turns at constant rate) from (`target_start`, `target_quat_start`) by `step` (n, 6:
    translation [m], rotation vector [rad]) over the policy step; `fraction` in (0, 1] is the elapsed part. It is then
    clamped to `max_offset` [m] / `max_rot_offset` [rad] from the measured tool pose (`pos`, `quat`): bounds the
    spring force and moment when the tool is blocked, inactive in free motion (the distance is the lag there). The
    feedforward velocity is the step's velocity; where the clamp holds the target, the target moves with the tool,
    so its velocity is the tool's (`tool_vel`, `tool_ang_vel`): the damping then adds no force to the bounded spring
    force. Runs at the physics rate in simulation and in the 1 kHz loop on the real FR3.
    Returns (target position (n, 3), target orientation (n, 4), target velocity (n, 3), target angular velocity
    (n, 3)).
    """
    target_pos, held = clamp_target_offset(target_start + fraction * step[:, :3], pos, max_offset)
    target_quat = _quat_mul(quat_from_rotvec(fraction * step[:, 3:]), target_quat_start)
    target_quat, held_rot = clamp_target_rot_offset(
        target_quat / target_quat.norm(dim=-1, keepdim=True), quat, max_rot_offset
    )
    target_vel = torch.where(held.unsqueeze(-1), tool_vel, step[:, :3] / policy_dt)
    target_ang_vel = torch.where(held_rot.unsqueeze(-1), tool_ang_vel, step[:, 3:] / policy_dt)
    return target_pos, target_quat, target_vel, target_ang_vel


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
    target_ang_vel: torch.Tensor | None = None,
) -> torch.Tensor:
    """Joint torques (n, num_joints) of Franka's Cartesian impedance example controller.

    tau = J^T [-K_p (x - x_d) - D_p (v - v_d);  -K_o e_o - D_o (w - w_d)]
          + (I - J^T J^T+) (k_ns (q_ns - q) - 2 sqrt(k_ns) dq)
    with D = 2 sqrt(K) as in franka_ros / libfranka, e_o from the quaternion error as in their code (vector part
    of q^-1 q_d, rotated to the base frame, sign flipped), and v_d, w_d the target velocities (0 in Franka's
    example, the feedforward terms here). Gravity and Coriolis are not included: the robot compensates gravity, and the
    Coriolis term is added on the real robot from libfranka's model (not available from the simulator).

    Args:
        jacobian: tool-tip geometric Jacobian (n, 6, m). pos, quat: tool-tip pose (n, 3), (n, 4).
        twist: tool-tip linear and angular velocity (n, 6). q, dq: joint positions and velocities (n, m).
        target_pos, target_vel, target_quat: target position (n, 3), velocity (n, 3), orientation (n, 4).
        k_pos [N/m], k_rot [N m/rad], k_ns [N m/rad]: stiffnesses. q_ns: null-space rest posture (n, m).
        damping: task-space damping matrix (n, 6, 6), e.g. `apparent_mass_damping`; None = Franka's 2 sqrt(K).
        target_ang_vel: target angular velocity (n, 3) [rad/s] (rotational feedforward); None = 0 (Franka).
    """
    d_pos, d_rot, d_ns = 2.0 * k_pos**0.5, 2.0 * k_rot**0.5, 2.0 * k_ns**0.5
    quat = torch.where((quat * target_quat).sum(-1, keepdim=True) < 0.0, -quat, quat)  # same hemisphere
    error_quat = _quat_mul(_quat_conj(quat), target_quat)
    error_rot = -_quat_apply(quat, error_quat[..., :3])
    angular_error = twist[..., 3:] if target_ang_vel is None else twist[..., 3:] - target_ang_vel
    velocity_error = torch.cat((twist[..., :3] - target_vel, angular_error), dim=-1)
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
