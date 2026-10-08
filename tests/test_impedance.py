"""Unit tests of the Cartesian impedance law and the target integration (no simulator)."""

import math

import torch

from stem_manip.utils.impedance import (
    _quat_conj,
    _quat_mul,
    apparent_mass_damping,
    cartesian_impedance_torque,
    clamp_target_offset,
    clamp_target_rot_offset,
    integrate_orientation,
    integrate_target,
    limit_torque_rate,
    quat_from_rotvec,
    rotvec_between,
    rotvec_from_quat,
    substep_command,
)

N, M = 2, 7
GAINS = {"k_pos": 400.0, "k_rot": 10.0, "k_ns": 0.5}


def _state(jacobian: torch.Tensor | None = None) -> dict:
    """Tool at the origin, identity orientation, at rest; target equal to the state."""
    torch.manual_seed(0)
    jacobian = torch.randn(N, 6, M) if jacobian is None else jacobian
    q = torch.randn(N, M)
    return {
        "jacobian": jacobian,
        "pos": torch.zeros(N, 3),
        "quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(N, 1),
        "twist": torch.zeros(N, 6),
        "q": q,
        "dq": torch.zeros(N, M),
        "target_pos": torch.zeros(N, 3),
        "target_vel": torch.zeros(N, 3),
        "target_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(N, 1),
        "q_ns": q.clone(),
    }


def _task_wrench(state: dict) -> torch.Tensor:
    """Wrench the law applies at the tool: with J = [I_6 | 0] the joint torques of the first 6 joints are it."""
    tau = cartesian_impedance_torque(**state, **GAINS)
    return tau[:, :6]


def _identity_jacobian() -> torch.Tensor:
    return torch.cat((torch.eye(6), torch.zeros(6, 1)), dim=1).repeat(N, 1, 1)


def test_zero_error_zero_torque():
    """Tool at its target, at rest, posture at rest: no torque."""
    tau = cartesian_impedance_torque(**_state(), **GAINS)
    assert torch.allclose(tau, torch.zeros(N, M), atol=1e-5)


def test_spring_pulls_towards_target():
    """Target 1 cm ahead in x: force K_p * 0.01 along +x, nothing else."""
    state = _state(_identity_jacobian())
    state["target_pos"][:, 0] = 0.01
    wrench = _task_wrench(state)
    assert torch.allclose(wrench[:, 0], torch.full((N,), 4.0), atol=1e-5)
    assert torch.allclose(wrench[:, 1:], torch.zeros(N, 5), atol=1e-5)


def test_damping_opposes_velocity_and_feedforward_cancels_it():
    """Moving at 0.1 m/s: damping force -2 sqrt(K) * 0.1; with the same target velocity it vanishes."""
    state = _state(_identity_jacobian())
    state["twist"][:, 1] = 0.1
    assert torch.allclose(_task_wrench(state)[:, 1], torch.full((N,), -2.0 * math.sqrt(400.0) * 0.1), atol=1e-5)
    state["target_vel"][:, 1] = 0.1
    assert torch.allclose(_task_wrench(state)[:, 1], torch.zeros(N), atol=1e-5)


def test_rotational_spring_turns_towards_target():
    """Target rotated by +0.2 rad about z: positive moment about z, K_o sin(0.1) (Franka's quaternion error)."""
    state = _state(_identity_jacobian())
    state["target_quat"] = torch.tensor([[0.0, 0.0, math.sin(0.1), math.cos(0.1)]]).repeat(N, 1)
    wrench = _task_wrench(state)
    assert torch.allclose(wrench[:, 5], torch.full((N,), 10.0 * math.sin(0.1)), atol=1e-5)
    assert torch.allclose(wrench[:, 3:5], torch.zeros(N, 2), atol=1e-5)


def test_nullspace_does_not_move_the_tool():
    """The posture torque lies in the null space of J^T: it produces no tool wrench (J tau_ns-direction check)."""
    state = _state()
    state["q_ns"] = state["q"] + 0.3
    tau = cartesian_impedance_torque(**state, **GAINS)
    # torques in the null space of J^T satisfy pinv(J^T) tau = 0, i.e. no task-space force
    assert torch.allclose(torch.linalg.pinv(state["jacobian"].transpose(1, 2)) @ tau.unsqueeze(-1), torch.zeros(N, 6, 1), atol=1e-4)
    assert tau.abs().max() > 1e-3


def test_integrate_target_limits_the_step():
    """Per-axis actions are scaled by max_step, and the step's length never exceeds max_step."""
    target = torch.zeros(N, 3)
    new, step = integrate_target(target, torch.tensor([[0.5, 0.0, 0.0], [1.0, 1.0, 1.0]]), 0.00125)
    assert torch.allclose(step[0], torch.tensor([0.000625, 0.0, 0.0]))
    assert math.isclose(float(step[1].norm()), 0.00125, rel_tol=1e-6)
    assert torch.allclose(new, step)
    _, step = integrate_target(target, torch.full((N, 3), 5.0), 0.00125)  # out-of-range actions are clipped
    assert torch.all(step.norm(dim=-1) <= 0.00125 + 1e-9)


def test_torque_rate_limit():
    tau = limit_torque_rate(torch.tensor([[5.0, -5.0, 0.5]]), torch.zeros(1, 3), 2.0)
    assert torch.allclose(tau, torch.tensor([[2.0, -2.0, 0.5]]))


def test_apparent_mass_damping_is_critical():
    """J = I, M = diag(m): Lambda = diag(m), D = 2 sqrt(k m) per axis (critical damping of mass m on spring k)."""
    masses = torch.tensor([2.0, 3.0, 5.0, 0.1, 0.2, 0.3])
    jacobian = torch.eye(6).repeat(N, 1, 1)
    damping = apparent_mass_damping(jacobian, torch.diag(masses).repeat(N, 1, 1), 400.0, 10.0, regularization=0.0)
    stiffness = torch.tensor([400.0] * 3 + [10.0] * 3)
    assert torch.allclose(damping, torch.diag(2.0 * (stiffness * masses).sqrt()).repeat(N, 1, 1), atol=1e-4)


def test_damping_matrix_replaces_franka_damping():
    """With D = diag(2 sqrt(K)) passed explicitly, the torque equals Franka's default."""
    state = _state()
    state["twist"] = torch.randn(N, 6)
    state["target_vel"] = torch.randn(N, 3)
    franka = torch.diag(torch.tensor([2 * 20.0] * 3 + [2 * math.sqrt(10.0)] * 3)).repeat(N, 1, 1)
    tau_default = cartesian_impedance_torque(**state, **GAINS)
    tau_matrix = cartesian_impedance_torque(**state, **GAINS, damping=franka)
    assert torch.allclose(tau_default, tau_matrix, atol=1e-4)


def test_quat_from_rotvec():
    """0.3 rad about z; zero rotation gives the identity."""
    q = quat_from_rotvec(torch.tensor([[0.0, 0.0, 0.3], [0.0, 0.0, 0.0]]))
    assert torch.allclose(q[0], torch.tensor([0.0, 0.0, math.sin(0.15), math.cos(0.15)]), atol=1e-6)
    assert torch.allclose(q[1], torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_integrate_orientation_limits_and_composes_in_base_frame():
    """Steps are limited by angle; a base-frame step is applied from the left (q_new = dq * q)."""
    start = quat_from_rotvec(torch.tensor([[math.pi, 0.0, 0.0]]))  # fork flipped (tool z down)
    new, step = integrate_orientation(start, torch.tensor([[0.0, 0.0, 1.0]]), math.radians(2.0))
    assert math.isclose(float(step.norm()), math.radians(2.0), rel_tol=1e-6)
    expected = quat_from_rotvec(torch.tensor([[0.0, 0.0, math.radians(2.0)]]))
    from stem_manip.utils.impedance import _quat_mul

    assert torch.allclose(new, _quat_mul(expected, start), atol=1e-6)
    _, step = integrate_orientation(start, torch.full((1, 3), 5.0), math.radians(2.0))
    assert float(step.norm()) <= math.radians(2.0) + 1e-9


def test_rotational_feedforward_cancels_damping():
    """Turning at 0.5 rad/s about z: damping moment -2 sqrt(K_o) * 0.5; with the same target rate it vanishes."""
    state = _state(_identity_jacobian())
    state["twist"][:, 5] = 0.5
    wrench = _task_wrench(state)
    assert torch.allclose(wrench[:, 5], torch.full((N,), -2.0 * math.sqrt(10.0) * 0.5), atol=1e-5)
    tau = cartesian_impedance_torque(**state, **GAINS, target_ang_vel=torch.tensor([[0.0, 0.0, 0.5]]).repeat(N, 1))
    assert torch.allclose(tau[:, 5], torch.zeros(N), atol=1e-5)


def test_step_change_is_limited_and_reaches_the_desired_step():
    """From rest, a full action ramps the step up by at most max_change per policy step until max_step; a zero
    action then ramps it down the same way (the target brakes instead of stopping at once)."""
    max_step, max_change = 0.0064, 0.0005
    target, step = torch.zeros(1, 3), torch.zeros(1, 3)
    full = torch.tensor([[1.0, 0.0, 0.0]])
    steps = []
    for _ in range(20):
        new_target, new_step = integrate_target(target, full, max_step, step, max_change)
        assert float((new_step - step).norm()) <= max_change + 1e-9
        target, step = new_target, new_step
        steps.append(float(step[0, 0]))
    assert math.isclose(steps[0], max_change, rel_tol=1e-6)
    assert math.isclose(steps[-1], max_step, rel_tol=1e-6)
    _, braked = integrate_target(target, torch.zeros(1, 3), max_step, step, max_change)
    assert math.isclose(float(braked[0, 0]), max_step - max_change, rel_tol=1e-6)


def test_rotation_step_change_is_limited():
    """The rotation vector's change per policy step is limited as well (here a reversal about z)."""
    max_rot, max_change = math.radians(1.44), math.radians(0.1)
    prev = torch.tensor([[0.0, 0.0, max_rot]])
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    _, step = integrate_orientation(identity, torch.tensor([[0.0, 0.0, -1.0]]), max_rot, prev, max_change)
    assert torch.allclose(step, prev - torch.tensor([[0.0, 0.0, max_change]]), atol=1e-9)


def test_speed_cap_holds_after_a_longer_previous_step():
    """The cap has priority over the step-change limit: after a step longer than the cap (the clamp dragged the
    target along with a fast tool), the next step is at most the cap, for translation and rotation."""
    prev = torch.tensor([[0.01, 0.0, 0.0], [0.0, 0.0, -0.008]])
    action = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    _, step = integrate_target(torch.zeros(N, 3), action, 0.0064, prev, 1e-4)
    assert (step.norm(dim=-1) <= 0.0064 + 1e-9).all()
    _, rot_step = integrate_orientation(torch.tensor([[0.0, 0.0, 0.0, 1.0]] * N), action, 0.02513, 3 * prev, 0.000349)
    assert (rot_step.norm(dim=-1) <= 0.02513 + 1e-9).all()


def test_rotvec_from_quat_inverts_quat_from_rotvec():
    """Round trip for zero, small and large rotations; q and -q give the same rotation vector."""
    rotvec = torch.tensor([[0.0, 0.0, 0.0], [1e-8, 0.0, 0.0], [0.3, -0.2, 0.1], [0.0, 3.0, 0.0]])
    quat = quat_from_rotvec(rotvec)
    assert torch.allclose(rotvec_from_quat(quat), rotvec, atol=1e-6)
    assert torch.allclose(rotvec_from_quat(-quat), rotvec, atol=1e-6)


def test_clamp_target_offset():
    """Offsets inside the bound are kept; outside, the target is pulled back along the offset to the bound."""
    pos = torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    target = pos + torch.tensor([[0.005, 0.0, 0.0], [0.0, 0.03, 0.04]])
    clamped, held = clamp_target_offset(target, pos, 0.01)
    assert held.tolist() == [False, True]
    assert torch.allclose(clamped[0], target[0])
    assert torch.allclose(clamped[1] - pos[1], torch.tensor([0.0, 0.006, 0.008]), atol=1e-7)


def test_clamp_target_rot_offset():
    """A 30 deg target offset about x is reduced to 10 deg about the same axis; a 5 deg offset is kept."""
    tool = quat_from_rotvec(torch.tensor([[0.0, 0.0, 0.7], [0.0, 0.0, 0.7]]))
    offsets = torch.tensor([[math.radians(5.0), 0.0, 0.0], [math.radians(30.0), 0.0, 0.0]])
    target = _quat_mul(quat_from_rotvec(offsets), tool)
    clamped, held = clamp_target_rot_offset(target, tool, math.radians(10.0))
    assert held.tolist() == [False, True]
    remaining = rotvec_from_quat(_quat_mul(clamped, _quat_conj(tool)))
    assert torch.allclose(remaining[0], offsets[0], atol=1e-6)
    assert torch.allclose(remaining[1], torch.tensor([math.radians(10.0), 0.0, 0.0]), atol=1e-6)


def test_rotvec_between():
    """Rotation vector (base frame) from orientation b to orientation a: a = R(v) b."""
    b = quat_from_rotvec(torch.tensor([[0.0, 0.0, 0.7], [0.3, -0.2, 0.1]]))
    v = torch.tensor([[0.1, 0.0, 0.0], [0.0, -0.05, 0.02]])
    assert torch.allclose(rotvec_between(_quat_mul(quat_from_rotvec(v), b), b), v, atol=1e-6)


def test_integrate_target_without_change_limit():
    """`prev_step` without `max_change` means no change limit (not an error)."""
    target, step = integrate_target(torch.zeros(N, 3), torch.ones(N, 3), 0.0032, torch.zeros(N, 3))
    assert torch.allclose(step.norm(dim=-1), torch.full((N,), 0.0032))
    assert torch.allclose(target, step)


SUBSTEP = {"policy_dt": 0.032, "max_offset": 0.004, "max_rot_offset": math.radians(3.0)}


def test_substep_command_free_motion():
    """Free motion: the target is interpolated through the step, the clamp is inactive and the feedforward velocity
    is the step's velocity (the tool lags 0.5 mm, far below the bound)."""
    start = torch.zeros(N, 3)
    start_quat = quat_from_rotvec(torch.zeros(N, 3))
    step = torch.tensor([[0.0032, 0.0, 0.0, 0.0, 0.0, 0.0251], [0.0, 0.002, 0.0, 0.01, 0.0, 0.0]])
    pos = start - torch.tensor([0.0005, 0.0, 0.0])
    tool_vel = torch.full((N, 3), 0.05)  # must not be used: the target is not held
    pos_t, quat_t, vel, ang_vel = substep_command(
        start, start_quat, step, 0.5, pos, start_quat, tool_vel, tool_vel, **SUBSTEP
    )
    assert torch.allclose(pos_t, start + 0.5 * step[:, :3])
    assert torch.allclose(rotvec_between(quat_t, start_quat), 0.5 * step[:, 3:], atol=1e-6)
    assert torch.allclose(vel, step[:, :3] / 0.032)
    assert torch.allclose(ang_vel, step[:, 3:] / 0.032)


def test_substep_command_blocked():
    """Blocked tool: the target is held at the bound (4 mm / 3 deg from the tool) and moves with the tool, so the
    feedforward velocity is the tool's (the damping adds no force on top of the spring); env 1 stays free."""
    pos = torch.zeros(N, 3)
    quat = quat_from_rotvec(torch.zeros(N, 3))
    start = torch.tensor([[0.003, 0.0, 0.0], [0.0, 0.0, 0.0]])  # env 0: target already 3 mm ahead
    start_quat = quat_from_rotvec(torch.tensor([[0.0, 0.0, math.radians(2.5)], [0.0, 0.0, 0.0]]))
    step = torch.tensor([[0.0032, 0.0, 0.0, 0.0, 0.0, math.radians(1.44)], [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]])
    tool_vel = torch.tensor([[0.01, 0.0, 0.0], [0.05, 0.0, 0.0]])
    tool_ang_vel = torch.tensor([[0.0, 0.0, 0.02], [0.0, 0.0, 0.0]])
    pos_t, quat_t, vel, ang_vel = substep_command(
        start, start_quat, step, 1.0, pos, quat, tool_vel, tool_ang_vel, **SUBSTEP
    )
    assert torch.allclose(pos_t[0], torch.tensor([0.004, 0.0, 0.0]))
    assert torch.allclose(rotvec_between(quat_t, quat)[0], torch.tensor([0.0, 0.0, math.radians(3.0)]), atol=1e-6)
    assert torch.allclose(vel[0], tool_vel[0])
    assert torch.allclose(ang_vel[0], tool_ang_vel[0])
    assert torch.allclose(pos_t[1], torch.tensor([0.001, 0.0, 0.0]))  # env 1: free
    assert torch.allclose(vel[1], step[1, :3] / 0.032)
