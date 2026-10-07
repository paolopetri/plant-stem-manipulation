"""Unit tests of the Cartesian impedance law and the target integration (no simulator)."""

import math

import torch

from stem_manip.utils.impedance import (
    apparent_mass_damping,
    cartesian_impedance_torque,
    integrate_orientation,
    integrate_target,
    quat_from_rotvec,
    limit_torque_rate,
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
