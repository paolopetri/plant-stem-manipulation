"""Unit tests of the tool-tip kinematics (pose and geometric Jacobian in the robot base frame), worked examples."""

import math

import torch

from stem_manip.utils.tool_tip import tool_tip_jacobian, tool_tip_pose

S = math.sqrt(0.5)
IDENTITY = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
YAW_90 = torch.tensor([[0.0, 0.0, S, S]])  # 90 deg about z
YAW_MINUS_90 = torch.tensor([[0.0, 0.0, -S, S]])


def test_pose_in_rotated_shifted_base():
    """Base at (1, 0, 0) yawed 90 deg; body at world (1, 1, 0), unrotated; tip 0.1 m along the body x axis.

    Body in the base frame: (1, 0, 0), yawed -90 deg; its x axis is the base's -y, so the tip is at (1, -0.1, 0).
    """
    pos, quat = tool_tip_pose(
        torch.tensor([[1.0, 0.0, 0.0]]),
        YAW_90,
        torch.tensor([[1.0, 1.0, 0.0]]),
        IDENTITY,
        torch.tensor([[0.1, 0.0, 0.0]]),
        IDENTITY,
    )
    assert torch.allclose(pos, torch.tensor([[1.0, -0.1, 0.0]]), atol=1e-6)
    assert torch.allclose(quat, YAW_MINUS_90, atol=1e-6)


def test_pose_applies_offset_rotation():
    """Base and body at the world origin, unrotated; tip offset yawed 90 deg: the tip orientation is the offset's."""
    pos, quat = tool_tip_pose(
        torch.zeros(1, 3), IDENTITY, torch.zeros(1, 3), IDENTITY, torch.tensor([[0.0, 0.0, 0.2]]), YAW_90
    )
    assert torch.allclose(pos, torch.tensor([[0.0, 0.0, 0.2]]), atol=1e-6)
    assert torch.allclose(quat, YAW_90, atol=1e-6)


def _one_joint_jacobian() -> torch.Tensor:
    """One revolute joint about the world z axis through the origin, body origin at world (1, 0, 0): body
    Jacobian column v = z x (1, 0, 0) = (0, 1, 0), w = (0, 0, 1). Shape (1, 6, 1)."""
    return torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).view(1, 6, 1)


def test_jacobian_shifts_linear_rows_to_the_tip():
    """Tip 0.5 m along the body x axis (world (1.5, 0, 0)): v_tip = z x (1.5, 0, 0) = (0, 1.5, 0); w unchanged."""
    jacobian = tool_tip_jacobian(_one_joint_jacobian(), IDENTITY, IDENTITY, torch.tensor([[0.5, 0.0, 0.0]]))
    assert torch.allclose(jacobian, torch.tensor([0.0, 1.5, 0.0, 0.0, 0.0, 1.0]).view(1, 6, 1), atol=1e-6)


def test_jacobian_in_rotated_base():
    """Same arm, base yawed 90 deg: world (0, 1.5, 0) is base (1.5, 0, 0); the z axis is the same in both frames."""
    jacobian = tool_tip_jacobian(_one_joint_jacobian(), YAW_90, IDENTITY, torch.tensor([[0.5, 0.0, 0.0]]))
    assert torch.allclose(jacobian, torch.tensor([1.5, 0.0, 0.0, 0.0, 0.0, 1.0]).view(1, 6, 1), atol=1e-6)


def test_jacobian_uses_body_orientation_for_the_lever():
    """Body yawed 90 deg in the world: the offset 0.5 m along its x axis points to world +y, tip at world (1, 0.5,
    0); v_tip = z x (1, 0.5, 0) = (-0.5, 1, 0)."""
    jacobian = tool_tip_jacobian(_one_joint_jacobian(), IDENTITY, YAW_90, torch.tensor([[0.5, 0.0, 0.0]]))
    assert torch.allclose(jacobian, torch.tensor([-0.5, 1.0, 0.0, 0.0, 0.0, 1.0]).view(1, 6, 1), atol=1e-6)
