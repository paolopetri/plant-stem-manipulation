"""Unit tests of the independent FR3 slot-base kinematics used by the checks; no simulator needed."""

import math

import pytest
import torch

from stem_manip.utils import fr3_reference
from stem_manip.utils.fr3_reference import flange_pose, rotvec_from_matrix, rpy_matrix, slot_base_pose

DTYPE = torch.float64
# start pose of the push task (env_cfg.START_JOINT_POS) and its slot base (0.40, 0, 0.50) m (user, 2026-10-09)
START_Q = (-0.013, -0.689, 0.011, -2.496, 0.007, 1.808, 0.779)
START_TIP = (0.40, 0.0, 0.50)


def _rotz(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=DTYPE)


def test_flange_at_zero_joints():
    """All joints 0: the flange is 0.088 m in front of the base axis at 0.333 + 0.316 + 0.384 - 0.107 = 0.926 m,
    pointing down (FR3 datasheet kinematics)."""
    pose = flange_pose(torch.zeros(1, 7, dtype=DTYPE))[0]
    assert torch.allclose(pose[:3, 3], torch.tensor([0.088, 0.0, 0.926], dtype=DTYPE), atol=1e-12)
    assert torch.allclose(pose[:3, :3], torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=DTYPE)), atol=1e-12)
    assert torch.allclose(pose[3], torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=DTYPE))


def test_joint1_turns_the_arm_about_the_base_axis_and_joint7_turns_the_flange():
    q = torch.zeros(2, 7, dtype=DTYPE)
    q[0, 0] = 0.7
    q[1, 6] = 0.7
    zero = flange_pose(torch.zeros(1, 7, dtype=DTYPE))[0]
    turned = flange_pose(q)
    assert torch.allclose(turned[0, :3, :3], _rotz(0.7) @ zero[:3, :3], atol=1e-12)
    assert torch.allclose(turned[0, :3, 3], _rotz(0.7) @ zero[:3, 3], atol=1e-12)
    assert torch.allclose(turned[1, :3, :3], zero[:3, :3] @ _rotz(0.7), atol=1e-12)
    assert torch.allclose(turned[1, :3, 3], zero[:3, 3], atol=1e-12)


def test_slot_base_at_the_start_pose():
    """The joint angles were solved for the slot base at (0.40, 0, 0.50) m and rounded to 1 mrad: with levers up to
    about 0.6 m that leaves up to a few mm (5 mm, as `scripts/check_push_env.py` START_TOL)."""
    pose = slot_base_pose(torch.tensor([START_Q], dtype=DTYPE), "fork_v2")[0]
    assert float((pose[:3, 3] - torch.tensor(START_TIP, dtype=DTYPE)).norm()) < 5e-3


def test_slot_base_is_the_yaml_frame_in_the_flange():
    """fork_v2: slot base at (0.039952, -0.039952, 0.0125) m in fr3_link8, turned -45 deg about the flange axis."""
    q = torch.tensor([START_Q], dtype=DTYPE)
    flange, tip = flange_pose(q)[0], slot_base_pose(q, "fork_v2")[0]
    in_flange = torch.linalg.inv(flange) @ tip
    assert torch.allclose(in_flange[:3, 3], torch.tensor([0.039952, -0.039952, 0.0125], dtype=DTYPE), atol=1e-12)
    assert torch.allclose(in_flange[:3, :3], _rotz(-0.785398), atol=1e-12)


def test_rpy_matrix_is_fixed_axis_roll_pitch_yaw():
    """URDF rpy: R = Rz(yaw) Ry(pitch) Rx(roll). With roll = pitch = 90 deg: x -> -z, y -> x."""
    rot = rpy_matrix((math.pi / 2, math.pi / 2, 0.0))
    assert torch.allclose(rot[:, 0], torch.tensor([0.0, 0.0, -1.0], dtype=DTYPE), atol=1e-12)
    assert torch.allclose(rot[:, 1], torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE), atol=1e-12)
    assert torch.allclose(rpy_matrix((0.0, 0.0, -0.785398)), _rotz(-0.785398), atol=1e-12)


@pytest.mark.parametrize("angle", [0.0, 1e-7, 1e-3, 0.05, 1.0, 3.0])
def test_rotvec_from_matrix_inverts_rodrigues(angle: float):
    axis = torch.tensor([1.0, -2.0, 0.5], dtype=DTYPE)
    axis = axis / axis.norm()
    k = torch.tensor([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]], dtype=DTYPE)
    rot = torch.eye(3, dtype=DTYPE) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k
    rotvec = rotvec_from_matrix(rot.unsqueeze(0))[0]
    assert torch.allclose(rotvec, angle * axis, atol=1e-12)


def test_slot_base_in_flange_rejects_a_non_identity_mount(tmp_path, monkeypatch):
    """The yaml gives slot_base in the fr3_link8 frame, the URDF builder puts it under the end-effector body: the two
    agree only with an identity mount, so any other mount must fail loudly."""
    ee_dir = tmp_path / "fr3" / "end_effectors" / "tilted"
    ee_dir.mkdir(parents=True)
    (ee_dir / "tilted.yaml").write_text(
        "mount: {xyz: [0, 0, 0.01], rpy: [0, 0, 0]}\nslot_base: {xyz: [0.04, 0, 0.01], rpy: [0, 0, 0]}\n"
    )
    monkeypatch.setattr(fr3_reference, "REPO_ASSETS_DIR", tmp_path)
    with pytest.raises(ValueError, match="mount"):
        fr3_reference.slot_base_in_flange("tilted")
