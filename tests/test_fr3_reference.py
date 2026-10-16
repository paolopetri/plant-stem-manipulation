"""Unit tests of the independent FR3 slot-base kinematics used by the checks; no simulator needed."""

import math

import pytest
import torch

from stem_manip.utils import fr3_reference
from stem_manip.utils.fr3_reference import flange_pose, rotvec_from_matrix, rpy_matrix, slot_base_pose

DTYPE = torch.float64
# held-out start pose of the push task (env_cfg.EVAL_START_POSES["a_above_tip"]) and its slot base (0.40, 0, 0.50) m
# (user, 2026-10-09)
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
    flange, slot_base = flange_pose(q)[0], slot_base_pose(q, "fork_v2")[0]
    in_flange = torch.linalg.inv(flange) @ slot_base
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


# -- start-pose table of the push task (user, 2026-10-15, #160 / #163), checked with the independent FK above
STEM_REFERENCE = (0.575, 0.0)  # [m] stem axis the table is solved for: the spawn-area centre
SLOT_BASE_DISTANCE = 0.205  # [m] stem axis to slot base, horizontal
TABLE_HEIGHTS = (0.10, 0.45)  # [m] slot-base height
TABLE_AZIMUTH = math.radians(60.0)  # [rad] |azimuth| about the stem axis, 0 = towards the robot
MIN_TABLE_MARGIN = 0.5  # [rad] joint margin of every table pose
MIN_EVAL_MARGIN = 0.25  # [rad] held-out poses: above the joint-margin termination
# FR3 joint limits [rad] (franka_description fr3.urdf)
FR3_LIMITS = (
    (-2.9007, 2.9007),
    (-1.8361, 1.8361),
    (-2.9007, 2.9007),
    (-3.077, -0.1169),
    (-2.8763, 2.8763),
    (0.4398, 4.6216),
    (-3.0508, 3.0508),
)
POSE_POS_TOL = 5e-3  # [m] joint angles rounded to 1 mrad (as test_slot_base_at_the_start_pose)
POSE_ANGLE_TOL = math.radians(1.0)
JOINT_NOISE = 0.02  # [rad] reset noise; a held-out pose differs from every table pose by more than twice it


def _start_poses():
    from stem_manip.tasks.push_position.env_cfg import EVAL_START_POSES, START_POSES

    held_out = {k: torch.tensor([v], dtype=DTYPE) for k, v in EVAL_START_POSES.items()}
    return torch.tensor(START_POSES, dtype=DTYPE), held_out


def _cylindrical(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slot base of q (n, 7) about the reference stem axis: horizontal distance, azimuth (0 = towards the robot,
    counterclockwise seen from above), height, and the slot base's rotation (n, 3, 3)."""
    pose = slot_base_pose(q, "fork_v2")
    rel = pose[:, :3, 3] - torch.tensor((*STEM_REFERENCE, 0.0), dtype=DTYPE)
    azimuth = torch.atan2(-rel[:, 1], -rel[:, 0])
    return rel[:, :2].norm(dim=-1), azimuth, pose[:, 2, 3], pose[:, :3, :3]


def _margin(q: torch.Tensor) -> torch.Tensor:
    limits = torch.tensor(FR3_LIMITS, dtype=DTYPE)
    return torch.minimum(q - limits[:, 0], limits[:, 1] - q).amin(dim=-1)


def _facing_errors(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fork horizontal and slot facing the stem: angle of the slot base's z from straight down, and angle of its x
    (along the fork, out) from the horizontal direction to the stem axis [rad]."""
    pose = slot_base_pose(q, "fork_v2")
    x, z = pose[:, :3, 0], pose[:, :3, 2]
    tilt = torch.acos((-z[:, 2]).clamp(-1.0, 1.0))
    to_stem = torch.tensor((*STEM_REFERENCE, 0.0), dtype=DTYPE) - pose[:, :3, 3]
    to_stem[:, 2] = 0.0
    aim = torch.acos(((x * to_stem).sum(-1) / (x.norm(dim=-1) * to_stem.norm(dim=-1))).clamp(-1.0, 1.0))
    return tilt, aim


def test_start_pose_table_slot_base_beside_the_stem():
    table, _ = _start_poses()
    distance, azimuth, height, _ = _cylindrical(table)
    assert table.shape[0] > 0 and table.shape[1] == 7
    assert ((distance - SLOT_BASE_DISTANCE).abs() < POSE_POS_TOL).all()
    assert ((height > TABLE_HEIGHTS[0] - POSE_POS_TOL) & (height < TABLE_HEIGHTS[1] + POSE_POS_TOL)).all()
    assert (azimuth.abs() < TABLE_AZIMUTH + POSE_ANGLE_TOL).all()
    assert (azimuth > 0).any() and (azimuth < 0).any()  # both sides of the stem


def test_start_pose_table_joint_margin():
    table, _ = _start_poses()
    assert (_margin(table) >= MIN_TABLE_MARGIN).all()


def test_start_pose_table_fork_horizontal_slot_facing_the_stem():
    table, _ = _start_poses()
    tilt, aim = _facing_errors(table)
    assert (tilt < POSE_ANGLE_TOL).all()
    assert (aim < POSE_ANGLE_TOL).all()


def test_held_out_start_poses():
    """(a) the old start pose, slot base at (0.40, 0, 0.50) m; (b) beside the stem at 90 deg azimuth; (c) low, slot
    base at 0.08 m. (b) and (c) beside the stem like the table, slot facing it."""
    _, held_out = _start_poses()
    assert set(held_out) == {"a_above_tip", "b_side_90", "c_low"}
    assert held_out["a_above_tip"].tolist() == [list(START_Q)]
    a = slot_base_pose(held_out["a_above_tip"], "fork_v2")[0]
    assert float((a[:3, 3] - torch.tensor(START_TIP, dtype=DTYPE)).norm()) < POSE_POS_TOL
    distance, azimuth, _, _ = _cylindrical(held_out["b_side_90"])
    assert abs(float(distance[0]) - SLOT_BASE_DISTANCE) < POSE_POS_TOL
    assert abs(abs(float(azimuth[0])) - math.pi / 2) < POSE_ANGLE_TOL
    distance, _, height, _ = _cylindrical(held_out["c_low"])
    assert abs(float(distance[0]) - SLOT_BASE_DISTANCE) < POSE_POS_TOL
    assert abs(float(height[0]) - 0.08) < POSE_POS_TOL
    for name in ("b_side_90", "c_low"):
        tilt, aim = _facing_errors(held_out[name])
        assert float(tilt[0]) < POSE_ANGLE_TOL and float(aim[0]) < POSE_ANGLE_TOL, name
    for name, q in held_out.items():
        assert float(_margin(q)[0]) >= MIN_EVAL_MARGIN, name


def test_held_out_start_poses_are_not_in_the_table():
    table, held_out = _start_poses()
    for name, q in held_out.items():
        assert float((table - q).abs().amax(dim=-1).min()) > 2 * JOINT_NOISE, name
