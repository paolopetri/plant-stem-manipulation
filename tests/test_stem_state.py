"""Unit tests for the push task's stem state (`mdp/stem_state.py`), built with a fake stem model, stem asset and robot
asset of known poses (no simulator). The tests check, in a robot base that is translated and rotated against the
world:

- points along a straight and an arc stem at given arc lengths; the stem base at arc length 0, the tip at the stem
  length;
- zero curvature for a straight stem, the known curvature 2 asin(l / 2R) / l for an arc of chords;
- the distance of points (robot base frame) to the stem's centre line;
- the axial compression of every stem joint from fake joint wrenches of known axial load (+ = compression, - =
  tension), the base clamp left out.

Stems: segment poses as the stem models return them (segment centre + quaternion (x, y, z, w), local +z = tangent),
float64 so that the tolerances test the math. Two envs: env 0 with a robot base at BASE_POS rotated 90 deg about z,
env 1 with the robot base at the world origin, unrotated.
"""

import math
from types import SimpleNamespace

import torch

from stem_manip.tasks.push_position.mdp.stem_state import StemState

NUM_SEGMENTS = 20
SEGMENT_LENGTH = 0.02  # [m]
STEM_LENGTH = 0.40  # [m]
ARC_RADIUS = 0.2  # [m]
STEM_BASE_W = (0.6, 0.1, 0.05)  # [m] world frame, both envs
BASE_POS = (0.5, -0.2, 0.1)  # [m] robot base of env 0, world frame
DTYPE = torch.float64


def _base_frame_env0(point_w: tuple[float, float, float]) -> tuple[float, float, float]:
    """World point in env 0's robot base (at BASE_POS, rotated +90 deg about z): R^T (p - t), with
    R^T v = (v_y, -v_x, v_z)."""
    v = [p - t for p, t in zip(point_w, BASE_POS, strict=True)]
    return (v[1], -v[0], v[2])


def _robot() -> SimpleNamespace:
    pos = torch.tensor([BASE_POS, (0.0, 0.0, 0.0)], dtype=DTYPE)
    half = 0.25 * math.pi  # half of 90 deg about z
    quat = torch.tensor([(0.0, 0.0, math.sin(half), math.cos(half)), (0.0, 0.0, 0.0, 1.0)], dtype=DTYPE)
    return SimpleNamespace(
        data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=pos), root_quat_w=SimpleNamespace(torch=quat))
    )


def _stem_state(poses: torch.Tensor, wrenches: torch.Tensor | None = None) -> StemState:
    """Stem state over a fake model that returns `poses` (one stem, copied to both envs), the segment length and the
    joint `wrenches` (num_envs, num_segments, 6)."""
    model = SimpleNamespace(
        segment_poses=lambda stem: stem.poses,
        segment_length=lambda stem: stem.segment_length,
        joint_wrenches=lambda stem: stem.wrenches,
    )
    stem = SimpleNamespace(poses=poses.expand(2, -1, -1).clone(), segment_length=SEGMENT_LENGTH, wrenches=wrenches)
    return StemState(model, stem, _robot())


def _straight_stem() -> torch.Tensor:
    """Upright stem from STEM_BASE_W along world z."""
    poses = torch.zeros(NUM_SEGMENTS, 7, dtype=DTYPE)
    poses[:, :3] = torch.tensor(STEM_BASE_W, dtype=DTYPE)
    poses[:, 2] += (torch.arange(NUM_SEGMENTS, dtype=DTYPE) + 0.5) * SEGMENT_LENGTH
    poses[:, 6] = 1.0
    return poses


def _arc_node(k: int) -> tuple[float, float, float]:
    """Node k (end of segment k - 1) of the arc stem, world frame: chords of length l on a circle of radius R in the
    x-z plane through STEM_BASE_W, starting along +z."""
    phi = 2.0 * math.asin(0.5 * SEGMENT_LENGTH / ARC_RADIUS)
    x, y, z = STEM_BASE_W
    return (x + ARC_RADIUS * (1.0 - math.cos(k * phi)), y, z + ARC_RADIUS * math.sin(k * phi))


def _arc_stem() -> torch.Tensor:
    phi = 2.0 * math.asin(0.5 * SEGMENT_LENGTH / ARC_RADIUS)
    nodes = torch.tensor([_arc_node(k) for k in range(NUM_SEGMENTS + 1)], dtype=DTYPE)
    tilt = (torch.arange(NUM_SEGMENTS, dtype=DTYPE) + 0.5) * phi  # rotation about +y taking +z to the chord
    poses = torch.zeros(NUM_SEGMENTS, 7, dtype=DTYPE)
    poses[:, :3] = 0.5 * (nodes[:-1] + nodes[1:])
    poses[:, 4] = torch.sin(0.5 * tilt)
    poses[:, 6] = torch.cos(0.5 * tilt)
    return poses


def _expected(points_w: list[tuple[float, float, float]]) -> torch.Tensor:
    """(2, k, 3): the world points in env 0's robot base and in env 1's (= world)."""
    return torch.tensor([[_base_frame_env0(p) for p in points_w], points_w], dtype=DTYPE)


def test_points_along_a_straight_stem():
    x, y, z = STEM_BASE_W
    points = _stem_state(_straight_stem()).points_b((0.0, 0.13, STEM_LENGTH))
    expected = _expected([(x, y, z), (x, y, z + 0.13), (x, y, z + STEM_LENGTH)])  # base, inside a segment, tip
    torch.testing.assert_close(points, expected, rtol=0.0, atol=1e-12)


def test_points_along_an_arc_stem():
    """Base, a node in the middle and the tip lie on the arc's nodes (segment ends lie on the circle)."""
    points = _stem_state(_arc_stem()).points_b((0.0, 0.2, STEM_LENGTH))
    torch.testing.assert_close(points, _expected([_arc_node(0), _arc_node(10), _arc_node(20)]), rtol=0.0, atol=1e-12)


def test_straight_stem_has_zero_curvature():
    curvature = _stem_state(_straight_stem()).curvature()
    torch.testing.assert_close(curvature, torch.zeros(2, NUM_SEGMENTS - 1, dtype=DTYPE), rtol=0.0, atol=1e-9)


def test_arc_stem_has_the_arc_curvature():
    """Neighbouring chords meet at the angle 2 asin(l / 2R), over the dual length l."""
    expected = 2.0 * math.asin(0.5 * SEGMENT_LENGTH / ARC_RADIUS) / SEGMENT_LENGTH
    curvature = _stem_state(_arc_stem()).curvature()
    torch.testing.assert_close(curvature, torch.full((2, NUM_SEGMENTS - 1), expected, dtype=DTYPE), rtol=1e-9, atol=0.0)


def test_distance_of_points_to_the_centre_line():
    """Beside the straight stem: the horizontal distance; above the tip and below the base: the distance to the end."""
    x, y, z = STEM_BASE_W
    points_w = [(x + 0.03, y, z + 0.2), (x, y - 0.05, z + 0.01), (x, y, z + STEM_LENGTH + 0.07), (x, y, z - 0.02)]
    distances = _stem_state(_straight_stem()).distance_b(_expected(points_w))
    torch.testing.assert_close(
        distances, torch.tensor([[0.03, 0.05, 0.07, 0.02]] * 2, dtype=DTYPE), rtol=0.0, atol=1e-12
    )


def test_axial_compression_of_every_joint():
    """The chain model's joint wrenches: force (x, y, z) and moment (x, y, z) in the joint frame, x along the stem
    axis, x force > 0 = compression; entry 0 is the base clamp (no stem joint). Env 0 is pressed down (compression
    growing towards the base), env 1 is pulled at the tip (tension 0.9 N). Shear forces and moments are set too and
    must not enter."""
    compression = torch.linspace(0.5, 2.4, NUM_SEGMENTS - 1, dtype=DTYPE)  # [N] joints 1 ... 19
    wrenches = torch.zeros(2, NUM_SEGMENTS, 6, dtype=DTYPE)
    wrenches[:, 0] = torch.tensor([7.0, 8.0, 9.0, 0.1, 0.2, 0.3], dtype=DTYPE)  # base clamp: left out
    wrenches[0, 1:, 0] = compression
    wrenches[1, 1:, 0] = -0.9
    wrenches[:, 1:, 1:3] = 0.3  # shear [N]
    wrenches[:, 1:, 3:] = 0.02  # twist and bending moments [N m]
    result = _stem_state(_straight_stem(), wrenches).axial_compression()
    expected = torch.stack([compression, torch.full((NUM_SEGMENTS - 1,), -0.9, dtype=DTYPE)])
    torch.testing.assert_close(result, expected, rtol=0.0, atol=1e-12)
