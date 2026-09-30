"""Unit tests for `stem_manip.utils.stem_geometry` (no simulator needed).

Requirements:
- Straight rod -> zero curvature at every joint.
- Segments sampled on a circular arc of radius R -> curvature 1/R (within discretization tolerance).
- Straight rod with a pure twist -> zero curvature, twist rate = twist angle / L_dual.
- Uniformly stretched rod -> the imposed axial strain, zero curvature and twist.
- Bent but unstretched rod (circular arc) -> zero axial strain.
- `point_pose` at segment index / offset matches the analytic point on the rod.

Test rods: segment frames as in the Newton cable (origin at the segment centre, local +Z = tangent),
poses as position + quaternion (x, y, z, w). Two envs, the second one shifted, to cover batching.
Rods are built in float64 so that the tolerances test the math, not float32 round-off (the simulator's
float32 positions limit the strain resolution; see docs/TODO.md).
"""

import math

import torch

from stem_manip.utils import stem_geometry

NUM_SEGMENTS = 20
SEGMENT_LENGTH = 0.02  # [m]
ARC_RADIUS = 0.2  # [m]
DTYPE = torch.float64
ENV_SHIFT = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)  # offset of the second env [m]


def _batch(positions: torch.Tensor, quaternions: torch.Tensor) -> torch.Tensor:
    """Stack one rod into poses of shape (2, num_segments, 7); the second env is shifted by ENV_SHIFT."""
    poses = torch.cat([positions, quaternions], dim=-1)
    shifted = poses.clone()
    shifted[:, :3] += ENV_SHIFT
    return torch.stack([poses, shifted])


def _segment_lengths() -> torch.Tensor:
    return torch.full((NUM_SEGMENTS,), SEGMENT_LENGTH, dtype=DTYPE)


def _straight_rod(stretch: float = 0.0, twist_per_joint: float = 0.0) -> torch.Tensor:
    """Rod along world z. `stretch` scales the centre spacing, `twist_per_joint` [rad] rotates about z."""
    i = torch.arange(NUM_SEGMENTS, dtype=DTYPE)
    positions = torch.zeros(NUM_SEGMENTS, 3, dtype=DTYPE)
    positions[:, 2] = (i + 0.5) * SEGMENT_LENGTH * (1.0 + stretch)
    quaternions = torch.zeros(NUM_SEGMENTS, 4, dtype=DTYPE)
    quaternions[:, 2] = torch.sin(0.5 * i * twist_per_joint)
    quaternions[:, 3] = torch.cos(0.5 * i * twist_per_joint)
    return _batch(positions, quaternions)


def _arc_nodes() -> tuple[torch.Tensor, float]:
    """Nodes (segment end points) on a circle of radius ARC_RADIUS in the x-z plane, starting along +z.

    Returns the nodes, shape (NUM_SEGMENTS + 1, 3), and the angle between neighbouring segments [rad].
    """
    phi = 2.0 * math.asin(0.5 * SEGMENT_LENGTH / ARC_RADIUS)  # each segment is a chord of length SEGMENT_LENGTH
    k = torch.arange(NUM_SEGMENTS + 1, dtype=DTYPE)
    nodes = torch.zeros(NUM_SEGMENTS + 1, 3, dtype=DTYPE)
    nodes[:, 0] = ARC_RADIUS * (1.0 - torch.cos(k * phi))
    nodes[:, 2] = ARC_RADIUS * torch.sin(k * phi)
    return nodes, phi


def _arc_rod() -> torch.Tensor:
    """Unstretched rod bent into a circular arc: segment i is the chord between nodes i and i + 1."""
    nodes, phi = _arc_nodes()
    positions = 0.5 * (nodes[:-1] + nodes[1:])
    # rotation about +y that takes +z to the chord direction
    tilt = (torch.arange(NUM_SEGMENTS, dtype=DTYPE) + 0.5) * phi
    quaternions = torch.zeros(NUM_SEGMENTS, 4, dtype=DTYPE)
    quaternions[:, 1] = torch.sin(0.5 * tilt)
    quaternions[:, 3] = torch.cos(0.5 * tilt)
    return _batch(positions, quaternions)


def _assert_all_close(actual: torch.Tensor, expected: float, rtol: float = 0.0, atol: float = 0.0) -> None:
    assert actual.shape == (2, NUM_SEGMENTS - 1)
    torch.testing.assert_close(actual, torch.full_like(actual, expected), rtol=rtol, atol=atol)


def test_straight_rod_has_zero_curvature():
    curvature = stem_geometry.joint_curvature(_straight_rod(), _segment_lengths())
    _assert_all_close(curvature, 0.0, atol=1e-9)


def test_arc_curvature_is_inverse_radius():
    curvature = stem_geometry.joint_curvature(_arc_rod(), _segment_lengths())
    # chords overestimate 1/R by a factor of about 1 + phi^2 / 24 = 1.0004
    _assert_all_close(curvature, 1.0 / ARC_RADIUS, rtol=1e-3)


def test_pure_twist_gives_twist_rate_and_no_curvature():
    twist_per_joint = 0.05  # [rad]
    poses = _straight_rod(twist_per_joint=twist_per_joint)
    _assert_all_close(stem_geometry.joint_curvature(poses, _segment_lengths()), 0.0, atol=1e-9)
    _assert_all_close(
        stem_geometry.joint_twist_rate(poses, _segment_lengths()), twist_per_joint / SEGMENT_LENGTH, rtol=1e-9
    )


def test_stretched_rod_gives_axial_strain_only():
    strain = 0.01
    poses = _straight_rod(stretch=strain)
    _assert_all_close(stem_geometry.joint_axial_strain(poses, _segment_lengths()), strain, atol=1e-9)
    _assert_all_close(stem_geometry.joint_curvature(poses, _segment_lengths()), 0.0, atol=1e-9)
    _assert_all_close(stem_geometry.joint_twist_rate(poses, _segment_lengths()), 0.0, atol=1e-9)


def test_bent_rod_has_zero_axial_strain():
    strain = stem_geometry.joint_axial_strain(_arc_rod(), _segment_lengths())
    # the centre-distance formula would give about -phi^2 / 8 = -1.25e-3 here
    _assert_all_close(strain, 0.0, atol=1e-9)


def test_point_pose_matches_analytic_point():
    poses = _arc_rod()
    nodes, _ = _arc_nodes()
    segment_index = 5

    centre = stem_geometry.point_pose(poses, segment_index, offset=0.0)
    assert centre.shape == (2, 7)
    torch.testing.assert_close(centre, poses[:, segment_index])

    end = stem_geometry.point_pose(poses, segment_index, offset=0.5 * SEGMENT_LENGTH)
    expected_position = torch.stack([nodes[segment_index + 1], nodes[segment_index + 1] + ENV_SHIFT])
    torch.testing.assert_close(end[:, :3], expected_position, rtol=0.0, atol=1e-9)
    torch.testing.assert_close(end[:, 3:], poses[:, segment_index, 3:])
