"""Stem geometry from per-segment poses: bending, twist and stretch per joint, pose of the selected stem point,
points at given arc lengths.

Input convention (as `CableObject.data.segment_pose_w` of the Newton cable):
- `segment_poses`: shape `(num_envs, num_segments, 7)`, position [m] + quaternion (x, y, z, w), world frame.
  The segment frame has its origin at the segment centre and its local +Z along the stem tangent.
- `segment_lengths`: shape `(num_segments,)`, rest length of each segment [m].

The segments are rigid, so all deformation sits in the `num_segments - 1` joints between neighbours. Joint i
connects segments i and i + 1 and represents the stem length L_dual = 0.5 * (L_i + L_(i+1)).

Batched over envs, no simulator needed. Tests: `tests/test_stem_geometry.py`.
"""

from collections.abc import Sequence

import torch

from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul


def _dual_lengths(segment_lengths: torch.Tensor) -> torch.Tensor:
    """Stem length represented by each joint, shape `(num_segments - 1,)` [m]."""
    return 0.5 * (segment_lengths[:-1] + segment_lengths[1:])


def _tangents(quaternions: torch.Tensor) -> torch.Tensor:
    """Stem tangent (local +Z of the segment frame) in the world frame, shape `(..., 3)`."""
    z_axis = torch.zeros_like(quaternions[..., :3])
    z_axis[..., 2] = 1.0
    return quat_apply(quaternions, z_axis)


def _bend_twist_angles(segment_poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Bend and twist angle of each joint [rad], each of shape `(num_envs, num_segments - 1)`.

    Swing-twist decomposition of the relative rotation q_rel = q_i^-1 * q_(i+1) about the tangent (local +Z):
    the z and w components hold the rotation about the tangent (twist), x and y the rotation about the two
    axes perpendicular to it (bending). The bend angle is in [0, pi], the twist angle is signed, in [-pi, pi].
    """
    quaternions = segment_poses[..., 3:]
    q_rel = quat_mul(quat_conjugate(quaternions[:, :-1]), quaternions[:, 1:])
    x, y, z, w = q_rel.unbind(dim=-1)
    bend_angle = 2.0 * torch.atan2(torch.sqrt(x * x + y * y), torch.sqrt(z * z + w * w))
    # q and -q are the same rotation: flip to w >= 0 so that the twist angle is the short way round
    twist_angle = 2.0 * torch.atan2(torch.where(w < 0.0, -z, z), w.abs())
    return bend_angle, twist_angle


def joint_curvature(segment_poses: torch.Tensor, segment_lengths: torch.Tensor) -> torch.Tensor:
    """Bending curvature of each joint, shape `(num_envs, num_segments - 1)` [1/m] (= 1 / bending radius)."""
    bend_angle, _ = _bend_twist_angles(segment_poses)
    return bend_angle / _dual_lengths(segment_lengths)


def joint_twist_rate(segment_poses: torch.Tensor, segment_lengths: torch.Tensor) -> torch.Tensor:
    """Signed twist rate of each joint, shape `(num_envs, num_segments - 1)` [rad/m]."""
    _, twist_angle = _bend_twist_angles(segment_poses)
    return twist_angle / _dual_lengths(segment_lengths)


def joint_axial_strain(segment_poses: torch.Tensor, segment_lengths: torch.Tensor) -> torch.Tensor:
    """Axial strain of each joint, shape `(num_envs, num_segments - 1)` [-], positive in tension.

    Gap between the end of segment i and the start of segment i + 1, measured along the tangent of segment i
    and divided by L_dual. Pure bending leaves the two points coincident and gives zero strain (the distance
    between segment centres would cut the corner and report a false compression).
    """
    positions = segment_poses[..., :3]
    half_segments = 0.5 * segment_lengths.unsqueeze(-1) * _tangents(segment_poses[..., 3:])
    gap = (positions - half_segments)[:, 1:] - (positions + half_segments)[:, :-1]
    tangents = _tangents(segment_poses[:, :-1, 3:])
    return (gap * tangents).sum(dim=-1) / _dual_lengths(segment_lengths)


def point_pose(segment_poses: torch.Tensor, segment_index: int, offset: float) -> torch.Tensor:
    """Pose of a point on the stem, shape `(num_envs, 7)`: position [m] + quaternion (x, y, z, w).

    Args:
        segment_poses: Segment poses, shape `(num_envs, num_segments, 7)`.
        segment_index: Segment that carries the point.
        offset: Distance from the segment centre along the tangent [m], in [-L/2, L/2].

    Returns:
        The point's position and the orientation of its segment.
    """
    pose = segment_poses[:, segment_index]
    position = pose[:, :3] + offset * _tangents(pose[:, 3:])
    return torch.cat([position, pose[:, 3:]], dim=-1)


def stem_points(segment_poses: torch.Tensor, arc_lengths: Sequence[float], segment_length: float) -> torch.Tensor:
    """Points on the stem's centre line at the given arc lengths, shape `(num_envs, len(arc_lengths), 3)` [m].

    Args:
        segment_poses: Segment poses, shape `(num_envs, num_segments, 7)`.
        arc_lengths: Distances from the base along the stem [m], in [0, num_segments * segment_length]
            (0 = base, the stem length = tip); outside that range a ValueError is raised.
        segment_length: Rest length of each segment [m] (all segments equally long).

    Returns:
        The positions, in the frame of `segment_poses`.
    """
    num_segments = segment_poses.shape[1]
    points = []
    for s in arc_lengths:
        if not 0.0 <= s <= num_segments * segment_length + 1e-9:
            raise ValueError(f"Arc length {s} m outside the stem [0, {num_segments * segment_length}] m.")
        index = min(int(s // segment_length), num_segments - 1)
        points.append(point_pose(segment_poses, index, s - (index + 0.5) * segment_length)[:, :3])
    return torch.stack(points, dim=1)
