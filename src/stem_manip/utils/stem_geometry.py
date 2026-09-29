"""Stem geometry from per-segment poses: curvature and the pose of the selected stem point.

Requirements:
- Input: segment poses as in `CableObject.data.segment_pose_w` (torch view shape `(num_envs, num_segments, 7)`,
  position + quaternion (x, y, z, w)) and the segment rest lengths.
- `joint_curvature(...)`: per-joint curvature `(num_envs, num_segments - 1)` [1/m],
  kappa_i = angle(q_i^-1 * q_{i+1}) / L_dual with L_dual = 0.5 * (L_i + L_{i+1}).
- `point_pose(...)`: pose of the point of interest, given a segment index and an offset along that segment.
- Batched over envs, no simulator imports (unit-testable without Isaac Sim).

Verify: `tests/test_stem_geometry.py` (straight rod -> 0, circular arc of radius R -> 1/R).
See docs/TODO.md -> M1.
"""
