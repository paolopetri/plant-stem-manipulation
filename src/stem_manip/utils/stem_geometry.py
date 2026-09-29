"""Stem geometry from per-segment poses: curvature and the pose of the selected stem point.

Requirements:
- Input: segment poses as in `CableObject.data.segment_pose_w` (torch view shape `(num_envs, num_segments, 7)`,
  position + quaternion (x, y, z, w)) and the segment rest lengths.
- Per joint, split the relative rotation q_rel = q_i^-1 * q_{i+1} into bending (axis perpendicular to the
  stem tangent) and twist (axis = tangent) via a swing-twist decomposition; L_dual = 0.5 * (L_i + L_{i+1}).
  - `joint_curvature(...)`: bending curvature `(num_envs, num_segments - 1)` [1/m] = bend_angle / L_dual.
  - `joint_twist_rate(...)`: twist rate `(num_envs, num_segments - 1)` [rad/m] = twist_angle / L_dual.
  - `joint_axial_strain(...)`: `(num_envs, num_segments - 1)` [-] = centre distance / L_dual - 1.
  (The total angle of q_rel mixes bending and twist, so it must not be used as curvature.)
- `point_pose(...)`: pose of the point of interest, given a segment index and an offset along that segment.
- Batched over envs, no simulator imports (unit-testable without Isaac Sim).

Verify: `tests/test_stem_geometry.py` (straight rod -> 0, circular arc of radius R -> 1/R).
See docs/TODO.md -> M1.
"""
