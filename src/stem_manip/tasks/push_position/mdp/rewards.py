"""Reward terms.

Requirements:
- Distance of the stem point to the target (e.g. exp(-d/std) and/or -d).
- Curvature penalty below the hard limit, using `stem_manip.utils.stem_geometry.joint_curvature`.
- Action rate / magnitude penalties. All weights in the env cfg.

See docs/TODO.md -> M4.
"""
