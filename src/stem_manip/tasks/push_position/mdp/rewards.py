"""Reward terms.

Requirements:
- Distance of the stem point to the target (e.g. exp(-d/std) and/or -d).
- Damage penalties below the hard limits, using `stem_manip.utils.stem_geometry`: bending curvature first;
  twist rate and axial strain (tearing when the fork drags along the stem) once thresholds are known.
- Action rate / magnitude penalties. All weights in the env cfg.

See docs/TODO.md -> M4.
"""
