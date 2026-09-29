"""Termination terms.

Requirements:
- Damage limits exceeded (no-damage constraint, limits from `damage` in `assets/stem/stem.yaml`):
  bending curvature first; twist rate and axial strain once thresholds are known.
- Time out; stem or end-effector out of bounds.
- Each term must be shown to fire in a scripted test case.

See docs/TODO.md -> M4.
"""
