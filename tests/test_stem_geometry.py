"""Unit tests for `stem_manip.utils.stem_geometry` (no simulator needed).

Requirements:
- Straight rod -> zero curvature at every joint.
- Segments sampled on a circular arc of radius R -> curvature 1/R (within discretization tolerance).
- Straight rod with a pure twist -> zero curvature, twist rate = twist angle / L_dual.
- Uniformly stretched rod -> the imposed axial strain, zero curvature and twist.
- `point_pose` at segment index / offset matches the analytic point on the rod.

See docs/TODO.md -> M1.
"""

import pytest

pytest.skip("stem_geometry is implemented in M1", allow_module_level=True)
