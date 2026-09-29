"""Sanity check of stem model v1 alone on Newton (no robot).

Requirements:
- Spawn `stem_cfg()` in a few envs on a ground plane, headless option, run a fixed number of steps.
- Report and check: base stays fixed; tip sags under gravity by a plausible amount; after displacing the tip
  (write segment state or apply a push) the stem springs back; curvature stays finite.
- Quantitative: stem clamped horizontally, compare the tip sag under self-weight with the analytic cantilever
  deflection delta = q L^4 / (8 E I), q = rho A g, I = pi d^4 / 64 (report the relative error).
- Print a short pass/fail summary.

Verify: runs without errors; all checks pass. See docs/TODO.md -> M1.
"""
