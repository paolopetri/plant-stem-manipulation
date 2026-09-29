"""Sanity check of the coupled scene: FR3 + fork pushing the stem (Newton proxy coupling).

Requirements:
- Scene as in the stage-1 env (robot in MuJoCo-Warp, stem in VBD, fork as proxy collider).
- Scripted motion: the tool_tip approaches the stem, pushes it sideways, retreats.
- Check: the stem deflects on contact and springs back after release; no penetration or explosions;
  report the max curvature during the push.

Verify: runs without errors; behavior as described. See docs/TODO.md -> M3.
"""
