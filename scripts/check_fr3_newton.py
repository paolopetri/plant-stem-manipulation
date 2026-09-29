"""Sanity check of the FR3 + fork on Newton / MuJoCo-Warp.

Requirements:
- Load `fr3_cfg("fork")` with a Newton physics cfg (the cfg currently uses PhysX-only schemas; adapt as needed).
- Check: bodies `fr3_link0`..`fr3_link7` + `fork`; start pose held (report sag, cf. the gravity question);
  differential IK on `fork` + `tool_tip_offset("fork")` moves the tool_tip to a target;
  a `FrameTransformer` with the same offset reports the tool_tip pose.

Verify: runs without errors; checks pass. See docs/TODO.md -> M2.
"""
