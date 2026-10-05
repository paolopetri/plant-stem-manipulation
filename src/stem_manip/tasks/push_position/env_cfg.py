"""Manager-based env cfg for stage 1 (position control of the stem point).

Requirements:
- Scene: FR3 + fork (`stem_manip.assets.fr3.fr3_cfg("fork_v2")`), stem (`stem_model(name).stem_cfg()` from `stem_manip.assets.stem`),
  ground, `FrameTransformer` on `fork_v2` + `tool_tip_offset("fork_v2")`.
- Physics: Newton with `CouplerProxyCfg` (robot in MuJoCo-Warp, stem in VBD, fork as proxy collider).
  Template: IsaacLab `isaaclab_tasks/core/lift/config/franka_soft/franka_cable_env_cfg.py`.
- Actions: relative end-effector position (differential IK on `fork_v2` with the tool_tip offset).
- Managers from `mdp/` (commands, observations, rewards, terminations, events); weights and ranges as cfg fields.
- The point of interest (segment index + offset) and the curvature limit come from cfg / `assets/stem/stem.yaml`.

Verify: zero/random agent runs headless with few envs; shapes and ranges as expected. See docs/TODO.md -> M4.
"""
