"""Stage 1: push the stem so that a selected stem point reaches a target position.

Requirements:
- Register one Gymnasium env when `env_cfg.py` exists, e.g.
  `gym.register(id="StemManip-Push-Position-FR3-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv",
  kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:StemPushPositionEnvCfg",
  "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:StemPushPositionPPORunnerCfg"})`.
- Not registered yet: an entry point to a missing cfg would break `isaaclab list_envs`.

Verify: `uv run isaaclab list_envs` shows the task. See docs/TODO.md -> M4.
"""
