"""Stage 1: push the stem so that a selected stem point reaches a target position.

Registers `StemManip-Push-Position-FR3-v0`. The RSL-RL agent cfg is added in M5 (`agents/rsl_rl_ppo_cfg.py`).

Verify: `uv run isaaclab list_envs` shows the task. See docs/TODO.md -> M4.
"""

import gymnasium as gym

gym.register(
    id="StemManip-Push-Position-FR3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:StemPushPositionEnvCfg"},
)
