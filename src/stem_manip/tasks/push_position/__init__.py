"""Stage 1: push the stem so that a selected stem point reaches a target position.

Registers `StemManip-Push-Position-FR3-v0` with the RSL-RL PPO agent (`agents/rsl_rl_ppo_cfg.py`, default agent, so
`isaaclab train --task ...` needs no `--rl_library`).

Verify: `uv run isaaclab list_envs` shows the task; `tests/test_push_agent_cfg.py`. See docs/TODO.md -> M5.
"""

import gymnasium as gym

gym.register(
    id="StemManip-Push-Position-FR3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:StemPushPositionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:StemPushPositionPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
