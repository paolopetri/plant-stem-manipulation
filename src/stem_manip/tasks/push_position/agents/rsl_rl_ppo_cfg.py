"""RSL-RL PPO runner cfg for stage 1.

Requirements:
- `StemPushPositionPPORunnerCfg(RslRlOnPolicyRunnerCfg)` with actor/critic MLP and PPO hyperparameters as fields.
  Starting point: the Franka cable lift agent cfg in IsaacLab `core/lift/config/franka_soft/agents/`.
- Check that the run logs our repo's git commit (RSL-RL stores the git state of registered repos).

Verify: short headless training run, mean reward increases. See docs/TODO.md -> M5.
"""
