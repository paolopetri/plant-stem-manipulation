"""RSL-RL PPO runner cfg for stage 1.

Requirements:
- `StemPushPositionPPORunnerCfg(RslRlOnPolicyRunnerCfg)` with actor/critic MLP and PPO hyperparameters as fields.
  Starting point: the Franka cable lift agent cfg in IsaacLab `core/lift/config/franka_soft/agents/`.
- Our repo's git commit must be logged with every run. Isaac Lab's `train` only records the Isaac Lab and
  RSL-RL repos, so this needs our own addition (e.g. a commit-hash field in the env cfg -> `params/env.yaml`).

Verify: short headless training run, mean reward increases. See docs/TODO.md -> M5.
"""
