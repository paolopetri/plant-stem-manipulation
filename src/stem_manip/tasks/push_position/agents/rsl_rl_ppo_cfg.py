"""RSL-RL PPO runner cfg for stage 1.

Starting values from Isaac Lab's Franka cable lift agent (`core/lift/config/franka_soft/agents/rsl_rl_ppo_cfg.py`,
M5 plan, user 2026-10-09). Logs to wandb (project `stem-manip`); RSL-RL's wandb writer also writes the local
tensorboard files. Our repo's commit is logged through the env cfg (`git_commit` -> `params/env.yaml`, wandb config).

Verify: `tests/test_push_agent_cfg.py`; short headless training run, mean reward increases. See docs/TODO.md -> M5.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class StemPushPositionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO with actor and critic MLPs on the policy observation (57 values, raw metres: normalized empirically)."""

    # hyperparameters: best config of the M5 weekend sweep (run 13), see docs/experiments/2026-10-10-m5-weekend.md
    num_steps_per_env = 48
    max_iterations = 1500  # placeholder until the speed benchmark (M5 plan, step 5)
    save_interval = 50
    experiment_name = "stem_push_position"
    logger = "wandb"
    wandb_project = "stem-manip"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.012,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        # 0.98 looks ~50 policy steps (1.6 s) ahead in a 15 s episode (0.99: ~100 steps, 3.2 s)
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
