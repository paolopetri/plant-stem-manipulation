"""The push task's RSL-RL PPO cfg uses the decided starting values and is registered with the task (no simulator).
Starting values: Isaac Lab's `core/lift/config/franka_soft/agents/rsl_rl_ppo_cfg.py` (M5 plan, user 2026-10-09).
"""

import gymnasium as gym
import pytest

from isaaclab_tasks.utils import load_cfg_from_registry

import stem_manip.tasks.push_position  # noqa: F401  (registers the task)
from stem_manip.tasks.push_position.agents.rsl_rl_ppo_cfg import StemPushPositionPPORunnerCfg

TASK = "StemManip-Push-Position-FR3-v0"
RUNNER = {
    "num_steps_per_env": 24,
    "save_interval": 50,
    "experiment_name": "stem_push_position",
    "obs_groups": {"actor": ["policy"], "critic": ["policy"]},  # no privileged critic obs yet (M5 second step)
    "clip_actions": None,  # the action term clips each axis to [-1, 1] (`integrate_target`)
    "logger": "wandb",
    "wandb_project": "stem-manip",
}
NETWORK = {"hidden_dims": [256, 128, 64], "activation": "elu", "obs_normalization": True}  # raw metres in the obs
INIT_STD = 1.0
ALGORITHM = {
    "value_loss_coef": 1.0,
    "use_clipped_value_loss": True,
    "clip_param": 0.2,
    "entropy_coef": 0.003,  # exp/m5-weekend run 06
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
    "learning_rate": 1.0e-3,
    "schedule": "adaptive",
    "gamma": 0.98,
    "lam": 0.95,
    "desired_kl": 0.01,
    "max_grad_norm": 1.0,
}


@pytest.fixture(scope="module")
def cfg() -> StemPushPositionPPORunnerCfg:
    return StemPushPositionPPORunnerCfg()


@pytest.mark.parametrize("field", RUNNER)
def test_runner_values_are_decided(cfg: StemPushPositionPPORunnerCfg, field: str):
    assert getattr(cfg, field) == RUNNER[field]


@pytest.mark.parametrize("model", ["actor", "critic"])
def test_networks_are_decided(cfg: StemPushPositionPPORunnerCfg, model: str):
    for field, value in NETWORK.items():
        assert getattr(getattr(cfg, model), field) == value, f"{model}.{field}"


def test_actor_initial_noise_std(cfg: StemPushPositionPPORunnerCfg):
    assert cfg.actor.distribution_cfg.init_std == INIT_STD


@pytest.mark.parametrize("field", ALGORITHM)
def test_ppo_values_are_decided(cfg: StemPushPositionPPORunnerCfg, field: str):
    assert getattr(cfg.algorithm, field) == pytest.approx(ALGORITHM[field])


def test_task_registers_the_rsl_rl_agent():
    """`isaaclab train --task ...` picks RSL-RL (default agent) and loads our runner cfg from the registry."""
    assert gym.spec(TASK).kwargs["default_agent"] == "rsl_rl"
    assert isinstance(load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point"), StemPushPositionPPORunnerCfg)
