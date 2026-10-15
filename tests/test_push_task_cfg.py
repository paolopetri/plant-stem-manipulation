"""The push task's command, stem-state observations and stem spawn area use the decided values (guards against silent
changes; no simulator). Decisions: user, 2026-10-09 (spawn area; rewards and terminations),
2026-10-08/09 (target command and stem-state observations, docs/notes/2026-10-09.md).
"""

import math
import subprocess
from pathlib import Path

import pytest

from isaaclab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg

from stem_manip.assets.stem import stem_params
from stem_manip.tasks.push_position import mdp
from stem_manip.tasks.push_position.env_cfg import START_POSES, STEM_MODEL, StemPushPositionEnvCfg

DECIDED_TARGET = {
    "distance_range": (0.03, 0.10),  # [m] horizontal distance from the tip's rest position
    "curvature_budget": 0.8,  # deepest target shape bends the stem to at most 0.8 x damage.max_curvature
    "angle_range": (-math.pi, math.pi),  # [rad] direction: full circle
}
SPAWN_AREA = {"x": (0.50, 0.65), "y": (-0.15, 0.15)}  # [m] stem base, robot base frame (= env frame); 2026-10-09
STEM_POINT_ARC_LENGTHS = (0.08, 0.16, 0.24, 0.32, 0.40)  # [m] 5 points up to the tip
POLICY_TERMS = [  # order of the concatenated policy observation (57 values)
    "slot_base_pos",  # 3
    "slot_base_rot",  # 6
    "applied_step",  # 6
    "setpoint_offset",  # 6
    "stem_base",  # 3
    "stem_points",  # 2 x 15 (previous policy step, then the current one)
    "target_pos",  # 3
]
# reward terms (user, 2026-10-09): function, weight, decided params
REWARDS = {
    "distance_coarse": (mdp.stem_point_distance_tanh, 1.0, {"std": 0.05}),  # stem tip to target [m]
    # ellipsoid (user, 2026-10-10): height error scaled by 10/3, i.e. std 1 cm sideways, 3 mm in height
    "distance_fine": (mdp.stem_point_distance_tanh, 1.0, {"std": 0.01, "z_scale": 0.01 / 0.003}),
    # run 13 config (docs/experiments/2026-10-10-m5-weekend.md)
    "height": (mdp.stem_point_height_tanh, 0.3, {"std": 0.003}),
    "approach": (mdp.approach_tanh, 0.5, {"std": 0.1}),  # slot base to the nearest stem point [m]
    "curvature": (mdp.curvature_penalty, -1.0, {"curvature_budget": 0.8}),  # quadratic above 0.8 x the limit
    "contact_force": (mdp.contact_force_penalty, -1.0, {"threshold": 2.0}),  # quadratic above 2 N
    "action_rate": (mdp.action_rate_l2, -0.01, {}),
    # early terminations only, not the time out; about -5 per termination at dt 0.032 s (user, 2026-10-15)
    "terminated": (mdp.is_terminated, -150.0, {}),
}
POLICY_DT = 0.032  # [s] 16 physics steps of 2 ms (31.25 Hz)
TERMINATION_PENALTY = -4.8  # in effect per termination: weight x policy dt (about -5; user, 2026-10-15)
TERMINATIONS = ["time_out", "curvature_limit", "contact_force_limit", "joint_margin"]  # no slot-base bound (user)
MIN_JOINT_MARGIN = 0.25  # [rad] distance to any FR3 joint limit (the sweeps' rule)
EPISODE_LENGTH_S = 15.0  # [s] approach and push (user, 2026-10-09)


@pytest.fixture(scope="module")
def cfg() -> StemPushPositionEnvCfg:
    return StemPushPositionEnvCfg()


@pytest.mark.parametrize("field", DECIDED_TARGET)
def test_target_sampling_is_decided(cfg: StemPushPositionEnvCfg, field: str):
    assert getattr(cfg.commands.stem_target, field) == pytest.approx(DECIDED_TARGET[field], rel=1e-9)


def test_stem_point_is_the_tip(cfg: StemPushPositionEnvCfg):
    """The command's arc length is the stem length (the tip), so a change of stem length cannot silently move the stem
    point off the tip."""
    assert cfg.commands.stem_target.arc_length == pytest.approx(stem_params(STEM_MODEL)["geometry"]["length"])


def test_stem_model_is_named_once(cfg: StemPushPositionEnvCfg):
    """The env cfg names the stem model (`chain`, supervisors 2026-10-06); the terms that read the stem take no
    stem-model or scene-entity params and the target command has no stem-model field (they read it through the stem
    state)."""
    assert cfg.stem_model == "chain"
    stem_terms = {
        "stem_base": cfg.observations.policy.stem_base,
        "stem_points": cfg.observations.policy.stem_points,
        "approach": cfg.rewards.approach,
        "curvature": cfg.rewards.curvature,
        "curvature_limit": cfg.terminations.curvature_limit,
        "fix_stem_base": cfg.events.fix_stem_base,
    }
    for name, term in stem_terms.items():
        assert not {"model", "asset_cfg", "robot_cfg"} & set(term.params), name
    assert not hasattr(cfg.commands.stem_target, "stem_model")


def test_episode_length_is_decided(cfg: StemPushPositionEnvCfg):
    assert cfg.episode_length_s == pytest.approx(EPISODE_LENGTH_S, abs=1e-12)


def test_git_commit_is_logged(cfg: StemPushPositionEnvCfg):
    """Our repo's commit (+ "-dirty" with uncommitted changes to tracked files) is part of the env cfg, so it ends up
    in every run's `params/env.yaml` and wandb config."""
    repo = Path(__file__).resolve().parents[1]
    git = lambda *args: subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).stdout.strip()  # noqa: E731
    dirty = git("status", "--porcelain", "--untracked-files=no") != ""
    assert cfg.git_commit == git("rev-parse", "--short=12", "HEAD") + ("-dirty" if dirty else "")


def test_target_is_sampled_once_per_episode(cfg: StemPushPositionEnvCfg):
    assert cfg.commands.stem_target.resampling_time_range[0] > cfg.episode_length_s


def test_policy_observation_terms(cfg: StemPushPositionEnvCfg):
    policy = cfg.observations.policy
    declared = [name for name, value in vars(policy).items() if isinstance(value, ObservationTermCfg)]
    assert declared == POLICY_TERMS  # the order of the concatenated observation
    assert policy.stem_base.params["arc_lengths"] == (0.0,)
    assert policy.stem_points.params["arc_lengths"] == pytest.approx(STEM_POINT_ARC_LENGTHS)
    assert policy.stem_points.history_length == 2
    assert policy.stem_base.history_length == 0


def test_stem_spawn_area_is_decided(cfg: StemPushPositionEnvCfg):
    """The stem base is reset uniformly within the decided area (absolute, env frame), after the default reset."""
    events = [name for name, value in vars(cfg.events).items() if isinstance(value, EventTermCfg)]
    assert events.index("spawn_stem") > events.index("reset_scene")  # overrides the default root pose
    spawn = cfg.events.spawn_stem
    assert spawn.func is mdp.reset_stem_base_uniform
    assert spawn.mode == "reset"
    assert spawn.params["asset_cfg"].name == "stem"
    assert spawn.params["x_range"] == pytest.approx(SPAWN_AREA["x"], abs=1e-12)
    assert spawn.params["y_range"] == pytest.approx(SPAWN_AREA["y"], abs=1e-12)


def test_spawn_redraws_stem_positions_at_the_fork(cfg: StemPushPositionEnvCfg):
    """Fork centre line from 0.09 m behind the slot base to the prong tips, 5 cm from the stem axis (user,
    2026-10-15, #163)."""
    spawn = cfg.events.spawn_stem
    assert spawn.params["fork_extent"] == pytest.approx((-0.09, 0.17), abs=1e-12)
    assert spawn.params["min_fork_distance"] == pytest.approx(0.05, abs=1e-12)


def test_spawn_redraws_stem_positions_at_the_arm(cfg: StemPushPositionEnvCfg):
    """Stem surface at least 5 cm from every arm body, as the start-pose table (user, 2026-10-15, #163)."""
    assert cfg.events.spawn_stem.params["min_arm_clearance"] == pytest.approx(0.05, abs=1e-12)


def test_start_pose_is_picked_from_the_table(cfg: StemPushPositionEnvCfg):
    """The arm starts at a random table pose with uniform noise of 0.02 rad (proposal, 2026-10-15, #163), before the
    stem is placed (the spawn event keeps the stem clear of the fork at its start pose)."""
    events = [name for name, value in vars(cfg.events).items() if isinstance(value, EventTermCfg)]
    assert events.index("reset_scene") < events.index("start_pose") < events.index("spawn_stem")
    start_pose = cfg.events.start_pose
    assert start_pose.func is mdp.reset_arm_start_pose
    assert start_pose.mode == "reset"
    assert start_pose.params["poses"] == START_POSES
    assert start_pose.params["joint_noise"] == pytest.approx(0.02, abs=1e-12)
    asset_cfg = start_pose.params["asset_cfg"]
    assert asset_cfg.name == "robot"
    assert asset_cfg.joint_names == [f"fr3_joint{i}" for i in range(1, 8)] and asset_cfg.preserve_order


def test_robot_base_frame_is_env_frame(cfg: StemPushPositionEnvCfg):
    """The spawn area is given in the robot base frame and written in the env frame: the two must coincide."""
    assert cfg.scene.robot.init_state.pos == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert cfg.scene.robot.init_state.rot == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-12)  # (x, y, z, w)


def test_termination_penalty_in_effect(cfg: StemPushPositionEnvCfg):
    """The reward manager scales a term by the policy dt: the termination penalty costs about -5 per termination."""
    assert cfg.sim.dt * cfg.decimation == pytest.approx(POLICY_DT, rel=1e-9)
    assert cfg.rewards.terminated.weight * cfg.sim.dt * cfg.decimation == pytest.approx(TERMINATION_PENALTY, rel=1e-9)


def test_reward_terms_are_decided(cfg: StemPushPositionEnvCfg):
    declared = {name: value for name, value in vars(cfg.rewards).items() if isinstance(value, RewardTermCfg)}
    assert list(declared) == list(REWARDS)
    for name, (func, weight, params) in REWARDS.items():
        term = declared[name]
        assert term.func is func, name
        assert term.weight == pytest.approx(weight, abs=1e-12), name
        for key, value in params.items():
            assert term.params[key] == pytest.approx(value, rel=1e-9), (name, key)


def test_damage_limits_come_from_stem_yaml(cfg: StemPushPositionEnvCfg):
    """The curvature and contact-force limits are read from `stem.yaml`, not repeated in the task."""
    damage = stem_params(STEM_MODEL)["damage"]
    assert cfg.rewards.curvature.params["max_curvature"] == damage["max_curvature"]
    assert cfg.terminations.curvature_limit.params["max_curvature"] == damage["max_curvature"]
    assert cfg.terminations.contact_force_limit.params["max_force"] == damage["max_contact_force"]


def test_termination_terms_are_decided(cfg: StemPushPositionEnvCfg):
    declared = {name: value for name, value in vars(cfg.terminations).items() if isinstance(value, TerminationTermCfg)}
    assert list(declared) == TERMINATIONS
    assert declared["time_out"].time_out
    assert not any(declared[name].time_out for name in TERMINATIONS[1:])  # damage / joint margin: real terminations
    assert declared["curvature_limit"].func is mdp.curvature_limit
    assert declared["contact_force_limit"].func is mdp.contact_force_limit
    assert declared["joint_margin"].func is mdp.joint_limit_margin
    assert declared["joint_margin"].params["min_margin"] == pytest.approx(MIN_JOINT_MARGIN, abs=1e-12)


def test_contact_sensor_averages_over_the_policy_step(cfg: StemPushPositionEnvCfg):
    """One history entry per physics step of the policy step; reward and termination read the same sensor."""
    sensor = cfg.scene.stem_contact
    assert sensor.history_length == cfg.decimation
    assert sensor.update_period == 0.0
    assert sensor.track_friction_forces
    name = cfg.rewards.contact_force.params["sensor_cfg"].name
    assert name == "stem_contact" == cfg.terminations.contact_force_limit.params["sensor_cfg"].name
