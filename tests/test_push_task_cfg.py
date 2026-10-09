"""The push task's command, stem-state observations and stem spawn area use the decided values (guards against silent
changes; no simulator). Decisions: user, 2026-10-09 (spawn area; rewards and terminations, M4 step 3),
2026-10-08/09 (M4 step 1, docs/notes/2026-10-09.md).
"""

import math

import pytest

from isaaclab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg

from stem_manip.assets.stem import stem_params
from stem_manip.tasks.push_position import mdp
from stem_manip.tasks.push_position.env_cfg import STEM_MODEL, StemPushPositionEnvCfg

DECIDED_TARGET = {
    "distance_range": (0.03, 0.10),  # [m] horizontal distance from the tip's rest position
    "curvature_budget": 0.8,  # deepest target shape bends the stem to at most 0.8 x damage.max_curvature
    "angle_range": (-math.pi, math.pi),  # [rad] direction: full circle
}
SPAWN_AREA = {"x": (0.50, 0.65), "y": (-0.15, 0.15)}  # [m] stem base, robot base frame (= env frame); 2026-10-09
STEM_POINT_ARC_LENGTHS = (0.08, 0.16, 0.24, 0.32, 0.40)  # [m] 5 points up to the tip
POLICY_TERMS = [  # order of the concatenated policy observation (57 values)
    "tool_tip_pos",  # 3
    "tool_tip_rot",  # 6
    "applied_step",  # 6
    "target_offset",  # 6
    "stem_base",  # 3
    "stem_points",  # 2 x 15 (previous policy step, then the current one)
    "target_pos",  # 3
]
# reward terms (user, 2026-10-09): function, weight, decided params
REWARDS = {
    "distance_coarse": (mdp.stem_point_distance_tanh, 1.0, {"std": 0.05}),  # stem tip to target [m]
    "distance_fine": (mdp.stem_point_distance_tanh, 1.0, {"std": 0.01}),
    "height": (mdp.stem_point_height_tanh, 0.0, {"std": 0.003}),  # ready for the side-push optimum, off for now
    "approach": (mdp.approach_tanh, 0.5, {"std": 0.1}),  # tool tip to the nearest stem point [m]
    "curvature": (mdp.curvature_penalty, -1.0, {"soft_fraction": 0.8}),  # quadratic above 0.8 x the limit
    "contact_force": (mdp.contact_force_penalty, -1.0, {"threshold": 2.0}),  # quadratic above 2 N
    "action_rate": (mdp.action_rate_l2, -0.01, {}),
    "terminated": (mdp.is_terminated, -10.0, {}),  # early terminations only (not the time out)
}
TERMINATIONS = ["time_out", "curvature_limit", "contact_force_limit", "joint_margin"]  # no tool-tip bound (user)
MIN_JOINT_MARGIN = 0.25  # [rad] distance to any FR3 joint limit (the sweeps' rule)
EPISODE_LENGTH_S = 15.0  # [s] approach and push (user, 2026-10-09; was 10 s)


@pytest.fixture(scope="module")
def cfg() -> StemPushPositionEnvCfg:
    return StemPushPositionEnvCfg()


@pytest.mark.parametrize("field", DECIDED_TARGET)
def test_target_sampling_is_decided(cfg: StemPushPositionEnvCfg, field: str):
    assert getattr(cfg.commands.stem_target, field) == pytest.approx(DECIDED_TARGET[field], rel=1e-9)


def test_stem_point_is_the_tip(cfg: StemPushPositionEnvCfg):
    """Segment index + offset along it give the arc length of the stem's tip."""
    geometry = stem_params(STEM_MODEL)["geometry"]
    segment_length = geometry["length"] / geometry["num_segments"]
    command = cfg.commands.stem_target
    assert (command.segment_index + 0.5) * segment_length + command.offset == pytest.approx(geometry["length"])


def test_episode_length_is_decided(cfg: StemPushPositionEnvCfg):
    assert cfg.episode_length_s == pytest.approx(EPISODE_LENGTH_S, abs=1e-12)


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


def test_robot_base_frame_is_env_frame(cfg: StemPushPositionEnvCfg):
    """The spawn area is given in the robot base frame and written in the env frame: the two must coincide."""
    assert cfg.scene.robot.init_state.pos == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert cfg.scene.robot.init_state.rot == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-12)  # (x, y, z, w)


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
