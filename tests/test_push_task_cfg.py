"""The push task's command, stem-state observations and stem spawn area use the decided values (guards against silent
changes; no simulator). Decisions: user, 2026-10-09 (spawn area), 2026-10-08/09 (M4 step 1, docs/notes/2026-10-09.md).
"""

import math

import pytest

from isaaclab.managers import EventTermCfg, ObservationTermCfg

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
