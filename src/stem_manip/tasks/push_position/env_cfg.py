"""Manager-based env cfg for stage 1 (position control of the stem point).

- Scene: FR3 + fork (`stem_manip.assets.fr3.fr3_cfg("fork_v2")`), stem (`stem_model(name).stem_cfg()` from
  `stem_manip.assets.stem`), ground. Stem model `chain`: robot and stem in one PhysX scene (as in
  `scripts/check_contact.py`).
- Actions: relative slot-base pose (position and rotation step, robot base frame), executed by Franka's Cartesian
  impedance law (`mdp.SlotBaseImpedanceAction`, the same controller as on the real FR3); robot in torque mode.
- Managers from `mdp/` (commands, observations, rewards, terminations, events); weights and ranges as cfg fields.
- The stem point (its arc length, the stem length = the tip) is a task-cfg field (`CommandsCfg`); the curvature limit
  comes from `assets/stem/stem.yaml`.
- Contact sensor on the stem: forces for rewards and terminations only, not observed.

Verify: `scripts/check_push_env.py`, `scripts/check_push_terms.py`; zero/random agent runs headless with few envs.
"""

import math
import subprocess
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from stem_manip.assets.fr3 import fr3_cfg, slot_base_offset
from stem_manip.assets.stem import stem_model, stem_params

from . import mdp

STEM_MODEL = "chain"
END_EFFECTOR = "fork_v2"
ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]  # order of the joint angles in the start poses below
# Held-out evaluation start poses (user, 2026-10-15, #160), arm joint angles [rad], not used in training:
# - a_above_tip: fork horizontal, pointing forward, slot base at (0.40, 0, 0.50) m, 10 cm above the stem tip and
#   behind the stem area; solved with differential IK, smallest distance to a joint limit 0.58 rad (joint 4). Also the
#   robot's default joint position (the null-space rest posture of the action term is the start pose of each episode).
# - b_side_90, c_low: beside the stem like the table, at azimuth 90 deg and at a slot-base height of 0.08 m; the other
#   coordinate (b: height 0.30 m, c: azimuth 60 deg) is a proposal (2026-10-15, #163). Solved by
#   `scripts/start_poses.py solve`.
EVAL_START_POSES = {
    "a_above_tip": (-0.013, -0.689, 0.011, -2.496, 0.007, 1.808, 0.779),
    "b_side_90": (-0.515, 0.378, 0.096, -1.873, -0.046, 2.249, -1.183),  # slot base 0.30 m, +90 deg; margin 1.20 rad
    "c_low": (-0.706, 0.588, 0.174, -2.318, -0.382, 2.881, -0.453),  # slot base 0.08 m, +60 deg; margin 0.76 rad
}
# Start-pose table (user, 2026-10-15, #160): slot base beside the stem, fork horizontal, slot facing the stem axis,
# for the stem at the spawn-area centre; `START_POSE_DISTANCE` from the stem axis (prongs 0.17 m + stem radius + 3 cm
# gap), slot-base height in `START_POSE_HEIGHTS`, azimuth about the stem axis within +-`START_POSE_AZIMUTH` of the
# direction to the robot. Poses with a joint margin below `START_POSE_MIN_MARGIN` or an arm body (any robot body but
# the fork) closer than `START_POSE_ARM_CLEARANCE` to the stem are rejected. With the 0.5 rad margin only azimuths of
# about 45-60 deg remain (joint 4 folds too far towards the robot; user, 2026-10-15).
START_POSE_DISTANCE = 0.205  # [m] stem axis to slot base, horizontal
START_POSE_HEIGHTS = (0.10, 0.45)  # [m] slot-base height
START_POSE_AZIMUTH = math.radians(60.0)  # [rad]
START_POSE_MIN_MARGIN = 0.5  # [rad]
START_POSE_ARM_CLEARANCE = 0.05  # [m] arm surface to the stem surface
# Arm joint angles [rad] (`ARM_JOINT_NAMES` order), solved by `scripts/start_poses.py solve`, seeded at a_above_tip.
START_POSES = (
    (0.699, 0.528, -0.178, -2.343, 0.317, 2.851, 2.073),  # slot base 0.10 m, -60 deg; margin 0.73 rad
    (0.666, 0.461, -0.147, -2.490, 0.325, 2.937, 1.874),  # slot base 0.10 m, -50 deg; margin 0.59 rad
    (-0.667, 0.462, 0.148, -2.490, -0.327, 2.937, -0.301),  # slot base 0.10 m, +50 deg; margin 0.59 rad
    (-0.701, 0.529, 0.180, -2.343, -0.320, 2.850, -0.500),  # slot base 0.10 m, +60 deg; margin 0.73 rad
    (0.629, 0.381, -0.138, -2.387, 0.139, 2.762, 2.204),  # slot base 0.15 m, -60 deg; margin 0.69 rad
    (0.576, 0.299, -0.101, -2.538, 0.098, 2.834, 2.044),  # slot base 0.15 m, -50 deg; margin 0.54 rad
    (-0.585, 0.300, 0.109, -2.538, -0.106, 2.834, -0.468),  # slot base 0.15 m, +50 deg; margin 0.54 rad
    (-0.637, 0.382, 0.145, -2.387, -0.146, 2.761, -0.628),  # slot base 0.15 m, +60 deg; margin 0.69 rad
    (0.489, 0.246, -0.016, -2.405, 0.008, 2.651, 2.298),  # slot base 0.20 m, -60 deg; margin 0.67 rad
    (0.386, 0.151, 0.068, -2.556, -0.024, 2.707, 2.133),  # slot base 0.20 m, -50 deg; margin 0.52 rad
    (-0.404, 0.151, -0.051, -2.556, 0.018, 2.707, -0.558),  # slot base 0.20 m, +50 deg; margin 0.52 rad
    (-0.503, 0.246, 0.030, -2.405, -0.016, 2.651, -0.722),  # slot base 0.20 m, +60 deg; margin 0.67 rad
    (0.353, 0.128, 0.117, -2.400, -0.026, 2.528, 2.322),  # slot base 0.25 m, -60 deg; margin 0.68 rad
    (0.234, 0.018, 0.223, -2.551, -0.007, 2.569, 2.121),  # slot base 0.25 m, -50 deg; margin 0.53 rad
    (-0.257, 0.018, -0.199, -2.551, 0.007, 2.569, -0.549),  # slot base 0.25 m, +50 deg; margin 0.53 rad
    (-0.372, 0.128, -0.098, -2.400, 0.022, 2.528, -0.749),  # slot base 0.25 m, +60 deg; margin 0.68 rad
    (0.247, 0.025, 0.224, -2.375, -0.008, 2.400, 2.310),  # slot base 0.30 m, -60 deg; margin 0.70 rad
    (0.133, -0.102, 0.328, -2.524, 0.050, 2.426, 2.079),  # slot base 0.30 m, -50 deg; margin 0.55 rad
    (-0.160, -0.101, -0.300, -2.524, -0.046, 2.426, -0.511),  # slot base 0.30 m, +50 deg; margin 0.55 rad
    (-0.270, 0.025, -0.201, -2.375, 0.007, 2.400, -0.739),  # slot base 0.30 m, +60 deg; margin 0.70 rad
    (0.169, -0.062, 0.301, -2.329, 0.024, 2.269, 2.286),  # slot base 0.35 m, -60 deg; margin 0.75 rad
    (0.066, -0.205, 0.388, -2.474, 0.102, 2.282, 2.039),  # slot base 0.35 m, -50 deg; margin 0.60 rad
    (-0.097, -0.202, -0.358, -2.474, -0.093, 2.282, -0.474),  # slot base 0.35 m, +50 deg; margin 0.60 rad
    (-0.195, -0.062, -0.274, -2.329, -0.022, 2.269, -0.717),  # slot base 0.35 m, +60 deg; margin 0.75 rad
    (0.113, -0.133, 0.350, -2.264, 0.054, 2.138, 2.263),  # slot base 0.40 m, -60 deg; margin 0.79 rad
    (0.023, -0.284, 0.417, -2.405, 0.135, 2.140, 2.010),  # slot base 0.40 m, -50 deg; margin 0.67 rad
    (-0.006, -0.430, 0.407, -2.530, 0.195, 2.122, 1.748),  # slot base 0.40 m, -40 deg; margin 0.55 rad
    (-0.025, -0.425, -0.378, -2.530, -0.180, 2.125, -0.189),  # slot base 0.40 m, +40 deg; margin 0.55 rad
    (-0.055, -0.281, -0.386, -2.405, -0.124, 2.141, -0.447),  # slot base 0.40 m, +50 deg; margin 0.67 rad
    (-0.141, -0.131, -0.322, -2.264, -0.049, 2.139, -0.696),  # slot base 0.40 m, +60 deg; margin 0.81 rad
    (0.074, -0.183, 0.377, -2.180, 0.074, 2.008, 2.246),  # slot base 0.45 m, -60 deg; margin 0.80 rad
    (-0.006, -0.339, 0.425, -2.317, 0.152, 2.002, 1.992),  # slot base 0.45 m, -50 deg; margin 0.76 rad
    (-0.026, -0.483, 0.401, -2.438, 0.199, 1.980, 1.736),  # slot base 0.45 m, -40 deg; margin 0.64 rad
    (-0.016, -0.605, 0.329, -2.539, 0.199, 1.952, 1.492),  # slot base 0.45 m, -30 deg; margin 0.54 rad
    (-0.012, -0.600, -0.305, -2.539, -0.184, 1.955, 0.065),  # slot base 0.45 m, +30 deg; margin 0.54 rad
    (-0.005, -0.478, -0.373, -2.438, -0.184, 1.983, -0.178),  # slot base 0.45 m, +40 deg; margin 0.64 rad
    (-0.027, -0.334, -0.395, -2.318, -0.140, 2.004, -0.431),  # slot base 0.45 m, +50 deg; margin 0.76 rad
    (-0.105, -0.181, -0.348, -2.180, -0.068, 2.009, -0.680),  # slot base 0.45 m, +60 deg; margin 0.90 rad
)
# [rad] uniform noise on every arm joint at reset (proposal, 2026-10-15, #163)
START_JOINT_NOISE = 0.02
# Fork vs a spawned stem (user, 2026-10-15, #163): the spawn event redraws stem positions whose axis comes closer than
# `FORK_STEM_MIN_DISTANCE` to the fork's centre line, the slot base's x axis from `FORK_AXIS_EXTENT[0]` to
# `FORK_AXIS_EXTENT[1]` (collision mesh: 0.088 m behind the slot base to the prong tips 0.170 m ahead). Distance = the
# fork's half cross-section (31.5 x 22 mm -> 38 mm) + stem radius 4 mm + 8 mm clearance.
FORK_AXIS_EXTENT = (-0.09, 0.17)  # [m] along the slot base's x axis
FORK_STEM_MIN_DISTANCE = 0.05  # [m]
# Arm vs a spawned stem (user, 2026-10-15, #163): the spawn event also redraws stem positions whose surface comes closer
# than this to an arm body (any robot body but the fork), the start-pose table's clearance. Wherever the fork is clear,
# the table poses and a_above_tip / c_low keep the arm at least 68 mm away; only b_side_90 needs it (2026-10-15).
STEM_MIN_ARM_CLEARANCE = START_POSE_ARM_CLEARANCE  # [m]

# Stem spawn area (user, 2026-10-09): the stem base is reset uniformly within this rectangle on the lab's plate in front
# of the robot (58 x 58 cm at the mounting-surface height, x 0.075-0.655 m, y +-0.29 m), robot base frame [m]. Near
# edge from joint 4's limit (margin >= 0.25 rad with the fork horizontal: slot base at x >= 0.40 m at the stem-tip
# height and >= 0.45 m lower down, and a target up to 10 cm towards the robot), far edge from the plate. Assumes robot
# base frame = env frame (robot root at the env origin, not rotated). The plate is not modelled as geometry (infinite
# ground plane); the fork may leave it.
STEM_SPAWN_X = (0.50, 0.65)
STEM_SPAWN_Y = (-0.15, 0.15)
# [m] stem axis the start-pose table is solved for: the spawn-area centre (user, 2026-10-15, #163)
START_POSE_STEM_XY = (sum(STEM_SPAWN_X) / 2, sum(STEM_SPAWN_Y) / 2)
# The spawn event ignores the stem's `init_state.pos`: to place the stem elsewhere, also set `events.spawn_stem = None`.
# [m] arc length of the target command's stem point from the stem base: the stem length, i.e. the tip (user, 2026-10-08)
STEM_POINT_ARC_LENGTH = 0.40

# Rewards and terminations (user, 2026-10-09; docs/notes/2026-10-09.md). Damage limits come from `stem.yaml`.
DAMAGE = stem_params(STEM_MODEL)["damage"]
DISTANCE_STD = (0.05, 0.01)  # [m] stem tip to target: coarse (targets 3-10 cm away), fine (success within 1 cm)
# [m] height error alone. One term, no coarse / fine pair (user, 2026-10-09): height errors span only ~0-33 mm (bowl
# drop up to ~15 mm + up to ~18 mm deeper); the side-push gap of 0-18 mm lies on this tanh's slope, and larger
# height errors occur only before the push, where the 3-D coarse term already gives the gradient.
HEIGHT_STD = 0.003
APPROACH_STD = 0.1  # [m] slot base to the nearest stem point (start: the fork is 0.2-0.4 m from the stem)
# curvature budget (user, 2026-10-09): 0.8 x the limit, i.e. 4 1/m, for clear C / S bends that need the fork's slot;
# the targets stay inside it and the curvature penalty starts at it
CURVATURE_BUDGET = 0.8
# [N] contact penalty above it, on the largest single contact (normal pushes 0.8-1.4 N measured; clamp bound 4 N)
CONTACT_FREE_FORCE = 2.0
MIN_JOINT_MARGIN = 0.25  # [rad] terminate closer to any FR3 joint limit (the sweeps' rule, 2026-10-08)
ARM_JOINTS = ["fr3_joint[1-7]"]
ROBOT_LINKS = [f"fr3_link{i}" for i in range(8)] + [END_EFFECTOR]  # all robot bodies that can touch the stem


def git_commit() -> str:
    """This repo's commit (12 characters, + "-dirty" with uncommitted changes to tracked files), logged with every run
    through the env cfg (`params/env.yaml`, wandb config); Isaac Lab only logs its own repo."""
    describe = ["git", "describe", "--always", "--dirty", "--abbrev=12", "--exclude=*"]  # hash, never a tag name
    result = subprocess.run(describe, cwd=Path(__file__).parent, capture_output=True, text=True)
    return result.stdout.strip() or "unknown"


@configclass
class StemPushSceneCfg(InteractiveSceneCfg):
    """Ground at the height of the robot's mounting surface, light, FR3 with the fork, the stem."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot = fr3_cfg(END_EFFECTOR, control="torque").replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=dict(zip(ARM_JOINT_NAMES, EVAL_START_POSES["a_above_tip"], strict=True))
        ),
    )
    stem = stem_model(STEM_MODEL).stem_cfg().replace(prim_path="{ENV_REGEX_NS}/Stem")
    # force of the robot (fork and arm links) on each stem segment, normal and friction, at every physics step;
    # history = one policy step (set in the env cfg's __post_init__). As in `scripts/check_contact.py`.
    stem_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Stem/seg_.*",
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Robot/.*{link}" for link in ROBOT_LINKS],
        track_friction_forces=True,
    )


@configclass
class ActionsCfg:
    """Relative slot-base pose (at most `max_step` / `max_rot_step` per policy step), Franka's Cartesian
    impedance law."""

    slot_base = mdp.SlotBaseImpedanceActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        body_name=END_EFFECTOR,
        slot_base_offset=slot_base_offset(END_EFFECTOR),
    )


@configclass
class CommandsCfg:
    """Target position of the stem tip, sampled once per episode (decided values: cfg defaults, user 2026-10-08), and
    the stem metrics (a metrics-only term, `mdp/metrics.py`)."""

    # resampling time far above the episode length: the target is sampled only at reset
    stem_target = mdp.StemTipTargetCommandCfg(
        arc_length=STEM_POINT_ARC_LENGTH,
        curvature_budget=CURVATURE_BUDGET,
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
    )
    # metrics only, logged and never observed: no command, nothing to resample
    stem_metrics = mdp.StemMetricsCfg(resampling_time_range=(1e9, 1e9))


@configclass
class ObservationsCfg:
    """Policy observations, 57 values in this order: slot base (position 3, orientation 6, applied step 6, setpoint
    offset 6), stem base 3, 5 stem points of the previous and the current policy step 30 (oldest first), target 3.
    All positions in the robot base frame [m]. The stem-state terms (`mdp/stem_state.py`) are to be changed to what
    the perception delivers; no contact forces (decision 2026-10-06)."""

    @configclass
    class PolicyCfg(ObsGroup):
        slot_base_pos = ObsTerm(func=mdp.slot_base_pos)
        slot_base_rot = ObsTerm(func=mdp.slot_base_rot6d)
        applied_step = ObsTerm(func=mdp.applied_step)
        setpoint_offset = ObsTerm(func=mdp.setpoint_offset)
        stem_base = ObsTerm(func=mdp.stem_points, params={"arc_lengths": (0.0,)})
        # 5 points up to the tip, plus the same points one policy step earlier (the stem's motion; user, 2026-10-08)
        stem_points = ObsTerm(
            func=mdp.stem_points,
            params={"arc_lengths": (0.08, 0.16, 0.24, 0.32, 0.40)},
            history_length=2,
        )
        target_pos = ObsTerm(func=mdp.generated_commands, params={"command_name": "stem_target"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Base clamping at startup; robot and stem back to their default states on reset, the arm to a start pose picked
    from the table (`poses`, overridable, e.g. through Hydra at play time), then the stem base moved to a random place
    in the spawn area, clear of the fork and the arm (upright, at rest; before the target command is resampled)."""

    fix_stem_base = EventTerm(func=mdp.fix_stem_base, mode="startup")
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    start_pose = EventTerm(
        func=mdp.reset_arm_start_pose,
        mode="reset",
        params={
            "poses": START_POSES,
            "joint_noise": START_JOINT_NOISE,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, preserve_order=True),
        },
    )
    spawn_stem = EventTerm(
        func=mdp.reset_stem_base_uniform,
        mode="reset",
        params={
            "x_range": STEM_SPAWN_X,
            "y_range": STEM_SPAWN_Y,
            "fork_extent": FORK_AXIS_EXTENT,
            "min_fork_distance": FORK_STEM_MIN_DISTANCE,
            "min_arm_clearance": STEM_MIN_ARM_CLEARANCE,
            "asset_cfg": SceneEntityCfg("stem"),
        },
    )


@configclass
class RewardsCfg:
    """Task (stem tip to target), approach (fork to stem), damage penalties below the limits, smooth motion, early
    terminations. Weights: a first guess for the baseline (user, 2026-10-09); termination penalty 2026-10-15."""

    distance_coarse = RewTerm(func=mdp.stem_point_distance_tanh, weight=1.0, params={"std": DISTANCE_STD[0]})
    # ellipsoid (user, 2026-10-10): height error scaled to std HEIGHT_STD (3 mm), so that a side push stalling on the
    # bowl a few mm above the target loses most of this term (5 mm: 7 % left instead of 54 % with the 1 cm sphere)
    distance_fine = RewTerm(
        func=mdp.stem_point_distance_tanh,
        weight=1.0,
        params={"std": DISTANCE_STD[1], "z_scale": DISTANCE_STD[1] / HEIGHT_STD},
    )
    # on against the side-push stall on the bowl (`height_error` stayed at +3 mm with weight 0); 0.3: weight 1.0 traded
    # sideways accuracy for height (M5 weekend runs 04-05, docs/experiments/2026-10-10-m5-weekend.md)
    height = RewTerm(func=mdp.stem_point_height_tanh, weight=0.3, params={"std": HEIGHT_STD})
    approach = RewTerm(func=mdp.approach_tanh, weight=0.5, params={"std": APPROACH_STD})
    curvature = RewTerm(
        func=mdp.curvature_penalty,
        weight=-1.0,
        params={"max_curvature": DAMAGE["max_curvature"], "curvature_budget": CURVATURE_BUDGET},
    )
    contact_force = RewTerm(
        func=mdp.contact_force_penalty,
        weight=-1.0,
        params={"threshold": CONTACT_FREE_FORCE, "sensor_cfg": SceneEntityCfg("stem_contact")},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # about -5 per termination in effect (x policy dt 0.032 s), so that continuing beats ending early; early
    # terminations only, not the time out (user, 2026-10-15)
    terminated = RewTerm(func=mdp.is_terminated, weight=-150.0)


@configclass
class TerminationsCfg:
    """Time out; damage limits (curvature, contact force); joint margin. No slot-base bound (user, 2026-10-09)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    curvature_limit = DoneTerm(func=mdp.curvature_limit, params={"max_curvature": DAMAGE["max_curvature"]})
    contact_force_limit = DoneTerm(
        func=mdp.contact_force_limit,
        params={"max_force": DAMAGE["max_contact_force"], "sensor_cfg": SceneEntityCfg("stem_contact")},
    )
    joint_margin = DoneTerm(
        func=mdp.joint_limit_margin,
        params={"min_margin": MIN_JOINT_MARGIN, "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )


@configclass
class StemPushPositionEnvCfg(ManagerBasedRLEnvCfg):
    """Stage 1: push the stem so that a selected stem point reaches a target position."""

    scene: StemPushSceneCfg = StemPushSceneCfg(num_envs=64, env_spacing=2.0)
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    stem_model: str = STEM_MODEL
    """Stem model the MDP reads the stem through (`mdp.stem_state`) and the base-fixing event uses
    (`stem_manip.assets.stem.STEM_MODELS`)."""
    git_commit: str = git_commit()

    def __post_init__(self):
        solver = stem_params(STEM_MODEL)["solver"]
        self.sim.dt = solver["sim_dt"]  # [s] the step the stem model was checked with (2 ms)
        self.decimation = 16  # policy at 1 / (16 * 2 ms) = 31.25 Hz
        self.sim.render_interval = self.decimation
        self.scene.stem_contact.history_length = self.decimation  # the contact terms average over one policy step
        self.sim.physics = stem_model(STEM_MODEL).physics_cfg()
        self.episode_length_s = 15.0  # [s] approach and push (user, 2026-10-09)
        # camera for the viewer and recorded videos (`--video`): env 0, robot and stem spawn area from the front side
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (1.6, 1.4, 1.0)
        self.viewer.lookat = (0.35, 0.0, 0.3)
