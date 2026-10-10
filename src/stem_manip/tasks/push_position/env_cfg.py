"""Manager-based env cfg for stage 1 (position control of the stem point).

Requirements:
- Scene: FR3 + fork (`stem_manip.assets.fr3.fr3_cfg("fork_v2")`), stem (`stem_model(name).stem_cfg()` from
  `stem_manip.assets.stem`), ground. Stem model `chain`: robot and stem in one PhysX scene (as in
  `scripts/check_contact.py`).
- Actions: relative tool-tip pose (position and rotation step, robot base frame), executed by Franka's Cartesian
  impedance law (`mdp.ToolTipImpedanceAction`, the same controller as on the real FR3); robot in torque mode.
- Managers from `mdp/` (commands, observations, rewards, terminations, events); weights and ranges as cfg fields.
- The stem point (segment index + offset, the tip) is a task-cfg field (`CommandsCfg`); the curvature limit comes from
  `assets/stem/stem.yaml`.

Implemented (M4, part 1): scene, 6-D impedance action, horizontal start pose, tool-tip pose observation, reset,
time out. Step 1 (2026-10-08): target command for the stem tip, stem-state observations (stem base, 5 points with
the previous policy step, target), obs 57. Step 2: stem spawn area (reset event `spawn_stem`). Step 3 (2026-10-09):
rewards and terminations, contact sensor on the stem (forces for rewards / terminations only, not observed).
Verify: `scripts/check_push_env.py`, `scripts/check_push_terms.py`; zero/random agent runs headless with few envs.
See docs/TODO.md -> M4.
"""

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

from stem_manip.assets.fr3 import fr3_cfg, tool_tip_offset
from stem_manip.assets.stem import stem_model, stem_params

from . import mdp

STEM_MODEL = "chain"
END_EFFECTOR = "fork_v2"
# Start pose (user, 2026-10-09; was (0.30, 0, 0.55) m, 2026-10-06): fork horizontal, pointing forward, tool tip at
# (0.40, 0, 0.50) m in the base frame, 10 cm above the stem tip and behind the stem area, so that less of the episode
# goes into the approach. Joint angles solved once with differential IK (2026-10-09); smallest distance to a joint
# limit 0.58 rad (joint 4).
START_JOINT_POS = {
    "fr3_joint1": -0.013,
    "fr3_joint2": -0.689,
    "fr3_joint3": 0.011,
    "fr3_joint4": -2.496,
    "fr3_joint5": 0.007,
    "fr3_joint6": 1.808,
    "fr3_joint7": 0.779,
}

# Stem spawn area (user, 2026-10-09): the stem base is reset uniformly within this rectangle on the lab's plate in front
# of the robot (58 x 58 cm at the mounting-surface height, x 0.075-0.655 m, y +-0.29 m), robot base frame [m]. Near
# edge from joint 4's limit (margin >= 0.25 rad with the fork horizontal: tool tip at x >= 0.40 m at the stem-tip
# height and >= 0.45 m lower down, and a target up to 10 cm towards the robot), far edge from the plate (2026-10-09; was
# x 0.275-0.525 m, 2026-10-06). Assumes robot base frame = env frame (robot root at the env origin, not rotated). The
# plate is not modelled as geometry (infinite ground plane); the fork may leave it.
STEM_SPAWN_X = (0.50, 0.65)
STEM_SPAWN_Y = (-0.15, 0.15)
# The spawn event ignores the stem's `init_state.pos`: to place the stem elsewhere, also set `events.spawn_stem = None`.

# Rewards and terminations (user, 2026-10-09; docs/notes/2026-10-09.md). Damage limits come from `stem.yaml`.
DAMAGE = stem_params(STEM_MODEL)["damage"]
DISTANCE_STD = (0.05, 0.01)  # [m] stem tip to target: coarse (targets 3-10 cm away), fine (success within 1 cm)
# [m] height error alone. One term, no coarse / fine pair (user, 2026-10-09): height errors span only ~0-23 mm (bowl
# drop up to ~15 mm + up to 8.3 mm deeper); the side-push gap of 0-8.3 mm lies on this tanh's slope, and larger
# height errors occur only before the push, where the 3-D coarse term already gives the gradient.
HEIGHT_STD = 0.003
APPROACH_STD = 0.1  # [m] tool tip to the nearest stem point (start: the fork is 0.2-0.4 m from the stem)
CURVATURE_SOFT_FRACTION = 0.8  # curvature penalty above 0.8 x the limit (= the targets' curvature budget)
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
        prim_path="{ENV_REGEX_NS}/Robot", init_state=ArticulationCfg.InitialStateCfg(joint_pos=START_JOINT_POS)
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
    """Relative tool-tip pose (at most `max_step` / `max_rot_step` per policy step), Franka's Cartesian impedance law."""

    tool_tip = mdp.ToolTipImpedanceActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        body_name=END_EFFECTOR,
        tool_offset=tool_tip_offset(END_EFFECTOR),
    )


@configclass
class CommandsCfg:
    """Target position of the stem tip, sampled once per episode (decided values: cfg defaults, user 2026-10-08)."""

    # resampling time far above the episode length: the target is sampled only at reset
    stem_target = mdp.StemTipTargetCommandCfg(stem_model=STEM_MODEL, resampling_time_range=(1e9, 1e9), debug_vis=True)


@configclass
class ObservationsCfg:
    """Policy observations, 57 values in this order: tool tip (position 3, orientation 6, applied step 6, target
    offset 6), stem base 3, 5 stem points of the previous and the current policy step 30 (oldest first), target 3.
    All positions in the robot base frame [m]. The stem-state terms (`mdp/stem_state.py`) are to be changed to what
    the perception delivers; no contact forces (decision 2026-10-06)."""

    @configclass
    class PolicyCfg(ObsGroup):
        tool_tip_pos = ObsTerm(
            func=mdp.tool_tip_pos, params={"body_name": END_EFFECTOR, "offset": tool_tip_offset(END_EFFECTOR)}
        )
        tool_tip_rot = ObsTerm(
            func=mdp.tool_tip_rot6d, params={"body_name": END_EFFECTOR, "offset": tool_tip_offset(END_EFFECTOR)}
        )
        applied_step = ObsTerm(func=mdp.applied_step)
        target_offset = ObsTerm(func=mdp.target_offset)
        stem_base = ObsTerm(func=mdp.stem_points, params={"arc_lengths": (0.0,), "model": STEM_MODEL})
        # 5 points up to the tip, plus the same points one policy step earlier (the stem's motion; user, 2026-10-08)
        stem_points = ObsTerm(
            func=mdp.stem_points,
            params={"arc_lengths": (0.08, 0.16, 0.24, 0.32, 0.40), "model": STEM_MODEL},
            history_length=2,
        )
        target_pos = ObsTerm(func=mdp.generated_commands, params={"command_name": "stem_target"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Base clamping at startup; robot and stem back to their default states on reset, then the stem base moved to a
    random place in the spawn area (upright, at rest; before the target command is resampled)."""

    fix_stem_base = EventTerm(func=mdp.fix_stem_base, mode="startup", params={"model": STEM_MODEL})
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    spawn_stem = EventTerm(
        func=mdp.reset_stem_base_uniform,
        mode="reset",
        params={"x_range": STEM_SPAWN_X, "y_range": STEM_SPAWN_Y, "asset_cfg": SceneEntityCfg("stem")},
    )


@configclass
class RewardsCfg:
    """Task (stem tip to target), approach (fork to stem), damage penalties below the limits, smooth motion, early
    terminations. Weights: a first guess for the baseline (user, 2026-10-09)."""

    distance_coarse = RewTerm(func=mdp.stem_point_distance_tanh, weight=1.0, params={"std": DISTANCE_STD[0]})
    # ellipsoid (user, 2026-10-10): height error scaled to std HEIGHT_STD (3 mm), so that a side push stalling on the
    # bowl a few mm above the target loses most of this term (5 mm: 7 % left instead of 54 % with the 1 cm sphere)
    distance_fine = RewTerm(
        func=mdp.stem_point_distance_tanh,
        weight=1.0,
        params={"std": DISTANCE_STD[1], "z_scale": DISTANCE_STD[1] / HEIGHT_STD},
    )
    # off for the baseline: switch on if the policy stalls on the bowl (side push, `height_error` metric stays > 0)
    # exp/m5-weekend run 04: on (weight 1.0), `height_error` stayed at +3 mm in runs 01-03
    height = RewTerm(func=mdp.stem_point_height_tanh, weight=1.0, params={"std": HEIGHT_STD})
    approach = RewTerm(
        func=mdp.approach_tanh,
        weight=0.5,
        params={
            "std": APPROACH_STD,
            "model": STEM_MODEL,
            "body_name": END_EFFECTOR,
            "offset": tool_tip_offset(END_EFFECTOR),
        },
    )
    curvature = RewTerm(
        func=mdp.curvature_penalty,
        weight=-1.0,
        params={
            "model": STEM_MODEL,
            "max_curvature": DAMAGE["max_curvature"],
            "soft_fraction": CURVATURE_SOFT_FRACTION,
        },
    )
    contact_force = RewTerm(
        func=mdp.contact_force_penalty,
        weight=-1.0,
        params={"threshold": CONTACT_FREE_FORCE, "sensor_cfg": SceneEntityCfg("stem_contact")},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    terminated = RewTerm(func=mdp.is_terminated, weight=-10.0)


@configclass
class TerminationsCfg:
    """Time out; damage limits (curvature, contact force); joint margin. No tool-tip bound (user, 2026-10-09)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    curvature_limit = DoneTerm(
        func=mdp.curvature_limit, params={"model": STEM_MODEL, "max_curvature": DAMAGE["max_curvature"]}
    )
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
    git_commit: str = git_commit()

    def __post_init__(self):
        solver = stem_params(STEM_MODEL)["solver"]
        self.sim.dt = solver["sim_dt"]  # [s] the step the stem model was checked with (2 ms)
        self.decimation = 16  # policy at 1 / (16 * 2 ms) = 31.25 Hz
        self.sim.render_interval = self.decimation
        self.scene.stem_contact.history_length = self.decimation  # the contact terms average over one policy step
        self.sim.physics = stem_model(STEM_MODEL).physics_cfg()
        self.episode_length_s = 15.0  # [s] approach and push (user, 2026-10-09; was 10 s)
        # camera for the viewer and recorded videos (`--video`): env 0, robot and stem spawn area from the front side
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (1.6, 1.4, 1.0)
        self.viewer.lookat = (0.35, 0.0, 0.3)
