"""Manager-based env cfg for stage 1 (position control of the stem point).

Requirements:
- Scene: FR3 + fork (`stem_manip.assets.fr3.fr3_cfg("fork_v2")`), stem (`stem_model(name).stem_cfg()` from
  `stem_manip.assets.stem`), ground. Stem model `chain`: robot and stem in one PhysX scene (as in
  `scripts/check_contact.py`).
- Actions: relative tool-tip pose (position and rotation step, robot base frame), executed by Franka's Cartesian
  impedance law (`mdp.ToolTipImpedanceAction`, the same controller as on the real FR3); robot in torque mode.
- Managers from `mdp/` (commands, observations, rewards, terminations, events); weights and ranges as cfg fields.
- The point of interest (segment index + offset) and the curvature limit come from cfg / `assets/stem/stem.yaml`.

Implemented (M4, part 1): scene, 6-D impedance action, horizontal start pose, tool-tip pose observation, reset,
time out.
Verify: `scripts/check_push_env.py`; zero/random agent runs headless with few envs. See docs/TODO.md -> M4.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from stem_manip.assets.fr3 import fr3_cfg, tool_tip_offset
from stem_manip.assets.stem import stem_model, stem_params

from . import mdp

STEM_MODEL = "chain"
END_EFFECTOR = "fork_v2"
# Start pose (user, 2026-10-06): fork horizontal, pointing forward, tool tip at (0.30, 0, 0.55) m in the base frame,
# above and behind the stem area. Joint angles solved once with differential IK (2026-10-07); smallest distance to a
# joint limit 0.54 rad (joint 4).
START_JOINT_POS = {
    "fr3_joint1": -0.013,
    "fr3_joint2": -1.013,
    "fr3_joint3": 0.008,
    "fr3_joint4": -2.540,
    "fr3_joint5": 0.007,
    "fr3_joint6": 1.527,
    "fr3_joint7": 0.776,
}


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


@configclass
class ActionsCfg:
    """Relative tool-tip pose (at most `max_step` / `max_rot_step` per policy step), Franka's Cartesian impedance law."""

    tool_tip = mdp.ToolTipImpedanceActionCfg(
        asset_name="robot",
        joint_names=["fr3_joint[1-7]"],
        body_name=END_EFFECTOR,
        tool_offset=tool_tip_offset(END_EFFECTOR),
    )


@configclass
class ObservationsCfg:
    """Policy observations (tool-tip pose and last action so far; stem state and target follow)."""

    @configclass
    class PolicyCfg(ObsGroup):
        tool_tip_pos = ObsTerm(
            func=mdp.tool_tip_pos, params={"body_name": END_EFFECTOR, "offset": tool_tip_offset(END_EFFECTOR)}
        )
        tool_tip_rot = ObsTerm(
            func=mdp.tool_tip_rot6d, params={"body_name": END_EFFECTOR, "offset": tool_tip_offset(END_EFFECTOR)}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Base clamping at startup; robot and stem back to their default states on reset."""

    fix_stem_base = EventTerm(func=mdp.fix_stem_base, mode="startup", params={"model": STEM_MODEL})
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})


@configclass
class RewardsCfg:
    """No reward terms yet (step 3)."""


@configclass
class TerminationsCfg:
    """Time out only (curvature limit and out of bounds: step 3)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class StemPushPositionEnvCfg(ManagerBasedRLEnvCfg):
    """Stage 1: push the stem so that a selected stem point reaches a target position."""

    scene: StemPushSceneCfg = StemPushSceneCfg(num_envs=64, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        solver = stem_params(STEM_MODEL)["solver"]
        self.sim.dt = solver["sim_dt"]  # [s] the step the stem model was checked with (2 ms)
        self.decimation = 16  # policy at 1 / (16 * 2 ms) = 31.25 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics = stem_model(STEM_MODEL).physics_cfg()
        self.episode_length_s = 10.0
